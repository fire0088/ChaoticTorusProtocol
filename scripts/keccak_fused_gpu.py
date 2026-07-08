"""
Fully GPU-resident K12 chunk absorption. Replaces the previous approach
(one host<->device round trip PER BLOCK, ~49 per chunk group) with exactly
ONE upload, ONE kernel launch, and ONE download for the ENTIRE multi-block
absorption of a whole batch of chunks -- the per-block coordination
overhead identified as the remaining bottleneck after the two earlier
fixes (incomplete batching, kernel recompilation) is eliminated by never
leaving the GPU until every chunk in the group has finished all its rounds.

Feasible cleanly because the rate (168 bytes) divides evenly into 21
64-bit lanes, matching Keccak's native lane representation exactly --
absorption is pure lane-XOR, no byte-level conversion needed inside the
kernel at all. The block loop (sequential, ~49 iterations for CTP's real
chunk size) and the 12-round permutation within each block both run
entirely in Taichi local (register/stack) memory per thread; only the
final state after all blocks is written back to global memory.

Validated in stages against keccak_p12_gpu.py (already validated against
the CPU reference) before being trusted, same discipline as everywhere
else in this project:
  Stage A: fused single-permutation-only kernel (n_blocks=1, no absorption)
           vs. the existing batched kernel, to isolate correctness of the
           local-vector round function from the absorption loop.
  Stage B: full multi-block absorption vs. batched_k12_sessions.py's
           (already-correct, round-trip-per-block) implementation.
"""

import numpy as np
import taichi as ti

from gpu_backend import detect_best_backend
import keccak_reference as ref

# Import keccak_p12_gpu FIRST -- it calls ti.init() at module level, and a
# second, independent ti.init() call (which this module used to make
# separately) resets Taichi's runtime, invalidating any fields declared
# before it. Found this the direct way: rc_field looked valid at
# declaration time but errored as a "0-dim field" the moment Stage A's
# comparison imported keccak_p12_gpu afterward and triggered a second
# init. Fix: let that import happen first, and never call ti.init() a
# second time in this process.
from keccak_p12_gpu import keccak_p12_batch, numpy_states_to_taichi_layout, _BACKEND_NAME

RC_NP = np.array(ref.RC, dtype=np.uint64)
rc_field = ti.field(dtype=ti.u64, shape=24)
rc_field.from_numpy(RC_NP)

R_OFFSETS_NP = np.array(ref.R_OFFSETS, dtype=np.int32)  # R_OFFSETS[x][y]
r_offsets_field = ti.field(dtype=ti.i32, shape=(5, 5))
r_offsets_field.from_numpy(R_OFFSETS_NP)

RATE_BYTES = 168
RATE_LANES = 21  # 168 / 8, exact


@ti.func
def rol64(x: ti.u64, n: ti.i32) -> ti.u64:
    result = x
    if n != 0:
        result = (x << ti.u64(n)) | (x >> ti.u64(64 - n))
    return result


@ti.kernel
def keccak_p12_fused(
    block_lanes: ti.template(),  # (N, n_blocks, 21) ti.u64 -- all input data, uploaded ONCE
    out_states: ti.template(),   # (N, 5, 5) ti.u64 -- final states, downloaded ONCE
    N: ti.i32,
    n_blocks: ti.i32,
):
    for b in range(N):
        # Local (per-thread) state -- lives in registers/local memory for
        # the ENTIRE absorption, never touching global memory until the
        # final write at the very end of this thread's work.
        s = ti.Vector([ti.u64(0)] * 25)

        for blk in range(n_blocks):
            # Absorb: pure lane XOR, no byte conversion needed (rate=21 lanes exactly).
            for i in ti.static(range(RATE_LANES)):
                s[i] ^= block_lanes[b, blk, i]

            # 12-round Keccak-p permutation, entirely on the local vector s.
            for rnd in range(12, 24):  # last 12 of 24 round constants
                # theta
                c = ti.Vector([ti.u64(0)] * 5)
                for x in ti.static(range(5)):
                    c[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20]
                d = ti.Vector([ti.u64(0)] * 5)
                for x in ti.static(range(5)):
                    d[x] = c[(x - 1) % 5] ^ rol64(c[(x + 1) % 5], 1)
                for x in ti.static(range(5)):
                    for y in ti.static(range(5)):
                        s[x + 5 * y] ^= d[x]

                # rho + pi
                t = ti.Vector([ti.u64(0)] * 25)
                for x in ti.static(range(5)):
                    for y in ti.static(range(5)):
                        nx = y
                        ny = (2 * x + 3 * y) % 5
                        t[nx + 5 * ny] = rol64(s[x + 5 * y], ref.R_OFFSETS[x][y])

                # chi
                for x in ti.static(range(5)):
                    for y in ti.static(range(5)):
                        a = t[((x + 1) % 5) + 5 * y]
                        bb = t[((x + 2) % 5) + 5 * y]
                        s[x + 5 * y] = t[x + 5 * y] ^ ((~a) & bb)

                # iota
                s[0] ^= rc_field[rnd]

        for x in ti.static(range(5)):
            for y in ti.static(range(5)):
                out_states[b, x, y] = s[x + 5 * y]


@ti.kernel
def keccak_p12_fused_absorb_squeeze(
    block_lanes: ti.template(),     # (N, n_absorb_blocks, 21) ti.u64 -- input, uploaded ONCE
    out_squeezed: ti.template(),    # (N, n_squeeze_blocks, 21) ti.u64 -- output, downloaded ONCE
    N: ti.i32,
    n_absorb_blocks: ti.i32,
    n_squeeze_blocks: ti.i32,
):
    """General fused absorb-then-squeeze, for the final-node computation
    (domain 0x06), which unlike the leaf/chaining-value computation needs
    a squeeze phase producing potentially many rate-sized output blocks
    (CTP's keystream request sizes exceed one 168-byte block). Batches
    across N independent sessions' final-node computations the same way
    keccak_p12_fused batches chaining values -- this is the piece that was
    missing: the leaf computation was fused, but the final node that
    combines the leaves into the actual output was not, and dominated
    runtime once measured as part of the complete pipeline rather than
    in isolation."""
    for b in range(N):
        s = ti.Vector([ti.u64(0)] * 25)

        for blk in range(n_absorb_blocks):
            for i in ti.static(range(RATE_LANES)):
                s[i] ^= block_lanes[b, blk, i]
            for rnd in range(12, 24):
                c = ti.Vector([ti.u64(0)] * 5)
                for x in ti.static(range(5)):
                    c[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20]
                d = ti.Vector([ti.u64(0)] * 5)
                for x in ti.static(range(5)):
                    d[x] = c[(x - 1) % 5] ^ rol64(c[(x + 1) % 5], 1)
                for x in ti.static(range(5)):
                    for y in ti.static(range(5)):
                        s[x + 5 * y] ^= d[x]
                t = ti.Vector([ti.u64(0)] * 25)
                for x in ti.static(range(5)):
                    for y in ti.static(range(5)):
                        nx = y
                        ny = (2 * x + 3 * y) % 5
                        t[nx + 5 * ny] = rol64(s[x + 5 * y], ref.R_OFFSETS[x][y])
                for x in ti.static(range(5)):
                    for y in ti.static(range(5)):
                        a = t[((x + 1) % 5) + 5 * y]
                        bb = t[((x + 2) % 5) + 5 * y]
                        s[x + 5 * y] = t[x + 5 * y] ^ ((~a) & bb)
                s[0] ^= rc_field[rnd]

        for sq in range(n_squeeze_blocks):
            for i in ti.static(range(RATE_LANES)):
                out_squeezed[b, sq, i] = s[i]
            if sq < n_squeeze_blocks - 1:
                for rnd in range(12, 24):
                    c = ti.Vector([ti.u64(0)] * 5)
                    for x in ti.static(range(5)):
                        c[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20]
                    d = ti.Vector([ti.u64(0)] * 5)
                    for x in ti.static(range(5)):
                        d[x] = c[(x - 1) % 5] ^ rol64(c[(x + 1) % 5], 1)
                    for x in ti.static(range(5)):
                        for y in ti.static(range(5)):
                            s[x + 5 * y] ^= d[x]
                    t = ti.Vector([ti.u64(0)] * 25)
                    for x in ti.static(range(5)):
                        for y in ti.static(range(5)):
                            nx = y
                            ny = (2 * x + 3 * y) % 5
                            t[nx + 5 * ny] = rol64(s[x + 5 * y], ref.R_OFFSETS[x][y])
                    for x in ti.static(range(5)):
                        for y in ti.static(range(5)):
                            a = t[((x + 1) % 5) + 5 * y]
                            bb = t[((x + 2) % 5) + 5 * y]
                            s[x + 5 * y] = t[x + 5 * y] ^ ((~a) & bb)
                    s[0] ^= rc_field[rnd]


def _lanes_to_bytes_vectorized(lanes: np.ndarray) -> np.ndarray:
    """Inverse of the reshape used in _prepare_block_lanes: (N, n_blocks, 21)
    uint64 -> (N, n_blocks * 168) uint8, vectorized."""
    N, n_blocks, _ = lanes.shape
    le = lanes.astype("<u8")
    return le.view(np.uint8).reshape(N, n_blocks * RATE_BYTES)


def _states_to_bytes_batch(states: np.ndarray) -> np.ndarray:
    """Vectorized replacement for calling ref.state_to_bytes() once per
    chunk in a Python loop (a second, slower-than-necessary bottleneck
    found the same way as _prepare_block_lanes' -- profiling showed this
    reassembly step at 25-30% of total pipeline time at every N tested,
    the next-largest phase after fixing the lane-conversion bottleneck).
    states: (Ng, 5, 5) uint64, states[i, x, y]. Returns (Ng, 200) uint8,
    matching keccak_reference.state_to_bytes's per-state layout (byte
    offset = 8*(x+5*y)) for the whole batch in one operation."""
    Ng = states.shape[0]
    lanes_ordered = states.transpose(0, 2, 1).reshape(Ng, 25)  # index = x + 5*y
    lanes_le = lanes_ordered.astype("<u8")
    return lanes_le.view(np.uint8).reshape(Ng, 200)


def _vectorized_chunk_and_pad(materials: list, rate_bytes: int = RATE_BYTES, chunk_size: int = None):
    """
    Vectorized replacement for the per-session Python loop that built S0
    and chunk lists. Requires all materials to be the SAME length --
    CTP's realistic case, since packets in a burst share a fixed size.

    Key insight this relies on: K12's pad10star1 padding depends only on
    a chunk's LENGTH and domain, never its content. So for same-length
    sessions, every corresponding chunk needs IDENTICAL padding bytes --
    computed once and broadcast across the batch via numpy, rather than
    calling pad10star1 once per chunk in a Python loop.

    v2: the first version of this function reintroduced the exact
    b"".join() bottleneck already found and fixed in _prepare_block_lanes
    (found again by direct timing, not assumed), and additionally
    concatenated the 1-byte length-encoding suffix onto the ENTIRE
    materials array (copying the full ~32KB/session array just to append
    one byte) instead of only where the suffix actually lands -- inside
    the final, much shorter chunk. Fixed: no join(), and the suffix is
    only ever concatenated onto the small last-chunk piece.

    Returns: S0 (N, chunk_size) uint8, and a dict {padded_length: array}
    where array has shape (N * n_chunks_at_that_length, padded_length)
    for "full" chunks or (N, padded_length) for a final shorter chunk.
    """
    import k12_full as k12
    if chunk_size is None:
        chunk_size = k12.CHUNK_SIZE

    N = len(materials)
    msg_len = len(materials[0])
    assert all(len(m) == msg_len for m in materials), \
        "_vectorized_chunk_and_pad requires uniform-length materials"
    assert msg_len >= chunk_size, \
        "this fast path assumes S0 comes entirely from material bytes (true at CTP's real packet size)"

    mat_arr = np.empty((N, msg_len), dtype=np.uint8)
    for i, m in enumerate(materials):
        mat_arr[i] = np.frombuffer(m, dtype=np.uint8)

    suffix = k12.length_encode(0)  # 1 byte, but kept general
    suffix_arr = np.frombuffer(suffix, dtype=np.uint8)

    S0 = mat_arr[:, :chunk_size].copy()
    remaining = mat_arr[:, chunk_size:]  # material bytes only; suffix appended below where it lands
    remaining_material_len = remaining.shape[1]
    remaining_total_len = remaining_material_len + len(suffix_arr)  # what the suffix conceptually extends

    n_full_chunks = remaining_total_len // chunk_size
    last_chunk_len = remaining_total_len % chunk_size
    # full chunks come entirely from material (verified: n_full_chunks*chunk_size <= remaining_material_len
    # whenever the suffix is shorter than one chunk, true here since suffix is 1 byte)
    full_chunks_material_len = n_full_chunks * chunk_size

    groups = {}
    if n_full_chunks > 0:
        full_chunks = remaining[:, :full_chunks_material_len].reshape(N, n_full_chunks, chunk_size)
        pad_full = np.frombuffer(ref.pad10star1(rate_bytes, chunk_size, 0x0B), dtype=np.uint8)
        padded_len = chunk_size + len(pad_full)
        padded_full = np.empty((N, n_full_chunks, padded_len), dtype=np.uint8)
        padded_full[:, :, :chunk_size] = full_chunks
        padded_full[:, :, chunk_size:] = pad_full
        groups[padded_len] = ("full", n_full_chunks, padded_full.reshape(N * n_full_chunks, padded_len))
    if last_chunk_len > 0:
        last_material_len = last_chunk_len - len(suffix_arr)
        last_chunk_material = remaining[:, full_chunks_material_len:full_chunks_material_len + last_material_len]
        pad_last = np.frombuffer(ref.pad10star1(rate_bytes, last_chunk_len, 0x0B), dtype=np.uint8)
        padded_len = last_chunk_len + len(pad_last)
        padded_last = np.empty((N, padded_len), dtype=np.uint8)
        padded_last[:, :last_material_len] = last_chunk_material
        padded_last[:, last_material_len:last_chunk_len] = suffix_arr
        padded_last[:, last_chunk_len:] = pad_last
        groups[padded_len] = ("last", 1, padded_last)

    return S0, groups, n_full_chunks, (1 if last_chunk_len > 0 else 0)


def _lanes_from_padded_array(padded_arr: np.ndarray) -> tuple:
    """Like _prepare_block_lanes, but takes an already-assembled (Ng,
    padded_len) uint8 array directly -- no per-chunk Python loop at all,
    since _vectorized_chunk_and_pad already produced one array for the
    whole group in a single numpy operation."""
    Ng, total_len = padded_arr.shape
    n_blocks = total_len // RATE_BYTES
    reshaped = padded_arr.reshape(Ng, n_blocks, RATE_LANES, 8)
    lanes = reshaped.view("<u8").reshape(Ng, n_blocks, RATE_LANES)
    return lanes, n_blocks


def _prepare_block_lanes(padded_group: list) -> np.ndarray:
    """Host-side (numpy, vectorized, done ONCE per call -- not per block):
    convert a list of equal-length padded byte chunks into (N, n_blocks, 21)
    uint64 lane arrays ready to upload.

    v2 -- profiling traced ~50% of total pipeline time (at every N tested,
    from 512 to 8192 sessions) to THIS function, not the GPU kernel, which
    is what produced the flat ~1.7-1.8x-slower ratio that looked like GPU
    hardware saturation but wasn't. Two real, separate causes, found by
    timing each line rather than guessing:
      1. `.astype(np.uint64)` was being called on data that `.view("<u8")`
         already produces as dtype uint64 on this platform (confirmed:
         `.dtype` reports uint64 BEFORE the astype call) -- the astype was
         a pure no-op full-array copy, costing ~63ms at N=24576 chunks for
         zero effect on the result. Removed.
      2. `b"".join(padded_group)` concatenating thousands of separate
         Python bytes objects cost ~83ms at the same scale -- fixed by
         writing directly into a preallocated numpy buffer instead of
         building an intermediate Python bytes object at all.
    """
    N = len(padded_group)
    total_len = len(padded_group[0])
    n_blocks = total_len // RATE_BYTES
    flat = np.empty((N, total_len), dtype=np.uint8)
    for i, p in enumerate(padded_group):
        flat[i] = np.frombuffer(p, dtype=np.uint8)
    reshaped = flat.reshape(N, n_blocks, RATE_LANES, 8)
    lanes = reshaped.view("<u8").reshape(N, n_blocks, RATE_LANES)
    return lanes, n_blocks


if __name__ == "__main__":
    import os

    print(f"Taichi backend in use: {_BACKEND_NAME}"
          + ("" if _BACKEND_NAME != "cpu" else " (no GPU backend found -- correctness only)"))

    # ---- Stage A: fused single-permutation kernel vs. the existing batched kernel ----
    print("\n=== Stage A: fused kernel (1 block, no absorption) vs. keccak_p12_gpu ===")
    # (keccak_p12_batch, numpy_states_to_taichi_layout already imported at module level)

    N = 32
    random_states = []
    for _ in range(N):
        s = [[int(np.random.randint(-(2**63), 2**63, dtype=np.int64)) & 0xFFFFFFFFFFFFFFFF
              for _ in range(5)] for _ in range(5)]
        random_states.append(s)

    # Existing (already-validated) batched kernel, direct permutation, no absorption.
    ti_states = numpy_states_to_taichi_layout(random_states)
    field_existing = ti.field(dtype=ti.u64, shape=(N, 5, 5))
    field_existing.from_numpy(ti_states)
    keccak_p12_batch(field_existing, 12)
    expected = field_existing.to_numpy()

    # Fused kernel: feed the SAME initial state as a single "absorbed block"
    # (i.e., treat the state itself as if it were XORed in as one block from
    # an all-zero start, which is mathematically the same operation).
    lanes_in = np.zeros((N, 1, 25), dtype=np.uint64)  # note: 25 here, not 21, for this isolated test
    for i in range(N):
        for x in range(5):
            for y in range(5):
                lanes_in[i, 0, x + 5 * y] = random_states[i][x][y]
    # Stage A uses all 25 lanes as a direct state-load test, so temporarily
    # widen RATE_LANES usage for this specific check only:
    block_field = ti.field(dtype=ti.u64, shape=(N, 1, 25))
    block_field.from_numpy(lanes_in)
    out_field = ti.field(dtype=ti.u64, shape=(N, 5, 5))

    @ti.kernel
    def _stage_a_kernel(block_lanes: ti.template(), out_states: ti.template(), n: ti.i32):
        for b in range(n):
            s = ti.Vector([ti.u64(0)] * 25)
            for i in ti.static(range(25)):
                s[i] ^= block_lanes[b, 0, i]
            for rnd in range(12, 24):
                c = ti.Vector([ti.u64(0)] * 5)
                for x in ti.static(range(5)):
                    c[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20]
                d = ti.Vector([ti.u64(0)] * 5)
                for x in ti.static(range(5)):
                    d[x] = c[(x - 1) % 5] ^ rol64(c[(x + 1) % 5], 1)
                for x in ti.static(range(5)):
                    for y in ti.static(range(5)):
                        s[x + 5 * y] ^= d[x]
                t = ti.Vector([ti.u64(0)] * 25)
                for x in ti.static(range(5)):
                    for y in ti.static(range(5)):
                        nx = y
                        ny = (2 * x + 3 * y) % 5
                        t[nx + 5 * ny] = rol64(s[x + 5 * y], ref.R_OFFSETS[x][y])
                for x in ti.static(range(5)):
                    for y in ti.static(range(5)):
                        a = t[((x + 1) % 5) + 5 * y]
                        bbb = t[((x + 2) % 5) + 5 * y]
                        s[x + 5 * y] = t[x + 5 * y] ^ ((~a) & bbb)
                s[0] ^= rc_field[rnd]
            for x in ti.static(range(5)):
                for y in ti.static(range(5)):
                    out_states[b, x, y] = s[x + 5 * y]

    _stage_a_kernel(block_field, out_field, N)
    got = out_field.to_numpy()
    stage_a_ok = np.array_equal(got, expected)
    print("Stage A (local-vector round function correctness): " + ("MATCH" if stage_a_ok else "MISMATCH"))

    if not stage_a_ok:
        print("STOPPING -- do not proceed to Stage B until Stage A passes.")
    else:
        # ---- Stage B: full multi-block absorption vs. the round-trip-per-block version ----
        print("\n=== Stage B: fully fused multi-block absorption vs. batched_k12_sessions ===")
        from batched_k12_sessions import batched_k12_many_sessions
        import k12_full as k12

        N_sessions = 12
        materials = [os.urandom(32800) for _ in range(N_sessions)]  # CTP's real per-packet size
        out_len = 4096

        # Reference: the already-validated (correct, round-trip-per-block) version.
        reference_results = [k12.kangarootwelve(m, out_len) for m in materials]

        # Fused version: build the chunk groups the same way, then run the
        # ENTIRE absorption for each length-group as one kernel call.
        session_S0, session_chunks = [], []
        for material in materials:
            S = material + k12.length_encode(0)
            S0 = S[:k12.CHUNK_SIZE]
            chunks = [S[i:i + k12.CHUNK_SIZE] for i in range(k12.CHUNK_SIZE, len(S), k12.CHUNK_SIZE)]
            session_S0.append(S0)
            session_chunks.append(chunks)

        flat_chunks, chunk_owner = [], []
        for sidx, chunks in enumerate(session_chunks):
            for c in chunks:
                flat_chunks.append(c)
                chunk_owner.append(sidx)

        padded_chunks = [c + ref.pad10star1(RATE_BYTES, len(c), 0x0B) for c in flat_chunks]
        groups = {}
        for idx, p in enumerate(padded_chunks):
            groups.setdefault(len(p), []).append((idx, p))

        cvs_flat = [None] * len(flat_chunks)
        _fused_field_cache = {}

        for padded_len, items in groups.items():
            indices = [i for i, _ in items]
            group_padded = [p for _, p in items]
            lanes, n_blocks = _prepare_block_lanes(group_padded)
            Ng = lanes.shape[0]

            key = (Ng, n_blocks)
            if key not in _fused_field_cache:
                _fused_field_cache[key] = (
                    ti.field(dtype=ti.u64, shape=(Ng, n_blocks, RATE_LANES)),
                    ti.field(dtype=ti.u64, shape=(Ng, 5, 5)),
                )
            block_f, out_f = _fused_field_cache[key]
            block_f.from_numpy(lanes)
            keccak_p12_fused(block_f, out_f, Ng, n_blocks)
            final_states = out_f.to_numpy()

            for local_i, flat_idx in enumerate(indices):
                state = final_states[local_i]
                sb = ref.state_to_bytes([[int(state[x, y]) for y in range(5)] for x in range(5)])
                cvs_flat[flat_idx] = sb[:32]

        cvs_by_session = [[] for _ in materials]
        for owner, cv in zip(chunk_owner, cvs_flat):
            cvs_by_session[owner].append(cv)

        fused_results = []
        for S0, cvs in zip(session_S0, cvs_by_session):
            final_input = S0 + b"\x03" + b"\x00" * 7 + b"".join(cvs) + k12.length_encode(len(cvs)) + b"\xFF\xFF"
            fused_results.append(k12._f_with_domain(final_input, domain=0x06, out_len=out_len))

        stage_b_ok = all(a == b for a, b in zip(fused_results, reference_results))
        print("Stage B (full fused multi-session K12 extraction): " + ("MATCH" if stage_b_ok else "MISMATCH"))

        # ---- Stage C: fuse the FINAL NODE too (this was the missing piece --
        # Stage B still called k12._f_with_domain for the final node, which
        # routes through the slow pure-Python reference and dominated
        # end-to-end runtime once measured as a complete pipeline rather
        # than in isolation; see benchmark_fused_k12.py's investigation). ----
        print("\n=== Stage C: fully fused final-node combination vs. reference ===")
        final_inputs = []
        for S0, cvs in zip(session_S0, cvs_by_session):
            final_inputs.append(S0 + b"\x03" + b"\x00" * 7 + b"".join(cvs) + k12.length_encode(len(cvs)) + b"\xFF\xFF")

        padded_finals = [fi + ref.pad10star1(RATE_BYTES, len(fi), 0x06) for fi in final_inputs]
        assert all(len(p) == len(padded_finals[0]) for p in padded_finals), \
            "test assumes equal-length sessions; production code must group by length as in Stage B"

        absorb_lanes, n_absorb_blocks = _prepare_block_lanes(padded_finals)
        n_squeeze_blocks = (out_len + RATE_BYTES - 1) // RATE_BYTES
        Nc = absorb_lanes.shape[0]

        absorb_field = ti.field(dtype=ti.u64, shape=(Nc, n_absorb_blocks, RATE_LANES))
        squeeze_field = ti.field(dtype=ti.u64, shape=(Nc, n_squeeze_blocks, RATE_LANES))
        absorb_field.from_numpy(absorb_lanes)
        keccak_p12_fused_absorb_squeeze(absorb_field, squeeze_field, Nc, n_absorb_blocks, n_squeeze_blocks)
        squeezed_lanes = squeeze_field.to_numpy()
        squeezed_bytes = _lanes_to_bytes_vectorized(squeezed_lanes)
        stage_c_results = [bytes(squeezed_bytes[i, :out_len]) for i in range(Nc)]

        stage_c_ok = all(a == b for a, b in zip(stage_c_results, reference_results))
        print("Stage C (fully fused final node, absorb+squeeze): " + ("MATCH" if stage_c_ok else "MISMATCH"))

        print("\n" + "=" * 60)
        overall_ok = stage_a_ok and stage_b_ok and stage_c_ok
        print("OVERALL: " + ("FULLY FUSED PIPELINE VALIDATED (leaf + final node)" if overall_ok
                              else "DO NOT TRUST -- MISMATCH DETECTED"))
