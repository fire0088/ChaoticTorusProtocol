"""
Batch KangarooTwelve keystream extraction across many CONCURRENT SESSIONS,
not just chunks within one session. Reuses keccak_p12_gpu.keccak_p12_batch
unchanged -- that kernel already takes an arbitrary batch of independent
Keccak-p[1600,12] states; it has no notion of "session" or "chunk", so
scaling the batch dimension from "4 chunks in one packet" to "4 chunks x
N sessions" requires no new kernel, only new orchestration.

Validated against running k12_full.kangarootwelve() independently per
session before being trusted.
"""

import os
import time

import numpy as np
import taichi as ti

import k12_full as k12
from keccak_p12_gpu import keccak_p12_batch, numpy_states_to_taichi_layout, _BACKEND_NAME
import keccak_reference as ref

# Persistent field cache, keyed by batch size N. This exists because of a
# real bug found via GPU testing: passing a FRESHLY ALLOCATED ti.field to
# a Taichi kernel retriggers expensive recompilation/retracing even when
# its shape exactly matches a field seen before -- confirmed directly:
# reusing the same field object measured ~0.1ms/call on this machine's
# CPU backend, while allocating a new field of the IDENTICAL shape on
# every call measured ~800ms/call, an ~8000x difference. That is what was
# producing the ~3.5s fixed floor at every N on the real GPU test (Taichi
# recompiling on every single invocation of batched_k12_many_sessions,
# since it allocated a fresh field internally every time). The fix is the
# same pattern batched_feistel_lattice.py already used correctly: allocate
# persistent fields once per shape and reuse them via .from_numpy(),
# never reallocate a field for a shape already seen.
_field_cache = {}


def _get_cached_field(n: int):
    if n not in _field_cache:
        _field_cache[n] = ti.field(dtype=ti.u64, shape=(n, 5, 5))
    return _field_cache[n]


def _bytes_to_lane_state(b: bytes):
    """Pad/absorb a <=168-byte block into a fresh 25-lane state (rate=168,
    capacity=256 bits), matching keccak_reference's bytes_to_state layout,
    for the FIRST absorption step of a short (<=1 block) input."""
    padded = b + b"\x00" * (200 - len(b))
    return ref.bytes_to_state(padded)


def _states_to_bytes_vectorized(states_np: np.ndarray) -> np.ndarray:
    """states_np: (N, 5, 5) uint64, lane[x,y]. Returns (N, 200) uint8,
    matching keccak_reference.state_to_bytes's byte layout (offset =
    8*(x+5*y), little-endian), fully vectorized -- no per-element Python
    loops, which is exactly what made the previous version catastrophically
    slow (see module docstring update below)."""
    N = states_np.shape[0]
    lanes_ordered = states_np.transpose(0, 2, 1).reshape(N, 25)  # index = x + 5*y
    lanes_le = lanes_ordered.astype("<u8")
    return lanes_le.view(np.uint8).reshape(N, 200)


def _bytes_to_states_vectorized(bytes_np: np.ndarray) -> np.ndarray:
    """Inverse of _states_to_bytes_vectorized. bytes_np: (N, 200) uint8."""
    N = bytes_np.shape[0]
    lanes_le = bytes_np.reshape(N, 25, 8).copy().view("<u8").reshape(N, 25)
    lanes_ordered = lanes_le.astype(np.uint64)
    return lanes_ordered.reshape(N, 5, 5).transpose(0, 2, 1)


def batched_k12_many_sessions(materials: list, out_len: int):
    """
    v2: fixed a real performance bug found via real GPU testing. The first
    version absorbed all-but-the-last rate-block of each chunk using our
    slow, pure-Python reference Keccak-f (meant for correctness validation,
    never for speed), and only batched the FINAL block's permutation on
    GPU. For CTP's real chunk size (8192 bytes, rate 168 bytes), that's
    ~48 slow Python permutation calls per chunk with only 1 GPU-batched
    call -- at N=128 sessions (~512 chunks), ~24,576 unaccelerated calls
    and only 512 accelerated ones. That is why the first version measured
    hundreds of times slower than pycryptodome: over 98% of the actual
    work never touched the GPU.

    Fixed by restructuring the loop: instead of iterating over chunks and
    sequentially absorbing each one's blocks in Python, group chunks by
    their padded length (K12 allows the final chunk to be shorter than
    8192 bytes, so lock-step batching only applies within a same-length
    group -- see the length-grouping logic below, added after the
    uniform-length assumption failed on real ~32800-byte CTP input) and,
    within each group, iterate over BLOCK INDEX rather than chunk index:
    at each block index, XOR that block into ALL chunks in the group at
    once (vectorized numpy) and run ONE batched GPU permutation call
    across the whole group simultaneously. Every chunk in a group advances
    through its absorption in lockstep. Total GPU kernel calls: roughly
    (number of distinct chunk lengths) x (blocks per chunk), independent
    of how many chunks/sessions are in each group. Total slow Python
    Keccak-f calls: zero.
    """
    session_S0 = []
    session_chunks = []
    for material in materials:
        S = material + k12.length_encode(0)
        S0 = S[:k12.CHUNK_SIZE]
        chunks = [S[i:i + k12.CHUNK_SIZE] for i in range(k12.CHUNK_SIZE, len(S), k12.CHUNK_SIZE)]
        session_S0.append(S0)
        session_chunks.append(chunks)

    flat_chunks = []
    chunk_owner = []
    for s, chunks in enumerate(session_chunks):
        for c in chunks:
            flat_chunks.append(c)
            chunk_owner.append(s)

    if len(flat_chunks) == 0:
        return [k12._f_with_domain(S0, domain=0x07, out_len=out_len) for S0 in session_S0]

    rate = 168
    domain = 0x0B
    # NOTE: only fixed this after the length-uniformity assumption below
    # failed on real data -- K12's spec allows the LAST chunk overall to
    # be shorter than CHUNK_SIZE (whatever remains after full chunks are
    # taken), so at CTP's real ~32800-byte input size, chunks are
    # [8192, 8192, 8192, 34] per session, not uniformly 8192. Chunks must
    # be grouped by their padded length before lock-step batching can
    # apply -- lockstep only works within a group that all need the same
    # number of absorption rounds.
    padded_chunks = [c + ref.pad10star1(rate, len(c), domain) for c in flat_chunks]

    groups = {}  # padded_length -> list of (flat_index, padded_bytes)
    for idx, p in enumerate(padded_chunks):
        groups.setdefault(len(p), []).append((idx, p))

    cvs_flat = [None] * len(flat_chunks)

    for padded_len, items in groups.items():
        indices = [i for i, _ in items]
        group_padded = [p for _, p in items]
        n_blocks = padded_len // rate
        N = len(group_padded)

        group_np = np.frombuffer(b"".join(group_padded), dtype=np.uint8).reshape(N, padded_len)

        state_bytes = np.zeros((N, 200), dtype=np.uint8)
        field = _get_cached_field(N)

        for block_idx in range(n_blocks):
            block = group_np[:, block_idx * rate:(block_idx + 1) * rate]
            state_bytes[:, :rate] ^= block
            states_np = _bytes_to_states_vectorized(state_bytes)
            field.from_numpy(states_np)
            keccak_p12_batch(field, 12)
            state_bytes = _states_to_bytes_vectorized(field.to_numpy())

        for local_i, flat_idx in enumerate(indices):
            cvs_flat[flat_idx] = bytes(state_bytes[local_i, :32])

    cvs_by_session = [[] for _ in materials]
    for owner, cv in zip(chunk_owner, cvs_flat):
        cvs_by_session[owner].append(cv)

    results = []
    for S0, cvs in zip(session_S0, cvs_by_session):
        final_input = S0 + b"\x03" + b"\x00" * 7 + b"".join(cvs) + k12.length_encode(len(cvs)) + b"\xFF\xFF"
        results.append(k12._f_with_domain(final_input, domain=0x06, out_len=out_len))
    return results


if __name__ == "__main__":
    _gpu_note = "" if _BACKEND_NAME != "cpu" else " (no GPU backend found -- correctness only, no throughput conclusion)"
    print(f"Taichi backend in use: {_BACKEND_NAME}{_gpu_note}")

    print("\n=== Correctness: batched multi-session K12 vs per-session reference ===")
    N_SESSIONS = 12
    materials = [os.urandom(32800) for _ in range(N_SESSIONS)]  # CTP's real per-packet size
    out_len = 4096

    batched_results = batched_k12_many_sessions(materials, out_len)
    reference_results = [k12.kangarootwelve(m, out_len) for m in materials]

    all_match = all(a == b for a, b in zip(batched_results, reference_results))
    print("ALL SESSIONS MATCH" if all_match else "MISMATCH DETECTED")

    print("\n=== Timing: batched (GPU/Taichi) vs sequential pycryptodome K12 (fast, real baseline) ===")
    from Crypto.Hash import KangarooTwelve as PyCryptoK12

    def fast_sequential_k12(materials, out_len):
        for m in materials:
            h = PyCryptoK12.new(data=m)
            h.read(out_len)

    for N in [1, 8, 32, 128]:
        materials = [os.urandom(32800) for _ in range(N)]

        batched_k12_many_sessions(materials, out_len)  # warm up
        t0 = time.perf_counter()
        batched_k12_many_sessions(materials, out_len)
        t1 = time.perf_counter()
        batched_time = t1 - t0

        fast_sequential_k12(materials, out_len)  # warm up
        t0 = time.perf_counter()
        fast_sequential_k12(materials, out_len)
        t1 = time.perf_counter()
        fast_sequential_time = t1 - t0

        print(f"  N={N:>3}: batched(GPU)={batched_time*1000:8.2f}ms  "
              f"sequential(pycryptodome, fast C)={fast_sequential_time*1000:8.2f}ms  "
              f"ratio={fast_sequential_time/batched_time:5.2f}x")

    print("\n=== For reference only: vs our own slow Python reference (NOT a fair baseline) ===")
    for N in [1, 8, 32]:
        materials = [os.urandom(32800) for _ in range(N)]

        batched_k12_many_sessions(materials, out_len)  # warm up
        t0 = time.perf_counter()
        batched_k12_many_sessions(materials, out_len)
        t1 = time.perf_counter()
        batched_time = t1 - t0

        t0 = time.perf_counter()
        for m in materials:
            k12.kangarootwelve(m, out_len)
        t1 = time.perf_counter()
        sequential_time = t1 - t0

        print(f"  N={N:>3}: batched={batched_time*1000:8.2f}ms  "
              f"sequential(pure-Python k12)={sequential_time*1000:8.2f}ms  "
              f"ratio={sequential_time/batched_time:5.2f}x")
