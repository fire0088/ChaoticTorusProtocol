"""
Breaks run_fused()'s work into phases and times each one separately, to
find out WHY GPU time scaled perfectly linearly with N above 512 sessions
(flat ~1.7-1.8x slower than pycryptodome from N=512 to N=8192) rather than
continuing the sub-linear improvement seen below N=512.

Two very different explanations would produce that same top-line number:
  (a) genuine GPU parallel-capacity saturation -- the actual kernel
      execution itself scales linearly once session count exceeds what
      the hardware can run truly simultaneously, or
  (b) a software bottleneck elsewhere (Python-side grouping/reassembly,
      numpy conversion, host<->device transfer) that scales linearly with
      N regardless of the GPU kernel's own behavior, and dominates once
      the kernel part is fast enough.
Only a phase-by-phase breakdown can tell these apart -- guessing which one
it is would be exactly the kind of unverified claim this project has
tried consistently not to make.
"""

import os
import time

import numpy as np
import taichi as ti
from Crypto.Hash import KangarooTwelve

from keccak_fused_gpu import (
    keccak_p12_fused, keccak_p12_fused_absorb_squeeze,
    _prepare_block_lanes, _lanes_to_bytes_vectorized, _states_to_bytes_batch,
    RATE_LANES, RATE_BYTES, _BACKEND_NAME,
)
import keccak_reference as ref
import k12_full as k12

_field_cache = {}


def _get_fields(shape_key, shapes):
    if shape_key not in _field_cache:
        _field_cache[shape_key] = tuple(ti.field(dtype=ti.u64, shape=s) for s in shapes)
    return _field_cache[shape_key]


def run_fused_instrumented(materials: list, out_len: int = 4096, warm: bool = False):
    """Same computation as benchmark_fused_k12.run_fused, but broken into
    timed phases. Set warm=True to skip timing (for the warmup call)."""
    t = {}

    def tic():
        return time.perf_counter()

    def toc(t0, key):
        dt = time.perf_counter() - t0
        if not warm:
            t[key] = t.get(key, 0.0) + dt

    # --- Phase 1: build S0/chunks (pure Python, no crypto) ---
    t0 = tic()
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
    toc(t0, "1_python_chunking")

    cvs_by_session = [[] for _ in materials]

    if flat_chunks:
        # --- Phase 2: padding + grouping (Python + numpy) ---
        t0 = tic()
        padded = [c + ref.pad10star1(RATE_BYTES, len(c), 0x0B) for c in flat_chunks]
        groups = {}
        for idx, p in enumerate(padded):
            groups.setdefault(len(p), []).append((idx, p))
        toc(t0, "2_leaf_padding_grouping")

        cvs_flat = [None] * len(flat_chunks)
        for plen, items in groups.items():
            indices = [i for i, _ in items]
            group_padded = [p for _, p in items]

            # --- Phase 3: bytes -> lanes conversion (numpy) ---
            t0 = tic()
            lanes, n_blocks = _prepare_block_lanes(group_padded)
            Ng = lanes.shape[0]
            toc(t0, "3_leaf_lane_conversion")

            block_f, out_f = _get_fields(
                ("leaf", Ng, n_blocks), [(Ng, n_blocks, RATE_LANES), (Ng, 5, 5)],
            )

            # --- Phase 4: upload to GPU ---
            t0 = tic()
            block_f.from_numpy(lanes)
            toc(t0, "4_leaf_upload")

            # --- Phase 5: kernel execution ---
            t0 = tic()
            keccak_p12_fused(block_f, out_f, Ng, n_blocks)
            toc(t0, "5_leaf_kernel")

            # --- Phase 6: download from GPU ---
            t0 = tic()
            final_states = out_f.to_numpy()
            toc(t0, "6_leaf_download")

            # --- Phase 7: reassembly (now vectorized, was per-chunk Python loop) ---
            t0 = tic()
            cv_bytes_batch = _states_to_bytes_batch(final_states)[:, :32]
            for local_i, flat_idx in enumerate(indices):
                cvs_flat[flat_idx] = bytes(cv_bytes_batch[local_i])
            toc(t0, "7_leaf_reassembly")

        for owner, cv in zip(chunk_owner, cvs_flat):
            cvs_by_session[owner].append(cv)

    # --- Phase 8: build final_input strings (Python) ---
    t0 = tic()
    final_inputs = []
    for S0, cvs in zip(session_S0, cvs_by_session):
        if cvs:
            final_inputs.append(S0 + b"\x03" + b"\x00" * 7 + b"".join(cvs) + k12.length_encode(len(cvs)) + b"\xFF\xFF")
        else:
            final_inputs.append(S0)
    domain_per_session = [0x06 if cvs else 0x07 for cvs in cvs_by_session]
    n_squeeze_blocks = (out_len + RATE_BYTES - 1) // RATE_BYTES
    toc(t0, "8_final_build_input")

    # --- Phase 9: padding + grouping for final node ---
    t0 = tic()
    by_domain_and_len = {}
    for i, (fi, dom) in enumerate(zip(final_inputs, domain_per_session)):
        padded = fi + ref.pad10star1(RATE_BYTES, len(fi), dom)
        by_domain_and_len.setdefault((dom, len(padded)), []).append((i, padded))
    toc(t0, "9_final_padding_grouping")

    results = [None] * len(materials)
    for (dom, plen), items in by_domain_and_len.items():
        indices = [i for i, _ in items]
        group_padded = [p for _, p in items]

        t0 = tic()
        lanes, n_absorb_blocks = _prepare_block_lanes(group_padded)
        Nc = lanes.shape[0]
        toc(t0, "10_final_lane_conversion")

        absorb_f, squeeze_f = _get_fields(
            ("final", dom, Nc, n_absorb_blocks, n_squeeze_blocks),
            [(Nc, n_absorb_blocks, RATE_LANES), (Nc, n_squeeze_blocks, RATE_LANES)],
        )

        t0 = tic()
        absorb_f.from_numpy(lanes)
        toc(t0, "11_final_upload")

        t0 = tic()
        keccak_p12_fused_absorb_squeeze(absorb_f, squeeze_f, Nc, n_absorb_blocks, n_squeeze_blocks)
        toc(t0, "12_final_kernel")

        t0 = tic()
        squeezed_lanes = squeeze_f.to_numpy()
        toc(t0, "13_final_download")

        t0 = tic()
        squeezed_bytes = _lanes_to_bytes_vectorized(squeezed_lanes)
        for local_i, idx in enumerate(indices):
            results[idx] = bytes(squeezed_bytes[local_i, :out_len])
        toc(t0, "14_final_reassembly")

    return results, t


if __name__ == "__main__":
    print(f"Taichi backend in use: {_BACKEND_NAME}"
          + ("" if _BACKEND_NAME != "cpu" else " (no GPU backend found -- results below are NOT GPU numbers)"))

    print("\n=== Correctness check ===")
    check_materials = [os.urandom(32800) for _ in range(12)]
    check_results, _ = run_fused_instrumented(check_materials, warm=True)
    pyc_check = [KangarooTwelve.new(data=m).read(4096) for m in check_materials]
    if check_results == pyc_check:
        print("MATCH -- proceeding.\n")
    else:
        print("MISMATCH -- DO NOT TRUST RESULTS BELOW. Stopping.")
        raise SystemExit(1)

    for N in [512, 2048, 8192]:
        print(f"\n=== Phase breakdown at N={N} ===")
        materials = [os.urandom(32800) for _ in range(N)]

        run_fused_instrumented(materials, warm=True)  # warm up, uncounted

        _, timings = run_fused_instrumented(materials, warm=False)
        total = sum(timings.values())
        for phase in sorted(timings.keys()):
            dt = timings[phase]
            pct = 100.0 * dt / total if total > 0 else 0.0
            bar = "#" * int(pct / 2)
            print(f"  {phase:28s} {dt*1000:9.2f} ms  ({pct:5.1f}%)  {bar}")
        print(f"  {'TOTAL':28s} {total*1000:9.2f} ms")
