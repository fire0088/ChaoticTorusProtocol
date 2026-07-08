"""
Throughput benchmark: fully-fused GPU K12 absorption (keccak_fused_gpu.py)
vs. pycryptodome's KangarooTwelve, at CTP's real per-packet input size.

Correctness must already be validated (run keccak_fused_gpu.py itself
first -- it prints "FUSED KERNEL VALIDATED" when safe to trust). This
script measures speed only.

IMPORTANT: fields are cached by shape and reused across calls. Passing a
freshly-allocated field of an already-seen shape to a Taichi kernel
retriggers expensive recompilation (confirmed directly: ~0.1ms/call
reusing the same object vs ~800ms/call for a fresh same-shape object, an
~8000x difference) -- this bit us once already in this same investigation
and is fixed here from the start rather than repeated.
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


def run_fused(materials: list, out_len: int = 4096):
    """Full K12 extraction using the FULLY fused pipeline: both the leaf
    (chaining-value) computation AND the final-node combination run as
    GPU kernels. An earlier version of this function left the final node
    on the slow Python reference path, which dominated end-to-end runtime
    once measured completely rather than in isolation -- see
    keccak_fused_gpu.py Stage C for the fix and its validation."""
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

    cvs_by_session = [[] for _ in materials]

    if flat_chunks:
        padded = [c + ref.pad10star1(RATE_BYTES, len(c), 0x0B) for c in flat_chunks]
        groups = {}
        for idx, p in enumerate(padded):
            groups.setdefault(len(p), []).append((idx, p))

        cvs_flat = [None] * len(flat_chunks)
        for plen, items in groups.items():
            indices = [i for i, _ in items]
            group_padded = [p for _, p in items]
            lanes, n_blocks = _prepare_block_lanes(group_padded)
            Ng = lanes.shape[0]
            block_f, out_f = _get_fields(
                ("leaf", Ng, n_blocks),
                [(Ng, n_blocks, RATE_LANES), (Ng, 5, 5)],
            )
            block_f.from_numpy(lanes)
            keccak_p12_fused(block_f, out_f, Ng, n_blocks)
            final_states = out_f.to_numpy()
            cv_bytes_batch = _states_to_bytes_batch(final_states)[:, :32]
            for local_i, flat_idx in enumerate(indices):
                cvs_flat[flat_idx] = bytes(cv_bytes_batch[local_i])

        for owner, cv in zip(chunk_owner, cvs_flat):
            cvs_by_session[owner].append(cv)

    # Final node: also fused, batched across sessions, grouped by final_input length.
    final_inputs = []
    for S0, cvs in zip(session_S0, cvs_by_session):
        if cvs:
            final_inputs.append(S0 + b"\x03" + b"\x00" * 7 + b"".join(cvs) + k12.length_encode(len(cvs)) + b"\xFF\xFF")
        else:
            final_inputs.append(S0)  # single-chunk (no-tree) case uses domain 0x07 instead; handled below
    domain_per_session = [0x06 if cvs else 0x07 for cvs in cvs_by_session]

    n_squeeze_blocks = (out_len + RATE_BYTES - 1) // RATE_BYTES
    results = [None] * len(materials)

    by_domain_and_len = {}
    for i, (fi, dom) in enumerate(zip(final_inputs, domain_per_session)):
        padded = fi + ref.pad10star1(RATE_BYTES, len(fi), dom)
        by_domain_and_len.setdefault((dom, len(padded)), []).append((i, padded))

    for (dom, plen), items in by_domain_and_len.items():
        indices = [i for i, _ in items]
        group_padded = [p for _, p in items]
        lanes, n_absorb_blocks = _prepare_block_lanes(group_padded)
        Nc = lanes.shape[0]
        absorb_f, squeeze_f = _get_fields(
            ("final", dom, Nc, n_absorb_blocks, n_squeeze_blocks),
            [(Nc, n_absorb_blocks, RATE_LANES), (Nc, n_squeeze_blocks, RATE_LANES)],
        )
        absorb_f.from_numpy(lanes)
        keccak_p12_fused_absorb_squeeze(absorb_f, squeeze_f, Nc, n_absorb_blocks, n_squeeze_blocks)
        squeezed_bytes = _lanes_to_bytes_vectorized(squeeze_f.to_numpy())
        for local_i, idx in enumerate(indices):
            results[idx] = bytes(squeezed_bytes[local_i, :out_len])

    return results


def run_pycryptodome(materials: list, out_len: int = 4096):
    return [KangarooTwelve.new(data=m).read(out_len) for m in materials]


if __name__ == "__main__":
    print(f"Taichi backend in use: {_BACKEND_NAME}"
          + ("" if _BACKEND_NAME != "cpu" else " (no GPU backend found -- results below are NOT GPU numbers)"))

    print("\n=== Correctness check (this function was just rewritten -- verify before timing) ===")
    check_materials = [os.urandom(32800) for _ in range(12)]
    fused_check = run_fused(check_materials)
    pyc_check = run_pycryptodome(check_materials)
    if fused_check == pyc_check:
        print("MATCH -- proceeding to timing.\n")
    else:
        print("MISMATCH -- DO NOT TRUST THE TIMING BELOW. Stopping.")
        raise SystemExit(1)

    print("\n=== Throughput: fully-fused GPU K12 vs. pycryptodome ===")
    print("(CTP's real per-packet input size: 32800 bytes; output: 4096 bytes)\n")

    for N in [1, 8, 32, 128, 512]:
        materials = [os.urandom(32800) for _ in range(N)]

        run_fused(materials)  # warm up: pays compilation cost once per distinct shape, not counted
        t0 = time.perf_counter()
        run_fused(materials)
        t1 = time.perf_counter()
        fused_time = t1 - t0

        run_pycryptodome(materials)
        t0 = time.perf_counter()
        run_pycryptodome(materials)
        t1 = time.perf_counter()
        pyc_time = t1 - t0

        ratio = pyc_time / fused_time
        verdict = "GPU FASTER" if ratio > 1 else "pycryptodome faster"
        print(f"  N={N:>4}: fused(GPU)={fused_time*1000:9.2f}ms  "
              f"pycryptodome={pyc_time*1000:8.2f}ms  ratio={ratio:6.3f}x  [{verdict}]")
