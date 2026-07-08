"""
Fully vectorized K12 extraction for the realistic CTP case: all sessions
in a batch share the same packet size. Building on the two fixes already
validated in keccak_fused_gpu.py (lane conversion, batch reassembly), this
eliminates the remaining per-session Python loops that profiling showed
dominating on real GPU hardware once the kernel itself became sub-1% of
total time: chunking, padding, and final-input construction.

Key fact this relies on: K12's pad10star1 padding depends only on a
chunk's LENGTH and domain byte, never its content -- so for same-length
sessions, every corresponding chunk needs bit-for-bit identical padding,
computed once and broadcast across the whole batch via numpy instead of
recomputed per session in a Python loop.

Validated against pycryptodome before any timing claim is made, same
discipline as every other module in this investigation.
"""

import os
import time

import numpy as np
import taichi as ti
from Crypto.Hash import KangarooTwelve

from keccak_fused_gpu import (
    keccak_p12_fused, keccak_p12_fused_absorb_squeeze,
    _vectorized_chunk_and_pad, _lanes_from_padded_array, _states_to_bytes_batch,
    _lanes_to_bytes_vectorized,
    RATE_LANES, RATE_BYTES, _BACKEND_NAME,
)
import keccak_reference as ref
import k12_full as k12

_field_cache = {}


def _get_fields(shape_key, shapes):
    if shape_key not in _field_cache:
        _field_cache[shape_key] = tuple(ti.field(dtype=ti.u64, shape=s) for s in shapes)
    return _field_cache[shape_key]


def run_fused_vectorized(materials: list, out_len: int = 4096):
    """Requires uniform-length materials (asserted inside
    _vectorized_chunk_and_pad). Every per-session Python loop from the
    earlier version is replaced with a single batch numpy operation."""
    N = len(materials)
    S0, groups, n_full_chunks, has_last = _vectorized_chunk_and_pad(materials)
    n_chunks_per_session = n_full_chunks + has_last

    # ---- Leaf (chaining value) computation, fully batched ----
    cvs_by_kind = {}  # "full" -> (N, n_full_chunks, 32), "last" -> (N, 1, 32)
    for padded_len, (kind, count, arr) in groups.items():
        Ng = arr.shape[0]
        lanes, n_blocks = _lanes_from_padded_array(arr)
        block_f, out_f = _get_fields(("leaf", Ng, n_blocks), [(Ng, n_blocks, RATE_LANES), (Ng, 5, 5)])
        block_f.from_numpy(lanes)
        keccak_p12_fused(block_f, out_f, Ng, n_blocks)
        final_states = out_f.to_numpy()
        cv_bytes = _states_to_bytes_batch(final_states)[:, :32]  # (Ng, 32)
        if kind == "full":
            cvs_by_kind["full"] = cv_bytes.reshape(N, count, 32)
        else:
            cvs_by_kind["last"] = cv_bytes.reshape(N, 1, 32)

    pieces = []
    if "full" in cvs_by_kind:
        pieces.append(cvs_by_kind["full"])
    if "last" in cvs_by_kind:
        pieces.append(cvs_by_kind["last"])
    all_cvs = np.concatenate(pieces, axis=1) if pieces else np.zeros((N, 0, 32), dtype=np.uint8)  # (N, n_chunks, 32)
    cvs_flat_per_session = all_cvs.reshape(N, n_chunks_per_session * 32)

    # ---- Final node input, built for the whole batch in one shot ----
    marker = np.frombuffer(b"\x03" + b"\x00" * 7, dtype=np.uint8)
    length_enc = np.frombuffer(k12.length_encode(n_chunks_per_session), dtype=np.uint8)
    tail = np.frombuffer(b"\xFF\xFF", dtype=np.uint8)

    final_input_arr = np.concatenate([
        S0,
        np.broadcast_to(marker, (N, len(marker))),
        cvs_flat_per_session,
        np.broadcast_to(length_enc, (N, len(length_enc))),
        np.broadcast_to(tail, (N, len(tail))),
    ], axis=1)

    final_input_len = final_input_arr.shape[1]
    pad_final = np.frombuffer(ref.pad10star1(RATE_BYTES, final_input_len, 0x06), dtype=np.uint8)
    padded_final = np.concatenate([final_input_arr, np.broadcast_to(pad_final, (N, len(pad_final)))], axis=1)

    n_absorb_blocks = padded_final.shape[1] // RATE_BYTES
    n_squeeze_blocks = (out_len + RATE_BYTES - 1) // RATE_BYTES

    absorb_lanes, _ = _lanes_from_padded_array(padded_final)
    absorb_f, squeeze_f = _get_fields(
        ("final", N, n_absorb_blocks, n_squeeze_blocks),
        [(N, n_absorb_blocks, RATE_LANES), (N, n_squeeze_blocks, RATE_LANES)],
    )
    absorb_f.from_numpy(absorb_lanes)
    keccak_p12_fused_absorb_squeeze(absorb_f, squeeze_f, N, n_absorb_blocks, n_squeeze_blocks)
    squeezed_bytes = _lanes_to_bytes_vectorized(squeeze_f.to_numpy())

    return [bytes(squeezed_bytes[i, :out_len]) for i in range(N)]


def run_pycryptodome(materials: list, out_len: int = 4096):
    return [KangarooTwelve.new(data=m).read(out_len) for m in materials]


if __name__ == "__main__":
    print(f"Taichi backend in use: {_BACKEND_NAME}"
          + ("" if _BACKEND_NAME != "cpu" else " (no GPU backend found -- results below are NOT GPU numbers)"))

    print("\n=== Correctness check (fully vectorized pipeline, new code -- verify before timing) ===")
    check_materials = [os.urandom(32800) for _ in range(37)]  # non-round number on purpose
    fused_check = run_fused_vectorized(check_materials)
    pyc_check = run_pycryptodome(check_materials)
    if fused_check == pyc_check:
        print("MATCH -- proceeding to timing.\n")
    else:
        mismatches = sum(1 for a, b in zip(fused_check, pyc_check) if a != b)
        print(f"MISMATCH ({mismatches}/{len(check_materials)} sessions wrong) -- DO NOT TRUST TIMING. Stopping.")
        raise SystemExit(1)

    print("=== Throughput: fully vectorized GPU pipeline vs. pycryptodome ===")
    print("(CTP's real per-packet input size: 32800 bytes; output: 4096 bytes)\n")

    for N in [512, 1024, 2048, 4096, 8192]:
        materials = [os.urandom(32800) for _ in range(N)]

        run_fused_vectorized(materials)  # warm up
        t0 = time.perf_counter()
        run_fused_vectorized(materials)
        t1 = time.perf_counter()
        fused_time = t1 - t0

        run_pycryptodome(materials)
        t0 = time.perf_counter()
        run_pycryptodome(materials)
        t1 = time.perf_counter()
        pyc_time = t1 - t0

        ratio = pyc_time / fused_time
        verdict = "GPU FASTER" if ratio > 1 else "pycryptodome faster"
        print(f"  N={N:>5}: fused(GPU)={fused_time*1000:9.2f}ms  "
              f"pycryptodome={pyc_time*1000:9.2f}ms  ratio={ratio:6.3f}x  [{verdict}]")
