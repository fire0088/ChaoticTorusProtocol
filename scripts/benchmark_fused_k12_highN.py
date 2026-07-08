"""
Extends benchmark_fused_k12.py's test to higher session counts, to answer
the open question from the N<=512 data: does the fused-GPU/pycryptodome
gap actually cross over to a GPU win at some higher N, or does it plateau
above 1x? Reuses the same (already correctness-checked) run_fused and
run_pycryptodome functions -- this script only adds more N values and
prints progress as it goes, since higher N means more compile-time for
new field shapes and this may take a while.
"""

import os
import time

from benchmark_fused_k12 import run_fused, run_pycryptodome, _BACKEND_NAME

if __name__ == "__main__":
    print(f"Taichi backend in use: {_BACKEND_NAME}"
          + ("" if _BACKEND_NAME != "cpu" else " (no GPU backend found -- results below are NOT GPU numbers)"))

    print("\n=== Correctness check before any timing ===")
    check_materials = [os.urandom(32800) for _ in range(12)]
    if run_fused(check_materials) == run_pycryptodome(check_materials):
        print("MATCH -- proceeding to timing.\n")
    else:
        print("MISMATCH -- DO NOT TRUST TIMING BELOW. Stopping.")
        raise SystemExit(1)

    print("=== Throughput at higher N: does the gap cross over or plateau? ===")
    print("(CTP's real per-packet input size: 32800 bytes; output: 4096 bytes)\n")

    results = []
    for N in [512, 1024, 2048, 4096, 8192]:
        print(f"  N={N:>6}: generating materials and warming up shapes...", flush=True)
        materials = [os.urandom(32800) for _ in range(N)]

        t_warm0 = time.perf_counter()
        run_fused(materials)  # warm up: pays compile cost for any new shape at this N
        t_warm1 = time.perf_counter()

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
        results.append((N, fused_time, pyc_time, ratio))
        print(f"  N={N:>6}: fused(GPU)={fused_time*1000:9.2f}ms  "
              f"pycryptodome={pyc_time*1000:9.2f}ms  ratio={ratio:6.3f}x  "
              f"[{verdict}]  (warmup took {(t_warm1-t_warm0)*1000:.0f}ms, not counted)")

    print("\n=== Trend summary ===")
    print(f"{'N':>7} {'GPU slower by':>15} {'GPU time growth':>18} {'CPU time growth':>18}")
    prev_gpu, prev_pyc = None, None
    for N, fused_time, pyc_time, ratio in results:
        slower = 1.0 / ratio
        gpu_growth = fused_time / prev_gpu if prev_gpu else float("nan")
        cpu_growth = pyc_time / prev_pyc if prev_pyc else float("nan")
        print(f"{N:>7} {slower:>14.2f}x {gpu_growth:>17.2f}x {cpu_growth:>17.2f}x")
        prev_gpu, prev_pyc = fused_time, pyc_time

    crossed = any(r > 1.0 for _, _, _, r in results)
    print("\nCrossed over to a GPU win within this range: " + ("YES" if crossed else "NO"))
