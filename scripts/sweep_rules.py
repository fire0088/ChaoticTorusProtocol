"""
Sweep single-value birth/survival rules for the 3D outer-totalistic
Life-like family (26-neighbor Moore neighborhood, toroidal) to check
whether ANY rule in this family sustains non-degenerate density, or
whether extinction/explosion is a structural property of the family
rather than a quirk of the specific B4/S5 parameterization tested so far.

Uses a smaller lattice (32^3) and fewer generations for sweep speed;
promising candidates should be re-verified at full scale (64^3) before
being trusted.
"""

import os
import time
import numpy as np
from scipy.ndimage import convolve1d

_BOX_1D = np.ones(3, dtype=np.int32)


def run_rule(birth, survive, n=32, generations=60, seed=None):
    if seed is None:
        seed = os.urandom(32)
    import hashlib
    nbits = n ** 3
    nbytes = (nbits + 7) // 8
    raw = hashlib.shake_256(seed).digest(nbytes)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
    grid = bits.reshape(n, n, n).astype(np.uint8)

    densities = []
    for gen in range(generations):
        g = grid.astype(np.int32)
        g = convolve1d(g, _BOX_1D, axis=0, mode="wrap")
        g = convolve1d(g, _BOX_1D, axis=1, mode="wrap")
        g = convolve1d(g, _BOX_1D, axis=2, mode="wrap")
        neighbor_sum = g - grid
        born = np.isin(neighbor_sum, birth) & (grid == 0)
        survives = np.isin(neighbor_sum, survive) & (grid == 1)
        grid = (born | survives).astype(np.uint8)
        densities.append(grid.sum() / (n ** 3))
    return densities


def classify(densities, tol=0.02):
    final = densities[-5:]
    mean_final = sum(final) / len(final)
    trend = densities[-1] - densities[len(densities) // 2]
    if mean_final < 0.001:
        return "EXTINCT"
    if mean_final > 0.85:
        return "SATURATED"
    if abs(trend) > 0.15:
        return "DRIFTING"  # monotonically growing/shrinking, not settled
    if max(final) - min(final) < tol and 0.001 < mean_final < 0.85:
        return "FROZEN"  # stable but static (still life / no dynamics)
    return "CANDIDATE"  # persistent, mid-range, still fluctuating


if __name__ == "__main__":
    t0 = time.time()
    results = {}
    trials = 4
    for b in range(1, 9):
        for s in range(1, 9):
            outcomes = []
            for _ in range(trials):
                d = run_rule([b], [s], n=32, generations=50)
                outcomes.append(classify(d))
            results[(b, s)] = outcomes

    print(f"Swept {len(results)} rules x {trials} trials in {time.time()-t0:.1f}s\n")

    # Summarize
    from collections import Counter
    print(f"{'B':>3} {'S':>3}  outcomes")
    candidates = []
    for (b, s), outcomes in sorted(results.items()):
        c = Counter(outcomes)
        summary = ", ".join(f"{k}:{v}" for k, v in c.items())
        print(f"{b:>3} {s:>3}  {summary}")
        if c.get("CANDIDATE", 0) >= trials // 2:
            candidates.append((b, s))

    print(f"\nRules with CANDIDATE behavior in >=50% of trials: {candidates}")
