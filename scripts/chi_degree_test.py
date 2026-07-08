"""
Compare chi variants of increasing algebraic degree. Degree 2 (the original,
single AND term) is a weak category by the same argument that made the
Carter Bays threshold rule weak -- low algebraic degree is exactly what
makes a Boolean function vulnerable to linearization/algebraic attacks.
Since chi is wrapped in a Feistel network (invertibility_test.py,
feistel_lattice.py), it does NOT need to be bijective on its own, which
gives complete freedom to raise its degree without re-deriving invertibility.

Degree here is exact by construction (the size of the largest AND-monomial
used), not something that needs to be measured -- but balance and
diffusion quality are NOT automatic just because degree went up, so those
are tested empirically as usual.
"""

import hashlib
import numpy as np
import os


def make_grid(seed: bytes, n: int) -> np.ndarray:
    nbits = n ** 3
    nbytes = (nbits + 7) // 8
    raw = hashlib.shake_256(seed).digest(nbytes)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
    return bits.reshape(n, n, n).astype(np.uint8)


def shift(grid, off):
    return np.roll(np.roll(np.roll(grid, off[0], 0), off[1], 1), off[2], 2)


def theta(grid):
    out = grid.copy()
    for axis in range(3):
        out ^= np.roll(grid, 1, axis=axis)
        out ^= np.roll(grid, -1, axis=axis)
    return out


# --- chi variants, increasing algebraic degree ---

def chi_degree2(grid):
    """Original: y = x XOR (NOT a AND b). Degree 2."""
    a = shift(grid, (1, 0, 0))
    b = shift(grid, (2, 0, 0))
    return grid ^ ((1 - a) & b)


def chi_degree3_pure(grid):
    """y = x XOR (a AND b AND c). Degree 3, single monomial."""
    a = shift(grid, (1, 0, 0))
    b = shift(grid, (2, 0, 0))
    c = shift(grid, (0, 1, 0))
    return grid ^ (a & b & c)


def chi_degree3_mixed(grid):
    """y = x XOR (a AND b) XOR (c AND d AND e). Degree 3, richer ANF
    (one degree-2 monomial, one degree-3 monomial) -- uses 5 distinct
    neighbor offsets across two axes."""
    a = shift(grid, (1, 0, 0))
    b = shift(grid, (2, 0, 0))
    c = shift(grid, (0, 1, 0))
    d = shift(grid, (0, 2, 0))
    e = shift(grid, (0, 0, 1))
    return grid ^ (a & b) ^ (c & d & e)


def chi_degree4(grid):
    """y = x XOR (a AND b AND c AND d). Degree 4, single monomial
    (increasingly biased toward 0 as degree rises -- AND of 4 independent
    uniform bits is 1 only 1/16 of the time)."""
    a = shift(grid, (1, 0, 0))
    b = shift(grid, (2, 0, 0))
    c = shift(grid, (0, 1, 0))
    d = shift(grid, (0, 2, 0))
    return grid ^ (a & b & c & d)


VARIANTS = {
    "degree2 (original)": chi_degree2,
    "degree3 (pure AND)": chi_degree3_pure,
    "degree3 (mixed ANF)": chi_degree3_mixed,
    "degree4 (pure AND)": chi_degree4,
}


def round_function(grid, chi_fn):
    return theta(chi_fn(grid))


def test_variant(name, chi_fn, n=32, generations=100, avalanche_trials=15):
    seed = os.urandom(32)
    grid = make_grid(seed, n)
    densities = []
    for _ in range(generations):
        grid = round_function(grid, chi_fn)
        densities.append(grid.sum() / n ** 3)

    diffs_gen1, diffs_gen5 = [], []
    for _ in range(avalanche_trials):
        sa = bytearray(os.urandom(32))
        sb = bytearray(sa)
        sb[os.urandom(1)[0] % 32] ^= (1 << (os.urandom(1)[0] % 8))
        ga, gb = make_grid(bytes(sa), n), make_grid(bytes(sb), n)
        for gen in range(1, 6):
            ga = round_function(ga, chi_fn)
            gb = round_function(gb, chi_fn)
            if gen == 1:
                diffs_gen1.append(100.0 * np.sum(ga != gb) / n ** 3)
            if gen == 5:
                diffs_gen5.append(100.0 * np.sum(ga != gb) / n ** 3)

    print(f"{name:22s}  density(last 10 gens)={np.mean(densities[-10:]):.4f}"
          f"  min={min(densities):.4f} max={max(densities):.4f}"
          f"  | avalanche gen1={np.mean(diffs_gen1):5.2f}%  gen5={np.mean(diffs_gen5):5.2f}%")


if __name__ == "__main__":
    print(f"{'variant':22s}  {'density stats':40s}  {'avalanche'}")
    for name, fn in VARIANTS.items():
        test_variant(name, fn)
