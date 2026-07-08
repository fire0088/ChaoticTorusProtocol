"""
Compare the existing threshold-counting Life-like rule against a
structurally different 3D update inspired by Keccak-f's round function:
a genuinely nonlinear step (chi-style: cell XOR (NOT neighbor_a AND
neighbor_b) for specific, non-symmetric offsets) composed with a linear
diffusion step (theta-style: XOR in parity of nearby cells along each axis).

This is a toy comparison, not a proposed final design -- the point is to
test whether moving away from a symmetric neighbor-COUNT function changes
the qualitative behavior (density stability, avalanche speed) relative to
the Carter Bays family.
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


def theta(grid: np.ndarray) -> np.ndarray:
    """Linear diffusion: XOR in each axis-neighbor pair (parity spread)."""
    out = grid.copy()
    for axis in range(3):
        out ^= np.roll(grid, 1, axis=axis)
        out ^= np.roll(grid, -1, axis=axis)
    return out


def chi(grid: np.ndarray, off_a=(1, 0, 0), off_b=(2, 0, 0)) -> np.ndarray:
    """
    Nonlinear confusion, modeled on Keccak-f's chi step:
        new_cell = cell XOR ((NOT neighbor_a) AND neighbor_b)
    Uses two SPECIFIC, asymmetric neighbor offsets rather than a symmetric
    count over all 26 neighbors -- this is the key structural difference
    from an outer-totalistic rule.
    """
    na = np.roll(np.roll(np.roll(grid, off_a[0], 0), off_a[1], 1), off_a[2], 2)
    nb = np.roll(np.roll(np.roll(grid, off_b[0], 0), off_b[1], 1), off_b[2], 2)
    return grid ^ ((1 - na) & nb)


def round_function(grid: np.ndarray) -> np.ndarray:
    return theta(chi(grid))


def hamming_bits(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


if __name__ == "__main__":
    n = 32

    print("=== Density stability: chi+theta round function ===")
    seed = os.urandom(32)
    grid = make_grid(seed, n)
    densities = []
    for gen in range(60):
        grid = round_function(grid)
        densities.append(grid.sum() / n ** 3)
    trace = [f"{densities[i]:.3f}" for i in range(0, 60, 5)]
    print("density trace:", trace)

    print("\n=== Avalanche test: 1-bit seed flip, chi+theta round function ===")
    trials = 20
    diffs_by_gen = {g: [] for g in [1, 2, 3, 4, 5, 8, 12]}
    for _ in range(trials):
        seed_a = bytearray(os.urandom(32))
        seed_b = bytearray(seed_a)
        seed_b[os.urandom(1)[0] % 32] ^= (1 << (os.urandom(1)[0] % 8))

        ga = make_grid(bytes(seed_a), n)
        gb = make_grid(bytes(seed_b), n)
        for gen in range(1, 13):
            ga = round_function(ga)
            gb = round_function(gb)
            if gen in diffs_by_gen:
                diffs_by_gen[gen].append(100.0 * hamming_bits(ga, gb) / n ** 3)

    for gen, diffs in diffs_by_gen.items():
        mean = sum(diffs) / len(diffs)
        print(f"  generation {gen:>2}: mean bit-difference = {mean:6.2f}%  (ideal: 50.00%)")

    print("\n=== For comparison: Carter Bays B4/S5 avalanche (same test) ===")
    from scipy.ndimage import convolve1d
    _BOX_1D = np.ones(3, dtype=np.int32)

    def cb_step(grid):
        g = grid.astype(np.int32)
        g = convolve1d(g, _BOX_1D, axis=0, mode="wrap")
        g = convolve1d(g, _BOX_1D, axis=1, mode="wrap")
        g = convolve1d(g, _BOX_1D, axis=2, mode="wrap")
        neighbor_sum = g - grid
        born = (neighbor_sum == 4) & (grid == 0)
        survives = (neighbor_sum == 5) & (grid == 1)
        return (born | survives).astype(np.uint8)

    diffs_by_gen_cb = {g: [] for g in [1, 2, 3, 4, 5, 8, 12]}
    for _ in range(trials):
        seed_a = bytearray(os.urandom(32))
        seed_b = bytearray(seed_a)
        seed_b[os.urandom(1)[0] % 32] ^= (1 << (os.urandom(1)[0] % 8))
        ga = make_grid(bytes(seed_a), n)
        gb = make_grid(bytes(seed_b), n)
        for gen in range(1, 13):
            ga = cb_step(ga)
            gb = cb_step(gb)
            if gen in diffs_by_gen_cb:
                diffs_by_gen_cb[gen].append(100.0 * hamming_bits(ga, gb) / n ** 3)

    for gen, diffs in diffs_by_gen_cb.items():
        mean = sum(diffs) / len(diffs)
        print(f"  generation {gen:>2}: mean bit-difference = {mean:6.2f}%  (ideal: 50.00%)")
