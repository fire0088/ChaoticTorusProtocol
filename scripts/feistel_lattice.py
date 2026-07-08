"""
Feistel-structured replacement for the chi+theta round function.

Motivation (see invertibility_test.py): theta (linear diffusion) is
invertible at n=64 by construction (confirmed at n=4, n=8, matching the
power-of-2 argument). But chi (nonlinear, single-axis offsets) is bijective
only for ODD cycle lengths -- confirmed empirically for L=3..19 -- and our
lattice axis length is 64 (even). Any translation-based offset on a
power-of-2-sized torus produces cycle lengths that divide 64, all of which
are even except the trivial length 1. So chi as originally specified is
NOT bijective at full scale, regardless of which offsets are chosen.

Fix: a Feistel network. Split the lattice into two halves and update one
half using an arbitrary function of the other. This is invertible BY
CONSTRUCTION regardless of whether the inner round function F is itself
bijective -- F can be non-injective, non-surjective, anything -- which
sidesteps the odd/even cycle-length problem entirely instead of hunting
for parameters that happen to dodge it.

    L' = R
    R' = L XOR F(R, round_constant)

Inverse:
    R = L'
    L = R' XOR F(L', round_constant)

Multiple rounds are required for L and R to each be influenced by the
other (a single round leaves L's own internal structure completely
unmixed by F). Round constants are PUBLIC (derived from generation and
round index only, not secret) and exist solely to break the round
function's translational symmetry across rounds -- the same purpose
Keccak's iota step serves.
"""

import hashlib
import numpy as np

_BOX_1D_WEIGHTS = None  # placeholder if a separable version is wanted later


def _theta(half: np.ndarray) -> np.ndarray:
    """Linear diffusion: XOR in all 6 axis-neighbors, wrapping each axis
    at its own (possibly non-64) length -- this operates on a half-lattice
    of shape (n//2, n, n)."""
    out = half.copy()
    for axis in range(3):
        out ^= np.roll(half, 1, axis=axis)
        out ^= np.roll(half, -1, axis=axis)
    return out


def _chi(half: np.ndarray, off_a=(1, 0, 0), off_b=(2, 0, 0)) -> np.ndarray:
    """Nonlinear step. Does NOT need to be bijective here -- see module
    docstring -- so no constraint on offsets or parity."""
    na = np.roll(np.roll(np.roll(half, off_a[0], 0), off_a[1], 1), off_a[2], 2)
    nb = np.roll(np.roll(np.roll(half, off_b[0], 0), off_b[1], 1), off_b[2], 2)
    return half ^ ((1 - na) & nb)


def _round_constant(shape, generation: int, round_idx: int) -> np.ndarray:
    """Public (non-secret) round constant, deterministic function of the
    generation and round index only. Breaks translational symmetry across
    both generations and Feistel rounds, the way Keccak's iota does."""
    nbits = int(np.prod(shape))
    nbytes = (nbits + 7) // 8
    tag = f"CTP-round-const-{generation}-{round_idx}".encode()
    raw = hashlib.shake_256(tag).digest(nbytes)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
    return bits.reshape(shape).astype(np.uint8)


def _F(half: np.ndarray, generation: int, round_idx: int) -> np.ndarray:
    """The Feistel round function: theta then chi then a public round
    constant. Need not be bijective (see module docstring)."""
    rc = _round_constant(half.shape, generation, round_idx)
    return _chi(_theta(half)) ^ rc


NUM_FEISTEL_ROUNDS = 8


def feistel_step(grid: np.ndarray, generation: int) -> np.ndarray:
    n = grid.shape[0]
    half = n // 2
    L, R = grid[:half].copy(), grid[half:].copy()
    for r in range(NUM_FEISTEL_ROUNDS):
        L, R = R, L ^ _F(R, generation, r)
    return np.concatenate([L, R], axis=0)


def feistel_step_inverse(grid: np.ndarray, generation: int) -> np.ndarray:
    n = grid.shape[0]
    half = n // 2
    L, R = grid[:half].copy(), grid[half:].copy()
    for r in reversed(range(NUM_FEISTEL_ROUNDS)):
        L, R = R ^ _F(L, generation, r), L
    return np.concatenate([L, R], axis=0)


if __name__ == "__main__":
    import os
    import hashlib as _h

    def make_grid(seed: bytes, n: int) -> np.ndarray:
        nbits = n ** 3
        nbytes = (nbits + 7) // 8
        raw = _h.shake_256(seed).digest(nbytes)
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
        return bits.reshape(n, n, n).astype(np.uint8)

    n = 64
    print(f"=== Round-trip correctness at full scale (n={n}) ===")
    all_ok = True
    for trial in range(20):
        grid = make_grid(os.urandom(32), n)
        generation = trial
        forward = feistel_step(grid, generation)
        back = feistel_step_inverse(forward, generation)
        ok = np.array_equal(grid, back)
        all_ok &= ok
        if not ok:
            print(f"  trial {trial}: MISMATCH")
    print("ALL ROUND-TRIPS OK" if all_ok else "FAILURES DETECTED")

    print(f"\n=== Density stability over 100 generations (n={n}) ===")
    grid = make_grid(os.urandom(32), n)
    densities = []
    for gen in range(100):
        grid = feistel_step(grid, gen)
        densities.append(grid.sum() / n ** 3)
    trace = [f"{densities[i]:.4f}" for i in range(0, 100, 10)]
    print("density trace:", trace)
    print(f"min={min(densities):.4f} max={max(densities):.4f}")

    print(f"\n=== Avalanche test (n={n}, 1-bit seed flip) ===")
    trials = 15
    diffs_by_gen = {g: [] for g in [1, 2, 3, 5, 8]}
    for _ in range(trials):
        seed_a = bytearray(os.urandom(32))
        seed_b = bytearray(seed_a)
        seed_b[os.urandom(1)[0] % 32] ^= (1 << (os.urandom(1)[0] % 8))
        ga = make_grid(bytes(seed_a), n)
        gb = make_grid(bytes(seed_b), n)
        for gen in range(1, 9):
            ga = feistel_step(ga, gen)
            gb = feistel_step(gb, gen)
            if gen in diffs_by_gen:
                diffs_by_gen[gen].append(100.0 * np.sum(ga != gb) / n ** 3)
    for gen, diffs in diffs_by_gen.items():
        mean = sum(diffs) / len(diffs)
        print(f"  generation {gen}: mean bit-difference = {mean:.2f}%  (ideal: 50.00%)")
