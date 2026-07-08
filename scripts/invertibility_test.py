"""
Invertibility analysis for the theta (linear diffusion) and chi (nonlinear)
components before wiring them into CTP. Bijectivity of the whole round
function follows automatically if each component is individually bijective
and iota (XOR with a public constant) trivially is, since a composition of
bijections is a bijection -- so we check each piece separately rather than
trying to brute-force the full 64^3 state space (impossible).
"""

import numpy as np


# ---------------------------------------------------------------------------
# theta: linear operator, out = grid XOR (6 axis-neighbor shifts)
# Over GF(2)^n for n a power of 2, x^n - 1 = (x-1)^n (Frobenius, char 2), so
# GF(2)[x]/(x^n-1) is a LOCAL ring with maximal ideal (x-1). An element is a
# unit (invertible under convolution) iff its value at x=1 is odd. Theta's
# polynomial is 1 + x + x^-1 + y + y^-1 + z + z^-1 -> at (1,1,1) that's 7,
# which is odd -> theta should be invertible for any n that's a power of 2.
# Verify numerically on a small case by building the explicit matrix.
# ---------------------------------------------------------------------------

def theta_matrix_rank_gf2(n):
    """Build the explicit N x N GF(2) matrix for theta on an n x n x n grid
    and return its rank mod 2 via Gaussian elimination. N = n^3."""
    N = n ** 3

    def idx(i, j, k):
        return (i % n) * n * n + (j % n) * n + (k % n)

    # Build matrix row by row: row r has 1s at the positions that XOR into cell r.
    M = np.zeros((N, N), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                r = idx(i, j, k)
                M[r, idx(i, j, k)] ^= 1
                M[r, idx(i + 1, j, k)] ^= 1
                M[r, idx(i - 1, j, k)] ^= 1
                M[r, idx(i, j + 1, k)] ^= 1
                M[r, idx(i, j - 1, k)] ^= 1
                M[r, idx(i, j, k + 1)] ^= 1
                M[r, idx(i, j, k - 1)] ^= 1

    # Gaussian elimination over GF(2)
    A = M.copy()
    rows, cols = A.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if A[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        A[[rank, pivot]] = A[[pivot, rank]]
        for r in range(rows):
            if r != rank and A[r, col] == 1:
                A[r] ^= A[rank]
        rank += 1
    return rank, N


# ---------------------------------------------------------------------------
# chi: nonlinear, y_i = x_i XOR (NOT x_{i+a} AND x_{i+b}), applied along a
# single axis with offsets (a, b). Because the offsets used in chi_theta_test
# are both along axis 0, chi decomposes into independent 1D cyclic problems
# of length n (one per (j,k) pair). Test bijectivity of the 1D cyclic map
# exhaustively for small cycle lengths to find the general pattern.
# ---------------------------------------------------------------------------

def chi_cycle_bijective(L, a=1, b=2):
    """Exhaustively test whether y_i = x_i XOR (NOT x_{(i+a)%L} AND x_{(i+b)%L})
    is a bijection on {0,1}^L. Returns True/False. Only feasible for small L."""
    seen = set()
    for state in range(2 ** L):
        bits = [(state >> i) & 1 for i in range(L)]
        out = 0
        for i in range(L):
            na = bits[(i + a) % L]
            nb = bits[(i + b) % L]
            y = bits[i] ^ ((1 - na) & nb)
            out |= (y << i)
        if out in seen:
            return False
        seen.add(out)
    return True


if __name__ == "__main__":
    print("=== theta invertibility (explicit GF(2) rank check) ===")
    for n in [3, 4, 5]:
        rank, N = theta_matrix_rank_gf2(n)
        print(f"n={n}: rank={rank} / N={N}  {'INVERTIBLE' if rank == N else 'SINGULAR'}")

    print("\n=== chi-on-cycle bijectivity (offsets a=1, b=2) for various cycle lengths ===")
    for L in range(3, 21):
        bij = chi_cycle_bijective(L, a=1, b=2)
        parity = "odd" if L % 2 else "even"
        print(f"L={L:>2} ({parity}): {'BIJECTIVE' if bij else 'NOT bijective'}")
