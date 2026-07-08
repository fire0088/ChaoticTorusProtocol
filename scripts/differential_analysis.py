"""
Differential cryptanalysis attempt against CTP's diffusion layer, aimed at
the specific failure mode the cube tester (cube_tester.py) does not cover:
not "is the algebraic degree low" but "does some SPECIFIC chosen input
difference propagate with anomalously HIGH probability" through the round
function. These are genuinely different attack families (Biham-Shamir
differential cryptanalysis vs. Dinur-Shamir cube/higher-order attacks).

Scope, stated up front rather than after the fact: a full, formal,
multi-round differential trail probability search over an 8-round Feistel
network on a 262,144-bit state is not something this analysis attempts --
that is a genuinely hard combinatorial search that real cipher analysis
usually automates with MILP/SAT solvers built for exactly this problem,
well beyond what is attempted here. What IS tractable and exact:

  Part A: chi is the only nonlinear step (theta is GF(2)-linear, so
  differences propagate through it deterministically, not
  probabilistically; iota XORs a PUBLIC constant into both sides of any
  difference pair, so it cancels and does not affect differential
  propagation at all -- worth stating plainly rather than leaving
  implicit). chi acts as a 6-input-to-1-output local function applied
  uniformly at every cell. Its exact differential behavior -- computed
  over all 64 possible local input differences, not sampled -- is fully
  tractable, unlike the full system, and gives the same kind of local
  bias information a standard S-box DDT would for a block cipher.

  Part B: empirical differential trail tracking through the REAL
  construction (ctp_cipher.py's actual theta/chi/iota, not a toy model),
  starting from low-weight input differences and watching whether any of
  them diffuse anomalously slowly across real rounds -- a practical stand-
  in for a trail search, not a substitute for one.
"""

import numpy as np

import ctp_cipher as ctp


# ---------------------------------------------------------------------------
# Part A: exact DDT for chi's local 6-input function
# ---------------------------------------------------------------------------

def chi_local(x: int, a: int, b: int, c: int, d: int, e: int) -> int:
    """Exactly ctp_cipher._chi's per-cell formula, as a scalar function of
    its 6 local inputs."""
    return x ^ (a & b) ^ (c & d & e)


def compute_chi_ddt():
    """Exact (not sampled) differential distribution table. Returns a dict
    mapping input difference (as a 6-bit int, bit order x,a,b,c,d,e) to the
    probability that the output difference is 1, computed over all 64
    possible base input values."""
    ddt = {}
    for din in range(64):
        count_out1 = 0
        for base in range(64):
            x, a, b, c, d, e = [(base >> i) & 1 for i in range(6)]
            base_out = chi_local(x, a, b, c, d, e)
            dx, da, db, dc, dd, de = [(din >> i) & 1 for i in range(6)]
            x2, a2, b2, c2, d2, e2 = x ^ dx, a ^ da, b ^ db, c ^ dc, d ^ dd, e ^ de
            other_out = chi_local(x2, a2, b2, c2, d2, e2)
            if base_out ^ other_out:
                count_out1 += 1
        ddt[din] = count_out1 / 64.0
    return ddt


# ---------------------------------------------------------------------------
# Part B: empirical differential trail tracking through the real construction
# ---------------------------------------------------------------------------

def make_grid_from_seed(seed: bytes, n: int) -> np.ndarray:
    import hashlib
    nbits = n * n * n
    nbytes = (nbits + 7) // 8
    raw = hashlib.shake_256(seed).digest(nbytes)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
    return bits.reshape(n, n, n).astype(np.uint8)


def track_difference_trail(seed: bytes, flip_position: tuple, n_rounds: int = 8,
                            n: int = ctp.LATTICE_N):
    """Start two lattices differing in exactly one bit (flip_position, a
    coordinate in the FULL n x n x n grid before the Feistel split), run
    them through n_rounds of the REAL Feistel round function (same code
    path as ctp_cipher.py), and report the Hamming weight of their
    difference after each round -- looking for any trail that diffuses
    anomalously slowly rather than reaching the ideal ~50% immediately, as
    already established for random differences by avalanche testing."""
    grid_a = make_grid_from_seed(seed, n)
    grid_b = grid_a.copy()
    grid_b[flip_position] ^= 1

    weights = []
    for gen in range(n_rounds):
        grid_a = ctp.feistel_step(grid_a, gen)
        grid_b = ctp.feistel_step(grid_b, gen)
        weight = int(np.sum(grid_a != grid_b))
        weights.append(weight)
    return weights


if __name__ == "__main__":
    import os

    print("=== Part A: Exact DDT for chi's local 6-input function ===")
    ddt = compute_chi_ddt()

    nonzero_diffs = {k: v for k, v in ddt.items() if k != 0}
    sorted_by_bias = sorted(nonzero_diffs.items(), key=lambda kv: abs(kv[1] - 0.5), reverse=True)
    max_bias_diff, max_bias_p = sorted_by_bias[0]
    max_bias = abs(max_bias_p - 0.5)

    print(f"Number of nonzero input differences: {len(nonzero_diffs)}")
    print(f"Highest-bias differential: input diff (x,a,b,c,d,e bits) = {max_bias_diff:06b}, "
          f"P(output diff=1) = {max_bias_p:.4f} (bias {max_bias:.4f})")
    if max_bias_diff == 1:
        print("  This is the isolated-x-only difference (a=b=c=d=e unchanged). Since")
        print("  chi(x,a,b,c,d,e) = x XOR (a AND b) XOR (c AND d AND e), an isolated")
        print("  difference in x alone gives Δy = Δx deterministically -- forced by the")
        print("  bare linear term, unavoidable for any function of this shape (the same")
        print("  property holds for Keccak's own chi step under the analogous isolated")
        print("  difference). Not novel, not exploitable in isolation; Part B below")
        print("  checks whether it survives into a multi-round trail once real")
        print("  diffusion (theta) is applied, which is the question that matters.")

    biases = [abs(v - 0.5) for v in nonzero_diffs.values()]
    excluding_trivial = [abs(v - 0.5) for k, v in nonzero_diffs.items() if k != 1]
    print(f"\nBias distribution over all 63 nonzero input differences:")
    print(f"  mean bias (all 63):            {np.mean(biases):.4f}")
    print(f"  mean bias (excluding the 1 trivial case): {np.mean(excluding_trivial):.4f}")
    print(f"  max bias excluding trivial case:           {np.max(excluding_trivial):.4f}")
    n_perfectly_unbiased = sum(1 for b in excluding_trivial if b == 0.0)
    print(f"  differentials with EXACTLY 0.5 probability (perfectly unbiased): "
          f"{n_perfectly_unbiased} / 62")

    print("\n=== Part B: Empirical differential trail tracking (real construction) ===")
    print("Starting from single-bit differences, tracking Hamming weight over 8 real Feistel rounds.")
    print("Looking for any trail that diffuses anomalously slowly (a sign of a usable differential).\n")

    n = ctp.LATTICE_N
    n_trials = 15
    all_weight_traces = []
    for trial in range(n_trials):
        seed = os.urandom(32)
        flip_pos = (
            np.random.randint(0, n),
            np.random.randint(0, n),
            np.random.randint(0, n),
        )
        weights = track_difference_trail(seed, flip_pos, n_rounds=8, n=n)
        all_weight_traces.append(weights)

    total_bits = n ** 3
    print(f"{'trial':>6}  " + "  ".join(f"r{i+1}" for i in range(8)))
    for i, w in enumerate(all_weight_traces):
        pct = [f"{100*x/total_bits:5.2f}%" for x in w]
        print(f"{i:>6}  " + "  ".join(pct))

    final_round_pcts = [100 * w[-1] / total_bits for w in all_weight_traces]
    print(f"\nFinal-round (round 8) difference weight across {n_trials} trials:")
    print(f"  mean: {np.mean(final_round_pcts):.2f}%  min: {np.min(final_round_pcts):.2f}%  "
          f"max: {np.max(final_round_pcts):.2f}%  (ideal: 50.00%)")

    slow_trials = [i for i, w in enumerate(all_weight_traces) if w[-1] / total_bits < 0.40]
    print(f"\nTrials with anomalously low final-round weight (<40%, a candidate slow-diffusing trail): "
          f"{slow_trials if slow_trials else 'none found'}")
