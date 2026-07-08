"""
Cube tester (Dinur-Shamir, 2009) aimed at CTP's keystream generator as a
whole -- ratchet + lattice + K12 extraction -- not just chi in isolation
(chi's local degree-3 property was already proven exactly via brute-force
ANF computation; this asks a different question: does the FULL composed
system, after burn-in, Feistel rounds, and sponge extraction, actually
reach high algebraic degree in its output, or does some combination of
these steps leave detectable low-degree structure?).

This is the piece Assumption 1 of the formal proof (security_proof.md)
depends on and cannot get from anywhere else: no third party has looked
at this composite construction, so this is new evidence, not a summary
of prior tests.

Method: fix a base point (a random master key + nonce). Choose k specific
bit positions within the master key as "cube" variables. For each of the
2^k assignments to those bits (holding everything else fixed), run the
real CTP encryption pipeline and extract one keystream bit. XOR-sum all
2^k results. If the system's algebraic degree in these k input bits were
truly k (maximal, as expected of a well-mixed construction), this sum
should be close to unbiased (near 50/50 across many independent trials
using different cubes/base points/output bit positions). A systematic
bias indicates the ANF degree in those variables is LESS than k -- a
real structural weakness, the same class of issue that made the original
Carter Bays rule and the original degree-2 chi both suspect.
"""

import os
import struct
import numpy as np

from ctp_cipher import CTP


def run_with_modified_key(base_key: bytes, cube_bit_positions: list, assignment: int,
                           nonce: bytes, plaintext: bytes, burn_in: int, evolve_every: int) -> int:
    """Flip the given cube bit positions in base_key according to
    `assignment` (a k-bit integer), run one real CTP encryption, and
    return the first bit of the resulting ciphertext (== first keystream
    bit, since plaintext is all-zero)."""
    key = bytearray(base_key)
    for i, bit_pos in enumerate(cube_bit_positions):
        if (assignment >> i) & 1:
            byte_idx = bit_pos // 8
            bit_idx = bit_pos % 8
            key[byte_idx] ^= (1 << bit_idx)
    ctp = CTP(bytes(key), nonce=nonce, burn_in=burn_in, evolve_every=evolve_every)
    pkt = ctp.encrypt(plaintext)
    ciphertext = pkt[40:]
    first_byte = ciphertext[0]
    return first_byte & 1  # first bit of first keystream byte


def cube_test_trial(k: int, burn_in: int = 0, evolve_every: int = 1,
                     plaintext_len: int = 64) -> int:
    """One trial: random base key/nonce, random choice of k cube bit
    positions (within the 256-bit master key), sum the output bit over
    all 2^k assignments. Returns the XOR-sum (0 or 1)."""
    base_key = os.urandom(32)
    nonce = os.urandom(CTP.NONCE_LEN)
    plaintext = b"\x00" * plaintext_len
    cube_bit_positions = list(np.random.choice(256, size=k, replace=False))

    xor_sum = 0
    for assignment in range(2 ** k):
        bit = run_with_modified_key(base_key, cube_bit_positions, assignment,
                                     nonce, plaintext, burn_in, evolve_every)
        xor_sum ^= bit
    return xor_sum


if __name__ == "__main__":
    import time
    from scipy import stats

    k = 7          # cube dimension: 2^7 = 128 evaluations per trial
    n_trials = 100  # independent (cube, base point) choices

    print(f"=== Cube tester: k={k} ({2**k} evaluations/trial), {n_trials} independent trials ===")
    print("Targeting the FULL CTP keystream generator (ratchet + lattice + K12),")
    print("not chi in isolation -- checking for detectable sub-maximal algebraic degree.\n")

    t0 = time.time()
    sums = []
    for trial in range(n_trials):
        s = cube_test_trial(k=k, burn_in=0, evolve_every=1)
        sums.append(s)
        if (trial + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  trial {trial+1}/{n_trials}  ({elapsed:.1f}s elapsed)")

    ones = sum(sums)
    zeros = n_trials - ones
    print(f"\nResults: {ones} ones, {zeros} zeros out of {n_trials} trials")

    # Binomial test: is P(sum=1) significantly different from 0.5?
    result = stats.binomtest(ones, n_trials, p=0.5)
    print(f"Binomial test p-value: {result.pvalue:.4f}")
    verdict = "PASS (no detected bias)" if result.pvalue >= 0.01 else "FAIL (significant bias detected!)"
    print(f"Verdict at alpha=0.01: {verdict}")

    print(f"\nTotal wall time: {time.time()-t0:.1f}s")
