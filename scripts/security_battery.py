"""
Security test battery for CTP's keystream generator (the thing that
actually matters cryptographically -- what the attacker sees). Implements
a subset of the NIST SP 800-22 statistical test suite from scratch
(monobit, block frequency, runs, longest-run, DFT/spectral, cumulative
sums, approximate entropy), plus full-pipeline avalanche tests (not just
the lattice in isolation, which was already tested separately), a
compression-based sanity check, and an exact (brute-force, not claimed)
verification of chi's algebraic degree via its Zhegalkin/ANF transform.

This is what Section "Known Limitations" of the paper has been listing as
an outstanding requirement since Revision 2. Running it doesn't make CTP
production-ready -- a real security review needs the full NIST STS/TestU01
batteries at much larger sample sizes and, more importantly, actual
third-party cryptanalysis -- but it's real evidence rather than a promise.
"""

import gzip
import math
import os
import struct

import numpy as np
from scipy.special import erfc, gammaincc

from ctp_cipher import CTP, AuthenticationError


# ---------------------------------------------------------------------------
# Keystream generation: encrypt all-zero plaintext, so ciphertext == keystream
# ---------------------------------------------------------------------------

def generate_keystream(key: bytes, total_bytes: int, packet_size: int = 16384,
                        evolve_every: int = 16, burn_in: int = 64) -> bytes:
    ctp = CTP(key, burn_in=burn_in, evolve_every=evolve_every)
    out = bytearray()
    while len(out) < total_bytes:
        n = min(packet_size, total_bytes - len(out))
        pkt = ctp.encrypt(b"\x00" * n)
        ciphertext = pkt[40:]  # strip seq(8) + tag(32) header
        out.extend(ciphertext)
    return bytes(out[:total_bytes])


def bits_from_bytes(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.int8)


# ---------------------------------------------------------------------------
# NIST SP 800-22 style tests (implemented directly, not via external library)
# ---------------------------------------------------------------------------

def monobit_test(bits: np.ndarray) -> float:
    n = len(bits)
    s = np.sum(2 * bits.astype(np.int64) - 1)
    s_obs = abs(s) / math.sqrt(n)
    return erfc(s_obs / math.sqrt(2))


def block_frequency_test(bits: np.ndarray, block_size: int = 128) -> float:
    n = len(bits)
    n_blocks = n // block_size
    bits = bits[: n_blocks * block_size].reshape(n_blocks, block_size)
    proportions = bits.mean(axis=1)
    chi_sq = 4 * block_size * np.sum((proportions - 0.5) ** 2)
    return gammaincc(n_blocks / 2, chi_sq / 2)


def runs_test(bits: np.ndarray) -> float:
    n = len(bits)
    pi = bits.mean()
    if abs(pi - 0.5) >= (2 / math.sqrt(n)):
        return 0.0  # test not applicable; frequency test would already fail
    v_obs = 1 + np.sum(bits[1:] != bits[:-1])
    num = abs(v_obs - 2 * n * pi * (1 - pi))
    den = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    return erfc(num / den)


def longest_run_test(bits: np.ndarray) -> float:
    """M=128 configuration (requires n >= 6272)."""
    M = 128
    n = len(bits)
    n_blocks = n // M
    if n_blocks < 49:
        return None
    bits = bits[: n_blocks * M].reshape(n_blocks, M)

    v_categories = [0, 0, 0, 0, 0, 0]  # <=4,5,6,7,8,>=9  (bin edges per NIST for M=128... using M=8 table for compactness)
    # NIST table for M=128: categories are <=4,5,6,7,8,>=9 with these pi:
    pi_vals = [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]

    def longest_run_in_block(block):
        max_run = 0
        cur = 0
        for b in block:
            if b == 1:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 0
        return max_run

    for block in bits:
        lr = longest_run_in_block(block)
        if lr <= 4:
            v_categories[0] += 1
        elif lr == 5:
            v_categories[1] += 1
        elif lr == 6:
            v_categories[2] += 1
        elif lr == 7:
            v_categories[3] += 1
        elif lr == 8:
            v_categories[4] += 1
        else:
            v_categories[5] += 1

    chi_sq = sum(
        (v_categories[i] - n_blocks * pi_vals[i]) ** 2 / (n_blocks * pi_vals[i])
        for i in range(6)
    )
    return gammaincc(5 / 2, chi_sq / 2)


def dft_spectral_test(bits: np.ndarray) -> float:
    n = len(bits)
    x = 2 * bits.astype(np.float64) - 1
    fft_vals = np.fft.fft(x)
    magnitudes = np.abs(fft_vals[: n // 2])
    threshold = math.sqrt(math.log(1 / 0.05) * n)
    n0 = 0.95 * n / 2
    n1 = np.sum(magnitudes < threshold)
    d = (n1 - n0) / math.sqrt(n * 0.95 * 0.05 / 4)
    return erfc(abs(d) / math.sqrt(2))


def cumulative_sums_test(bits: np.ndarray, mode: str = "forward") -> float:
    n = len(bits)
    x = 2 * bits.astype(np.int64) - 1
    if mode == "backward":
        x = x[::-1]
    s = np.cumsum(x)
    z = np.max(np.abs(s))

    def norm_cdf(v):
        return 0.5 * (1 + math.erf(v / math.sqrt(2)))

    total = 0.0
    start1 = int((-n / z + 1) / 4) if z != 0 else 0
    end1 = int((n / z - 1) / 4) if z != 0 else 0
    for k in range(start1, end1 + 1):
        total += norm_cdf((4 * k + 1) * z / math.sqrt(n)) - norm_cdf((4 * k - 1) * z / math.sqrt(n))
    start2 = int((-n / z - 3) / 4) if z != 0 else 0
    end2 = int((n / z - 1) / 4) if z != 0 else 0
    total2 = 0.0
    for k in range(start2, end2 + 1):
        total2 += norm_cdf((4 * k + 3) * z / math.sqrt(n)) - norm_cdf((4 * k + 1) * z / math.sqrt(n))
    p = 1 - total + total2
    return max(0.0, min(1.0, p))


def approximate_entropy_test(bits: np.ndarray, m: int = 2) -> float:
    n = len(bits)

    def phi(m_val):
        padded = np.concatenate([bits, bits[: m_val - 1]])
        counts = {}
        for i in range(n):
            pattern = tuple(padded[i : i + m_val])
            counts[pattern] = counts.get(pattern, 0) + 1
        c = np.array(list(counts.values())) / n
        return np.sum(c * np.log(c))

    ap_en = phi(m) - phi(m + 1)
    chi_sq = 2 * n * (math.log(2) - ap_en)
    return gammaincc(2 ** (m - 1), chi_sq / 2)


# ---------------------------------------------------------------------------
# Additional checks beyond the NIST battery
# ---------------------------------------------------------------------------

def chi_square_byte_uniformity(data: bytes) -> float:
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    expected = len(data) / 256
    chi_sq = np.sum((counts - expected) ** 2 / expected)
    return gammaincc(255 / 2, chi_sq / 2)


def compression_ratio_test(data: bytes) -> float:
    compressed = gzip.compress(data, compresslevel=9)
    return len(compressed) / len(data)  # should be ~= 1.0 (no compression gain) for random data


def full_pipeline_avalanche(key: bytes, trials: int = 30, plaintext_len: int = 4096):
    """Avalanche at the level the attacker actually sees: full CTP.encrypt()
    output, not the raw lattice. Nonce is held FIXED across the compared
    pair so the only difference between sessions is the flipped key bit --
    otherwise a fresh random nonce per session (now generated by default,
    see the v0.7 nonce fix) would confound the key's own contribution."""
    key_diffs = []
    for _ in range(trials):
        k1 = bytearray(os.urandom(32))
        k2 = bytearray(k1)
        k2[os.urandom(1)[0] % 32] ^= (1 << (os.urandom(1)[0] % 8))
        pt = os.urandom(plaintext_len)
        shared_nonce = os.urandom(CTP.NONCE_LEN)

        c1 = CTP(bytes(k1), nonce=shared_nonce, burn_in=64).encrypt(pt)[40:]
        c2 = CTP(bytes(k2), nonce=shared_nonce, burn_in=64).encrypt(pt)[40:]
        key_diffs.append(100.0 * sum(bin(a ^ b).count("1") for a, b in zip(c1, c2)) / (len(c1) * 8))

    return np.mean(key_diffs), np.std(key_diffs)


# ---------------------------------------------------------------------------
# Exact algebraic degree of chi via brute-force ANF (Zhegalkin) transform
# ---------------------------------------------------------------------------

def zhegalkin_transform(truth_table: np.ndarray) -> np.ndarray:
    """Fast Mobius transform over GF(2) -- converts a truth table into its
    Algebraic Normal Form coefficients. Standard technique, not a claim."""
    anf = truth_table.copy().astype(np.uint8)
    n = len(anf)
    step = 1
    while step < n:
        for i in range(0, n, step * 2):
            for j in range(i, i + step):
                anf[j + step] ^= anf[j]
        step *= 2
    return anf


def chi_algebraic_degree():
    """chi': y = x XOR (a AND b) XOR (c AND d AND e), 6 inputs total
    (x, a, b, c, d, e). Enumerate all 64 input combinations, compute y,
    build the truth table, transform to ANF, report exact degree (max
    popcount among monomials with a nonzero coefficient)."""
    n_vars = 6
    truth_table = np.zeros(2 ** n_vars, dtype=np.uint8)
    for idx in range(2 ** n_vars):
        bits = [(idx >> i) & 1 for i in range(n_vars)]
        x, a, b, c, d, e = bits
        y = x ^ (a & b) ^ (c & d & e)
        truth_table[idx] = y

    anf = zhegalkin_transform(truth_table)
    degree = 0
    monomials = []
    for idx in range(2 ** n_vars):
        if anf[idx]:
            deg = bin(idx).count("1")
            degree = max(degree, deg)
            monomials.append((idx, deg))
    return degree, monomials, n_vars


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Generating keystream sample (1 MB, default evolve_every=16) ===")
    key = os.urandom(32)
    sample = generate_keystream(key, total_bytes=1_000_000)
    bits = bits_from_bytes(sample)
    print(f"Sample size: {len(sample)} bytes = {len(bits)} bits\n")

    print("=== NIST SP 800-22 style statistical tests (alpha = 0.01) ===")
    tests = [
        ("Monobit (frequency)", monobit_test(bits)),
        ("Block frequency (M=128)", block_frequency_test(bits)),
        ("Runs", runs_test(bits)),
        ("Longest run of ones (M=128)", longest_run_test(bits)),
        ("DFT / spectral", dft_spectral_test(bits)),
        ("Cumulative sums (forward)", cumulative_sums_test(bits, "forward")),
        ("Cumulative sums (backward)", cumulative_sums_test(bits, "backward")),
        ("Approximate entropy (m=2)", approximate_entropy_test(bits, m=2)),
    ]
    for name, p in tests:
        if p is None:
            print(f"  {name:32s}  SKIPPED (insufficient data)")
        else:
            verdict = "PASS" if p >= 0.01 else "FAIL"
            print(f"  {name:32s}  p = {p:.6f}   [{verdict}]")

    print("\n=== Additional checks ===")
    p_chi2 = chi_square_byte_uniformity(sample)
    print(f"  Byte-value uniformity (chi-sq)   p = {p_chi2:.6f}   [{'PASS' if p_chi2 >= 0.01 else 'FAIL'}]")

    ratio = compression_ratio_test(sample)
    print(f"  gzip compression ratio           {ratio:.4f}   [{'PASS (no compression gain)' if ratio > 0.999 else 'FAIL (compressible!)'}]")

    print("\n=== Full-pipeline avalanche (actual ciphertext, not raw lattice) ===")
    kmean, kstd = full_pipeline_avalanche(key, trials=30)
    print(f"  1-bit key flip (nonce held fixed): mean={kmean:.2f}%  std={kstd:.2f}%  (ideal 50%)")
    print(f"  (plaintext-flip avalanche is not a meaningful test for a stream cipher --")
    print(f"   see pipeline_followup_test.py for why, and for the correct analogous test:")
    print(f"   packet-to-packet keystream independence within one evolving session.)")

    print("\n=== Exact algebraic degree of chi (brute-force ANF, not claimed) ===")
    degree, monomials, n_vars = chi_algebraic_degree()
    print(f"  Variables: {n_vars}   Nonzero ANF monomials: {len(monomials)}")
    print(f"  Exact algebraic degree: {degree}")
