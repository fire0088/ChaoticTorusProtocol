"""
Benchmark and correctness/security-property comparison:
  CTP v0.2 (research prototype)  vs  AES-256-GCM  vs  ChaCha20-Poly1305

Both reference ciphers use the `cryptography` library's audited,
hardware-accelerated (AES-NI where available) implementations.
"""

import os
import time
import statistics as stats

from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305

from ctp_cipher import CTP, AuthenticationError, ReplayError

PAYLOAD_SIZES = [1_024, 16_384, 262_144, 1_048_576]  # 1KB, 16KB, 256KB, 1MB
TRIALS_PER_SIZE = 5


# ---------------------------------------------------------------------------
# Thin wrappers giving all three ciphers the same encrypt(pt)/decrypt(pkt) shape
# ---------------------------------------------------------------------------

class AESGCMWrapper:
    name = "AES-256-GCM"

    def __init__(self, key: bytes):
        self.aead = AESGCM(key)
        self.seq = 0

    def encrypt(self, pt: bytes) -> bytes:
        nonce = self.seq.to_bytes(12, "big")
        ct = self.aead.encrypt(nonce, pt, None)
        self.seq += 1
        return nonce + ct

    def decrypt(self, pkt: bytes) -> bytes:
        nonce, ct = pkt[:12], pkt[12:]
        return self.aead.decrypt(nonce, ct, None)


class ChaCha20Wrapper:
    name = "ChaCha20-Poly1305"

    def __init__(self, key: bytes):
        self.aead = ChaCha20Poly1305(key)
        self.seq = 0

    def encrypt(self, pt: bytes) -> bytes:
        nonce = self.seq.to_bytes(12, "big")
        ct = self.aead.encrypt(nonce, pt, None)
        self.seq += 1
        return nonce + ct

    def decrypt(self, pkt: bytes) -> bytes:
        nonce, ct = pkt[:12], pkt[12:]
        return self.aead.decrypt(nonce, ct, None)


class CTPWrapper:
    name = "CTP v0.2 (prototype)"

    def __init__(self, key: bytes, burn_in: int = 64):
        self.enc = CTP(key, burn_in=burn_in)
        # decrypt side must share the SAME nonce as encrypt (as two real
        # endpoints would coordinate via the handshake) -- independently
        # constructing both with just the key is no longer sufficient now
        # that nonce defaults to a fresh random value per instance.
        self.dec = CTP(key, nonce=self.enc.nonce, burn_in=burn_in)

    def encrypt(self, pt: bytes) -> bytes:
        return self.enc.encrypt(pt)

    def decrypt(self, pkt: bytes) -> bytes:
        return self.dec.decrypt(pkt)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def bench_cipher(make_cipher, sizes, trials):
    results = {}
    for size in sizes:
        enc_times, dec_times = [], []
        for _ in range(trials):
            cipher = make_cipher()
            pt = os.urandom(size)

            t0 = time.perf_counter()
            pkt = cipher.encrypt(pt)
            t1 = time.perf_counter()
            recovered = cipher.decrypt(pkt)
            t2 = time.perf_counter()

            assert recovered == pt, "round-trip mismatch"
            enc_times.append(t1 - t0)
            dec_times.append(t2 - t1)

        results[size] = {
            "enc_mbps": (size / (1024 * 1024)) / stats.median(enc_times),
            "dec_mbps": (size / (1024 * 1024)) / stats.median(dec_times),
        }
    return results


def print_table(all_results, sizes):
    header = f"{'Payload':>10} | " + " | ".join(f"{name:>24}" for name in all_results)
    print(header)
    print("-" * len(header))
    for size in sizes:
        row = f"{size:>8}B | "
        cells = []
        for name, res in all_results.items():
            r = res[size]
            cells.append(f"enc {r['enc_mbps']:7.2f} MB/s / dec {r['dec_mbps']:7.2f} MB/s")
        print(row + " | ".join(f"{c:>24}" for c in cells))


def security_property_tests():
    print("\n=== Security property spot-checks ===")
    key = os.urandom(32)

    for name, make in [
        ("AES-256-GCM", lambda: (AESGCMWrapper(key), AESGCMWrapper(key))),
        ("ChaCha20-Poly1305", lambda: (ChaCha20Wrapper(key), ChaCha20Wrapper(key))),
        ("CTP v0.2", lambda: (CTPWrapper(key), None)),
    ]:
        print(f"\n-- {name} --")
        if name == "CTP v0.2":
            c = CTPWrapper(key)
            msg = b"integrity test message"
            pkt = bytearray(c.enc.encrypt(msg))
            pkt[-1] ^= 0x01
            try:
                c.dec.decrypt(bytes(pkt))
                print("[FAIL] tampered ciphertext accepted")
            except AuthenticationError:
                print("[OK] tampered ciphertext rejected")

            c2 = CTPWrapper(key)
            msg2 = b"replay test message"
            pkt2 = c2.enc.encrypt(msg2)
            c2.dec.decrypt(pkt2)
            try:
                c2.dec.decrypt(pkt2)
                print("[FAIL] replayed packet accepted")
            except ReplayError:
                print("[OK] replayed packet rejected")
        else:
            sender, receiver = make()
            msg = b"integrity test message"
            pkt = bytearray(sender.encrypt(msg))
            pkt[-1] ^= 0x01
            try:
                receiver.decrypt(bytes(pkt))
                print("[FAIL] tampered ciphertext accepted")
            except Exception:
                print("[OK] tampered ciphertext rejected (InvalidTag)")


def ctp_avalanche_test(trials=30, burn_in=16):
    """
    Implements the avalanche methodology promised in Section 3.5 of the
    v0.2 draft, at reduced trial count for a quick benchmark run (the spec
    calls for >=10,000 trials for a publishable result; this is a smoke test).
    """
    print(f"\n=== CTP avalanche smoke test (n={trials} trials, burn_in={burn_in}) ===")
    diffs = []
    msg = b"fixed reference plaintext for avalanche measurement" * 4
    for _ in range(trials):
        key_a = bytearray(os.urandom(32))
        key_b = bytearray(key_a)
        # flip one random bit
        byte_idx = os.urandom(1)[0] % 32
        bit_idx = os.urandom(1)[0] % 8
        key_b[byte_idx] ^= (1 << bit_idx)

        ca = CTP(bytes(key_a), burn_in=burn_in)
        cb = CTP(bytes(key_b), burn_in=burn_in)

        ks_a = ca._keystream(0, len(msg))
        ks_b = cb._keystream(0, len(msg))

        hamming = sum(bin(a ^ b).count("1") for a, b in zip(ks_a, ks_b))
        diffs.append(100.0 * hamming / (len(msg) * 8))

    mean = stats.mean(diffs)
    stdev = stats.stdev(diffs)
    print(f"mean bit-difference: {mean:.2f}%  stdev: {stdev:.2f}%  (ideal: 50.00%)")
    print(f"min: {min(diffs):.2f}%  max: {max(diffs):.2f}%")


if __name__ == "__main__":
    key = os.urandom(32)

    print("=== Throughput benchmark (median of", TRIALS_PER_SIZE, "trials per size) ===")
    print("NOTE: CTP uses burn_in=64 (spec default is 1024) for benchmark turnaround.\n"
          "This affects only lattice mixing depth, not the AEAD guarantees "
          "(see ctp_cipher.py docstring).\n")

    all_results = {}
    all_results["AES-256-GCM"] = bench_cipher(
        lambda: AESGCMWrapper(key), PAYLOAD_SIZES, TRIALS_PER_SIZE
    )
    all_results["ChaCha20-Poly1305"] = bench_cipher(
        lambda: ChaCha20Wrapper(key), PAYLOAD_SIZES, TRIALS_PER_SIZE
    )
    all_results["CTP v0.2"] = bench_cipher(
        lambda: CTPWrapper(key, burn_in=64), PAYLOAD_SIZES, TRIALS_PER_SIZE
    )

    print_table(all_results, PAYLOAD_SIZES)

    security_property_tests()
    ctp_avalanche_test(trials=30, burn_in=16)
