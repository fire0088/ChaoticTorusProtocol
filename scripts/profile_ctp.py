"""
Profile CTP's per-packet cost by component: lattice step, lattice packing,
SHAKE-256 absorb+squeeze, HMAC tag, ratchet update. Run for several
payload sizes to see how the balance shifts with size.
"""

import hashlib
import hmac
import os
import struct
import time

import numpy as np

from ctp_cipher import CTP, CarterBaysLattice, hkdf_sha3_256


def timeit(fn, n=20):
    # warm up
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / n


def profile_components(payload_size, burn_in=64):
    key = os.urandom(32)
    c = CTP(key, burn_in=burn_in)
    pt = os.urandom(payload_size)

    # 1. lattice.step() alone
    t_step = timeit(lambda: c.lattice.step(), n=20)

    # 2. packed_bytes() alone
    t_pack = timeit(lambda: c.lattice.packed_bytes(), n=50)

    lattice_bytes = c.lattice.packed_bytes()

    # 3. SHAKE-256 absorb+squeeze for the keystream (dominant cost scales with payload_size)
    material = c.state + lattice_bytes + struct.pack(">Q", 0)
    t_shake = timeit(lambda: hashlib.shake_256(material).digest(payload_size), n=20)

    # 4. HMAC-SHA3-256 tag over ciphertext
    ct = os.urandom(payload_size)
    t_hmac = timeit(lambda: hmac.new(c.k_auth, struct.pack(">Q", 0) + ct, hashlib.sha3_256).digest(), n=20)

    # 5. ratchet update: sha3-256(plaintext) + sha3-256(state||h||seq)  (lattice.step() counted separately above)
    def ratchet_hash_only():
        h_m = hashlib.sha3_256(pt).digest()
        hashlib.sha3_256(c.state + h_m + struct.pack(">Q", 0)).digest()
    t_ratchet_hash = timeit(ratchet_hash_only, n=20)

    total_measured = t_step + t_pack + t_shake + t_hmac + t_ratchet_hash

    return {
        "lattice.step()": t_step,
        "lattice.packed_bytes()": t_pack,
        "SHAKE-256 absorb+squeeze": t_shake,
        "HMAC-SHA3-256 tag": t_hmac,
        "ratchet SHA3 hashing": t_ratchet_hash,
        "SUM (approx per-packet cost)": total_measured,
    }


if __name__ == "__main__":
    for size in [1_024, 16_384, 262_144, 1_048_576]:
        print(f"\n=== payload = {size:>8} bytes (burn_in=64) ===")
        res = profile_components(size)
        total = res["SUM (approx per-packet cost)"]
        for k, v in res.items():
            pct = "" if k.startswith("SUM") else f"  ({100*v/total:5.1f}%)"
            print(f"  {k:32s} {v*1000:9.3f} ms{pct}")
