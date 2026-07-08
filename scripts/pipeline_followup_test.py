"""
Follow-up to security_battery.py's flawed "plaintext avalanche" test.

That test held (key, state) fixed and flipped a plaintext bit -- which, for
ANY synchronous stream cipher (CTP's Data Plane, AES-CTR, ChaCha20, RC4),
mathematically MUST produce a ciphertext difference of exactly the flipped
bits and nothing else, because C = M XOR Keystream(key, state) and the
keystream term cancels out identically when comparing two ciphertexts
under the same (key, state). A near-0% result there isn't a defect; it's
confirmation of correct XOR determinism. Testing "plaintext avalanche" in
the traditional block-cipher sense does not apply to stream ciphers at all.

But that same mathematical fact is precisely what makes nonce/state reuse
catastrophic for a stream cipher: if (key, state) EVER repeats across two
different plaintexts, XORing the ciphertexts recovers the XOR of the
plaintexts directly, with no cryptographic effort. This module (1) makes
that failure mode concrete and explicit rather than leaving it implicit,
and (2) runs the test that's actually meaningful for a stream cipher:
whether DIFFERENT packets within one normal, correctly-operating session
(sequence advancing, ratchet and lattice evolving) produce independent-
looking keystream, which is what packet_independence_test checks.
"""

import os
import numpy as np

from ctp_cipher import CTP


def demonstrate_two_time_pad(plaintext_len: int = 256):
    """Deliberately reuse (key, nonce) together -- the one thing CTP must
    never let happen in real operation -- to make the resulting break
    concrete. As of the v0.7 nonce fix, key reuse ALONE (with a fresh,
    distinct nonce) no longer collides -- this demo forces nonce reuse
    too, explicitly, to show the residual case that remains fundamentally
    catastrophic for any stream cipher, CTP included."""
    key = os.urandom(32)
    shared_nonce = os.urandom(CTP.NONCE_LEN)
    m1 = os.urandom(plaintext_len)
    m2 = os.urandom(plaintext_len)

    # Two fresh CTP sessions, same key AND same nonce (forced): both start
    # at seq=0 with identical derived keys and identical ratchet-init state.
    c1 = CTP(key, nonce=shared_nonce, burn_in=64).encrypt(m1)[40:]
    c2 = CTP(key, nonce=shared_nonce, burn_in=64).encrypt(m2)[40:]

    ct_xor = bytes(a ^ b for a, b in zip(c1, c2))
    pt_xor = bytes(a ^ b for a, b in zip(m1, m2))
    recovered_matches = ct_xor == pt_xor

    print("=== Two-time-pad demonstration (deliberate key AND nonce reuse) ===")
    print(f"Ciphertext XOR == Plaintext XOR: {recovered_matches}")
    if recovered_matches:
        print("Confirmed: reusing (key, nonce) together leaks the XOR of the "
              "two plaintexts directly -- with crib-dragging or any partial "
              "knowledge of one message, the other is fully recoverable. "
              "This is not a CTP-specific defect; it is true of every "
              "synchronous stream cipher (AES-CTR, ChaCha20, RC4 included). "
              "The v0.7 fix (ctp_cipher.py) means key reuse ALONE no longer "
              "causes this -- see the [OK] check in ctp_cipher.py's own "
              "self-test -- but reusing the full (key, nonce) pair together "
              "remains, and will always remain, catastrophic. Nonce "
              "uniqueness per session is still the sender's responsibility.")


def packet_independence_test(trials: int = 30, packet_size: int = 4096):
    """The test that's actually meaningful for a stream cipher: within ONE
    correctly-operating session, do successive packets' keystreams look
    independent of each other? This is what 'avalanche' should mean here --
    not plaintext-to-ciphertext propagation (inapplicable to stream
    ciphers), but state-to-state decorrelation as the session advances."""
    diffs = []
    for _ in range(trials):
        key = os.urandom(32)
        ctp = CTP(key, burn_in=64, evolve_every=16)
        zeros = b"\x00" * packet_size

        pkt0 = ctp.encrypt(zeros)[40:]   # keystream for seq=0
        pkt1 = ctp.encrypt(zeros)[40:]   # keystream for seq=1 (state advanced)

        diff = 100.0 * sum(bin(a ^ b).count("1") for a, b in zip(pkt0, pkt1)) / (len(pkt0) * 8)
        diffs.append(diff)

    return np.mean(diffs), np.std(diffs), min(diffs), max(diffs)


if __name__ == "__main__":
    demonstrate_two_time_pad()

    print("\n=== Packet-to-packet keystream independence (same session, seq 0 vs seq 1) ===")
    mean, std, lo, hi = packet_independence_test(trials=30)
    print(f"mean={mean:.2f}%  std={std:.2f}%  min={lo:.2f}%  max={hi:.2f}%  (ideal: 50%)")
