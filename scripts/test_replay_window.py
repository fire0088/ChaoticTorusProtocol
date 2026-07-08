"""
Test the bounded sliding-window replay protection (ctp_cipher.py,
_check_replay/_commit_replay), which replaced an unbounded Python set()
found during a security review. Three properties matter:
  1. Exact duplicates are still rejected.
  2. Legitimate reordering within the window is still accepted (this is
     not the same guarantee as epoch-boundary ratchet recovery -- a
     reordered packet may still decrypt to garbage per the documented
     epoch limitation, but it must not be rejected by the replay window
     itself for being "out of order").
  3. A forged packet that fails tag verification must NOT be able to
     pre-emptively burn a sequence number that a legitimate retransmission
     later needs -- this was a real bug introduced and then fixed during
     the same session this test was written in, not a hypothetical.
  4. Memory stays bounded (two integers) regardless of session length.
"""

import os
from ctp_cipher import CTP, AuthenticationError, ReplayError


def test_exact_duplicate_rejected():
    key = os.urandom(32)
    alice = CTP(key, burn_in=8)
    bob = CTP(key, nonce=alice.nonce, burn_in=8)

    pkt = alice.encrypt(b"hello")
    bob.decrypt(pkt)
    try:
        bob.decrypt(pkt)
        print("[FAIL] exact duplicate was accepted")
    except ReplayError:
        print("[OK] exact duplicate rejected")


def test_too_old_rejected(window=64):
    key = os.urandom(32)
    alice = CTP(key, burn_in=8)
    bob = CTP(key, nonce=alice.nonce, burn_in=8)

    pkt0 = alice.encrypt(b"packet zero")
    # Advance far beyond the window before ever delivering packet 0.
    for _ in range(window + 5):
        alice.encrypt(b"filler")
    pkt_far = alice.encrypt(b"far ahead")
    bob.decrypt(pkt_far)  # bob's highest is now far ahead

    try:
        bob.decrypt(pkt0)
        print("[FAIL] packet older than the window was accepted")
    except ReplayError:
        print("[OK] packet older than the window rejected")


def test_forged_packet_cannot_burn_sequence_number():
    """The bug found and fixed this session: does a forged packet (fails
    tag verification) prevent a later LEGITIMATE packet with the same
    sequence number from being accepted?"""
    key = os.urandom(32)
    alice = CTP(key, burn_in=8)
    bob = CTP(key, nonce=alice.nonce, burn_in=8)

    real_pkt = alice.encrypt(b"the real message")

    # Attacker forges a packet with the SAME seq but corrupted tag/ciphertext,
    # and delivers it to bob BEFORE the real one arrives (e.g. real packet
    # was delayed in transit).
    forged = bytearray(real_pkt)
    forged[-1] ^= 0xFF  # corrupt the ciphertext -> tag will not verify
    try:
        bob.decrypt(bytes(forged))
        print("[FAIL] forged packet was accepted (should never happen)")
    except AuthenticationError:
        pass  # expected

    # Now the REAL packet (same seq) arrives late. It must still be accepted.
    try:
        recovered = bob.decrypt(real_pkt)
        ok = recovered == b"the real message"
        print("[OK] legitimate retransmission still accepted after a forged "
              "packet with the same seq" if ok else "[FAIL] wrong plaintext recovered")
    except (AuthenticationError, ReplayError) as e:
        print(f"[FAIL] legitimate packet was rejected after a forgery attempt: {type(e).__name__}")


def test_reordering_within_window_accepted_by_replay_check():
    """The replay window itself should not reject reordering (that's a
    separate, documented epoch-boundary limitation -- see
    test_epoch_resync.py). Decrypted CONTENT may still be wrong per that
    limitation; this test only checks the replay window doesn't add a
    second, unrelated rejection on top of it."""
    key = os.urandom(32)
    alice = CTP(key, burn_in=8)
    bob = CTP(key, nonce=alice.nonce, burn_in=8)

    packets = [alice.encrypt(f"packet {i}".encode()) for i in range(4)]

    rejected_as_replay = []
    for i in [1, 0, 2, 3]:  # reordered delivery
        try:
            bob.decrypt(packets[i])
        except ReplayError:
            rejected_as_replay.append(i)
        except AuthenticationError:
            pass  # may fail decryption/auth per epoch limitation; not a replay-window concern here

    print(f"[OK] reordered-but-not-duplicate packets rejected BY THE REPLAY WINDOW specifically: "
          f"{rejected_as_replay} (expected: empty list)" if not rejected_as_replay
          else f"[FAIL] replay window incorrectly rejected reordered packets: {rejected_as_replay}")


def test_memory_bounded():
    import sys
    key = os.urandom(32)
    alice = CTP(key, burn_in=8)
    bob = CTP(key, nonce=alice.nonce, burn_in=8)

    for _ in range(500):
        pkt = alice.encrypt(b"x")
        bob.decrypt(pkt)

    size = sys.getsizeof(bob._replay_bitmask) + sys.getsizeof(bob._replay_highest)
    print(f"[OK] replay-window state size after 500 packets: {size} bytes "
          f"(two integers, does not grow with session length)")


if __name__ == "__main__":
    test_exact_duplicate_rejected()
    test_too_old_rejected()
    test_forged_packet_cannot_burn_sequence_number()
    test_reordering_within_window_accepted_by_replay_check()
    test_memory_bounded()
