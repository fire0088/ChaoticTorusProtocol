"""
Tests for ctp_session_manager.py:
  1. Multiplexing: multiple sessions interleaved over one "channel" route
     correctly by CID.
  2. Epoch-boundary rotation: a session's CID changes at each epoch, and
     the receiver's pre-registered lookahead lets a new-epoch packet route
     correctly with no extra signaling.
  3. The off-path spoofing attack: does the naive (no path validation)
     manager actually have the vulnerability its own docstring predicts?
     Demonstrated directly, not asserted.
"""

import os

from ctp_cipher import CTP, AuthenticationError, ReplayError
from ctp_session_manager import (
    CTPSession, NaiveCTPSessionManager, ValidatingCTPSessionManager,
    PathValidationError, PathValidationThrottled, respond_to_path_challenge, compute_cid,
)


def test_multiplexing():
    """Two independent sessions, packets interleaved, must each route to
    the correct session by CID alone."""
    key_a, key_b = os.urandom(32), os.urandom(32)
    alice = CTPSession(CTP(key_a, burn_in=8))
    carol = CTPSession(CTP(key_b, burn_in=8))

    bob_mgr = NaiveCTPSessionManager()
    dave_mgr_for_carol = bob_mgr  # same manager handles both sessions

    # Receiver needs the matching CTP objects (nonce-shared) to decrypt each session.
    bob_side_a = CTPSession(CTP(key_a, nonce=alice.ctp.nonce, burn_in=8))
    bob_side_c = CTPSession(CTP(key_b, nonce=carol.ctp.nonce, burn_in=8))
    bob_mgr.register_session(bob_side_a)
    bob_mgr.register_session(bob_side_c)

    msgs_a = [f"alice msg {i}".encode() for i in range(4)]
    msgs_c = [f"carol msg {i}".encode() for i in range(4)]

    packets = []
    for ma, mc in zip(msgs_a, msgs_c):
        packets.append(("A", alice.encrypt(ma), ma))
        packets.append(("C", carol.encrypt(mc), mc))

    all_ok = True
    for label, pkt, expected in packets:
        plaintext, session = bob_mgr.receive_packet(pkt, source_address=f"addr-{label}")
        correct = plaintext == expected
        matched_right_session = (session is bob_side_a) if label == "A" else (session is bob_side_c)
        all_ok &= correct and matched_right_session

    print("[OK]" if all_ok else "[FAIL]", "multiplexing: interleaved packets from two sessions "
          "route to the correct session by CID")


def test_epoch_rotation_routing(epoch_size: int = 4):
    """CID changes at each epoch boundary; receiver's lookahead registration
    should let this happen with no extra signaling."""
    key = os.urandom(32)
    alice = CTPSession(CTP(key, burn_in=8, epoch_size=epoch_size))
    bob_side = CTPSession(CTP(key, nonce=alice.ctp.nonce, burn_in=8, epoch_size=epoch_size))

    mgr = NaiveCTPSessionManager()
    mgr.register_session(bob_side)

    n_packets = epoch_size * 3  # spans multiple epoch boundaries
    seen_cids = set()
    all_ok = True
    for i in range(n_packets):
        msg = f"packet {i}".encode()
        pkt = alice.encrypt(msg)
        cid = pkt[:16]
        seen_cids.add(cid)
        plaintext, session = mgr.receive_packet(pkt, source_address="addr-alice")
        all_ok &= (plaintext == msg)

    n_epochs_spanned = n_packets // epoch_size + 1
    print(f"[OK]" if all_ok else "[FAIL]", f"epoch rotation: all {n_packets} packets across "
          f"~{len(seen_cids)} distinct CIDs (spanning multiple epochs) routed correctly "
          f"with no extra signaling")


def test_cid_spoofing_attack():
    """The question this whole feature was built to answer honestly: does
    the naive (no path validation) manager actually let an off-path
    attacker -- someone who can observe/relay packets in transit but has
    NO key material at all -- hijack the address association and deny
    service to the legitimate sender?

    Simulated attack: attacker intercepts a genuine, valid, in-flight
    packet (captured, not forged -- they have no way to forge anything
    without keys) and relays an exact copy of it to the receiver from a
    DIFFERENT claimed source address, racing it ahead of the real delivery.
    """
    key = os.urandom(32)
    alice = CTPSession(CTP(key, burn_in=8))
    bob_side = CTPSession(CTP(key, nonce=alice.ctp.nonce, burn_in=8))

    mgr = NaiveCTPSessionManager()
    mgr.register_session(bob_side)

    msg = b"legitimate message from alice"
    packet = alice.encrypt(msg)  # Alice's real packet, in flight to Bob

    # --- Attacker intercepts this exact packet and relays it first, from a
    # spoofed address, BEFORE Bob's manager ever sees Alice's real delivery.
    # The attacker needs no key material at all -- this is a pure relay of
    # bytes they observed on the wire.
    spoofed_address = "attacker-controlled-address"
    print("\n=== Off-path CID spoofing / address-hijack attack (NAIVE manager) ===")
    try:
        plaintext, session = mgr.receive_packet(packet, source_address=spoofed_address)
        attacker_relay_accepted = True
    except (AuthenticationError, ReplayError):
        attacker_relay_accepted = False

    address_after_attack = mgr.session_address.get(packet[:16])
    hijacked = (address_after_attack == spoofed_address)

    print(f"Attacker's relayed copy accepted: {attacker_relay_accepted}")
    print(f"Session's 'current address' after attack: {address_after_attack!r}")
    print(f"Address successfully hijacked to attacker's claimed address: {hijacked}")

    # --- Now Alice's REAL delivery arrives (same bytes, her actual send).
    try:
        mgr.receive_packet(packet, source_address="alices-real-address")
        legit_delivery_accepted = True
        legit_rejection_reason = None
    except (AuthenticationError, ReplayError) as e:
        legit_delivery_accepted = False
        legit_rejection_reason = type(e).__name__

    print(f"Alice's legitimate (real) delivery of the SAME packet accepted: {legit_delivery_accepted}"
          + (f"  (rejected as: {legit_rejection_reason})" if legit_rejection_reason else ""))

    print()
    if hijacked and not legit_delivery_accepted:
        print("VULNERABILITY CONFIRMED (naive manager): an off-path attacker with NO key")
        print("material hijacked the address association, and the legitimate sender's real")
        print("packet was then rejected as a replay -- a real denial-of-service vector.")
    else:
        print("Attack did not succeed as predicted -- see details above.")

    return hijacked and not legit_delivery_accepted


def test_cid_spoofing_attack_against_validating_manager():
    """The same exact attack, against ValidatingCTPSessionManager. The
    attacker still has no key material -- only now, the manager will not
    trust their claimed address without a correctly-authenticated response
    to a challenge, which the attacker cannot produce."""
    key = os.urandom(32)
    alice = CTPSession(CTP(key, burn_in=8))
    bob_side = CTPSession(CTP(key, nonce=alice.ctp.nonce, burn_in=8))

    mgr = ValidatingCTPSessionManager()
    mgr.register_session(bob_side, initial_trusted_address="alices-real-address")

    msg = b"legitimate message from alice"
    packet = alice.encrypt(msg)

    def attacker_challenge_responder(candidate_address, challenge):
        # The attacker has no k_auth. Best they can do is guess/return
        # garbage -- simulated here as random bytes, which will not match
        # the expected HMAC regardless of what they send.
        return os.urandom(32)

    print("\n=== Same attack, against ValidatingCTPSessionManager ===")
    spoofed_address = "attacker-controlled-address"
    attack_blocked = False
    try:
        mgr.receive_packet(packet, source_address=spoofed_address,
                            deliver_challenge_fn=attacker_challenge_responder)
        print("[FAIL] attacker's relayed copy was accepted -- vulnerability NOT fixed")
    except PathValidationError as e:
        attack_blocked = True
        print(f"[OK] attacker's relayed copy rejected: {e}")

    trusted_after_attack = mgr.trusted_address.get(packet[:16])
    print(f"Trusted address after attack attempt: {trusted_after_attack!r} "
          f"(unchanged from Alice's real address: {trusted_after_attack == 'alices-real-address'})")

    # Alice's real delivery, from her already-trusted address, should work
    # completely normally -- no challenge needed since she's already trusted.
    def unreachable_responder(addr, ch):
        raise AssertionError("challenge should not be issued for an already-trusted address")

    try:
        plaintext, _ = mgr.receive_packet(packet, source_address="alices-real-address",
                                           deliver_challenge_fn=unreachable_responder)
        legit_ok = (plaintext == msg)
        print(f"[OK] Alice's legitimate delivery still succeeds normally: {legit_ok}"
              if legit_ok else "[FAIL] Alice's legitimate delivery was disrupted")
    except Exception as e:
        legit_ok = False
        print(f"[FAIL] Alice's legitimate delivery was disrupted: {type(e).__name__}: {e}")

    fixed = attack_blocked and (trusted_after_attack == "alices-real-address") and legit_ok
    print(f"\nVULNERABILITY FIXED: {fixed}")
    return fixed


def test_legitimate_migration_succeeds():
    """Sanity check the fix doesn't just block everything: a LEGITIMATE
    address change (Alice actually moves networks) should still succeed,
    since the real Alice DOES have k_auth and can answer the challenge
    correctly."""
    key = os.urandom(32)
    alice = CTPSession(CTP(key, burn_in=8))
    bob_side = CTPSession(CTP(key, nonce=alice.ctp.nonce, burn_in=8))

    mgr = ValidatingCTPSessionManager()
    mgr.register_session(bob_side, initial_trusted_address="alices-wifi-address")

    def alice_responder(candidate_address, challenge):
        # The REAL Alice has k_auth and computes the correct response.
        return respond_to_path_challenge(alice.ctp.k_auth, challenge)

    msg = b"alice, now on cellular"
    packet = alice.encrypt(msg)

    print("\n=== Legitimate migration (real address change) ===")
    try:
        plaintext, _ = mgr.receive_packet(packet, source_address="alices-cellular-address",
                                           deliver_challenge_fn=alice_responder)
        ok = (plaintext == msg)
        new_trusted = mgr.trusted_address.get(packet[:16])
        print(f"[OK] legitimate migration succeeded, new trusted address: {new_trusted!r}"
              if ok and new_trusted == "alices-cellular-address"
              else "[FAIL] legitimate migration did not work correctly")
        return ok and new_trusted == "alices-cellular-address"
    except Exception as e:
        print(f"[FAIL] legitimate migration was incorrectly blocked: {type(e).__name__}: {e}")
        return False


def test_challenge_rate_limiting():
    """Verify the resource-exhaustion/harassment concern found during
    review is actually closed: repeated packets claiming the same
    unvalidated candidate address must not trigger a fresh challenge every
    single time."""
    key = os.urandom(32)
    alice = CTPSession(CTP(key, burn_in=8))
    bob_side = CTPSession(CTP(key, nonce=alice.ctp.nonce, burn_in=8))

    mgr = ValidatingCTPSessionManager(challenge_cooldown_seconds=1.0)
    mgr.register_session(bob_side, initial_trusted_address="alices-real-address")

    challenge_count = [0]

    def counting_attacker_responder(candidate_address, challenge):
        challenge_count[0] += 1
        return os.urandom(32)  # still can't produce a correct response

    print("\n=== Challenge rate limiting ===")
    fake_clock = [0.0]
    throttled_count = 0
    validation_failed_count = 0

    for i in range(5):
        msg = f"spoof attempt {i}".encode()
        # Each attempt needs a fresh CTP-level packet (different seq), but
        # the SAME claimed (spoofed) address, arriving in rapid succession.
        packet = alice.encrypt(msg)
        try:
            mgr.receive_packet(packet, source_address="attacker-controlled-address",
                                deliver_challenge_fn=counting_attacker_responder,
                                _now=fake_clock[0])
        except PathValidationThrottled:
            throttled_count += 1
        except PathValidationError:
            validation_failed_count += 1
        fake_clock[0] += 0.1  # 100ms between attempts -- well under the 1s cooldown

    print(f"Attempts made: 5, actual challenges issued: {challenge_count[0]}, "
          f"throttled: {throttled_count}, validation failures: {validation_failed_count}")
    rate_limited_correctly = challenge_count[0] == 1 and throttled_count == 4
    print(f"[OK]" if rate_limited_correctly else "[FAIL]",
          "only the first attempt triggers a real challenge; the rest are throttled")

    # Confirm the cooldown actually expires: after waiting past it, a new
    # attempt SHOULD trigger a fresh challenge (still failing validation,
    # since the attacker still lacks k_auth -- rate limiting bounds abuse,
    # it doesn't grant access).
    fake_clock[0] += 2.0  # past the 1.0s cooldown
    packet = alice.encrypt(b"spoof attempt after cooldown")
    try:
        mgr.receive_packet(packet, source_address="attacker-controlled-address",
                            deliver_challenge_fn=counting_attacker_responder,
                            _now=fake_clock[0])
    except PathValidationError:
        pass
    cooldown_expired_correctly = challenge_count[0] == 2
    print(f"[OK]" if cooldown_expired_correctly else "[FAIL]",
          "after the cooldown expires, a new challenge is issued (rate limiting doesn't "
          "permanently block a candidate address, it just paces retries)")

    return rate_limited_correctly and cooldown_expired_correctly


if __name__ == "__main__":
    test_multiplexing()
    test_epoch_rotation_routing()
    vulnerable_naive = test_cid_spoofing_attack()
    fixed = test_cid_spoofing_attack_against_validating_manager()
    migration_ok = test_legitimate_migration_succeeds()
    rate_limit_ok = test_challenge_rate_limiting()

    print("\n" + "=" * 60)
    print(f"SUMMARY: naive manager vulnerable = {vulnerable_naive}, "
          f"validating manager fixes it = {fixed}, "
          f"legitimate migration still works = {migration_ok}, "
          f"rate limiting works = {rate_limit_ok}")
