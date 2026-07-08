"""
Direct test of the epoch-based self-healing fix (ctp_cipher.py, epoch_size/
epoch_burn_in), against the exact defect found earlier: a single lost
packet permanently and silently corrupted every subsequent packet in the
session, with the HMAC tag verifying "successfully" throughout since it
does not depend on ratchet synchronization.
"""

import os
from ctp_cipher import CTP, AuthenticationError, ReplayError


def test_no_loss_baseline(epoch_size=8):
    """Sanity check: with no loss, decryption is correct across multiple
    epoch boundaries (confirms both sides transition epochs identically
    when nothing goes wrong)."""
    key = os.urandom(32)
    alice = CTP(key, burn_in=8, epoch_size=epoch_size)
    bob = CTP(key, nonce=alice.nonce, burn_in=8, epoch_size=epoch_size)

    n_packets = epoch_size * 3 + 2  # spans multiple epoch boundaries
    msgs = [f"packet {i}".encode() for i in range(n_packets)]
    all_ok = True
    for i, m in enumerate(msgs):
        pkt = alice.encrypt(m)
        recovered = bob.decrypt(pkt)
        ok = recovered == m
        all_ok &= ok
        if not ok:
            print(f"  packet {i}: MISMATCH (no-loss baseline should never fail)")
    print(f"No-loss baseline across {n_packets} packets, {n_packets // epoch_size + 1} epochs: "
          f"{'ALL CORRECT' if all_ok else 'FAILURES DETECTED'}")
    return all_ok


def test_single_loss_recovers_at_next_epoch(epoch_size=8, loss_position=2):
    """The key test: reproduce the original defect scenario (drop one
    packet), and confirm decryption is corrupted only for the remainder of
    the epoch in which the loss occurred, then recovers automatically and
    silently at the next epoch boundary -- no resync message, no
    request/response, no re-key."""
    key = os.urandom(32)
    alice = CTP(key, burn_in=8, epoch_size=epoch_size)
    bob = CTP(key, nonce=alice.nonce, burn_in=8, epoch_size=epoch_size)

    n_packets = epoch_size * 2  # loss in epoch 0, verify recovery in epoch 1
    msgs = [f"packet {i}".encode() for i in range(n_packets)]
    packets = [alice.encrypt(m) for m in msgs]

    results = {}
    for i, pkt in enumerate(packets):
        if i == loss_position:
            continue  # simulate loss: bob never sees this packet
        try:
            results[i] = bob.decrypt(pkt)
        except (AuthenticationError, ReplayError) as e:
            results[i] = f"REJECTED({type(e).__name__})"

    print(f"\nLoss simulated at packet {loss_position} (epoch_size={epoch_size}):")
    corrupted_in_epoch_0 = []
    correct_in_epoch_1_plus = True
    for i in range(n_packets):
        if i == loss_position:
            continue
        epoch = i // epoch_size
        correct = results[i] == msgs[i]
        marker = "OK" if correct else "GARBAGE/REJECTED"
        if epoch == 0 and i > loss_position and not correct:
            corrupted_in_epoch_0.append(i)
        if epoch >= 1 and not correct:
            correct_in_epoch_1_plus = False
        print(f"  packet {i:>3} (epoch {epoch}): {marker}")

    print(f"\nPackets corrupted after the loss, within epoch 0 (expected, bounded blast radius): "
          f"{corrupted_in_epoch_0}")
    print("Epoch 1 onward fully recovered with no explicit resync message: "
          f"{'YES' if correct_in_epoch_1_plus else 'NO -- FIX DID NOT WORK'}")
    return correct_in_epoch_1_plus


def test_reordering_within_epoch_still_breaks(epoch_size=8):
    """Honest documentation of the remaining limitation: this fix bounds
    LOSS to one epoch, but does not fix REORDERING within an epoch. This
    test is expected to show corruption -- it is not a bug, it is the
    documented boundary of what epoch_size fixes."""
    key = os.urandom(32)
    alice = CTP(key, burn_in=8, epoch_size=epoch_size)
    bob = CTP(key, nonce=alice.nonce, burn_in=8, epoch_size=epoch_size)

    msgs = [f"packet {i}".encode() for i in range(4)]
    packets = [alice.encrypt(m) for m in msgs]

    # Deliver packet 1 before packet 0 (reordering, not loss -- everything
    # eventually arrives, just out of order).
    order = [1, 0, 2, 3]
    all_correct = True
    for i in order:
        try:
            recovered = bob.decrypt(packets[i])
            correct = recovered == msgs[i]
        except (AuthenticationError, ReplayError):
            correct = False
        all_correct &= correct

    print(f"\nReordering within one epoch (delivery order {order}): "
          f"{'still fully correct (unexpected!)' if all_correct else 'corrupted, as documented -- reordering is NOT fixed by epoch_size'}")


if __name__ == "__main__":
    test_no_loss_baseline()
    recovered = test_single_loss_recovers_at_next_epoch()
    test_reordering_within_epoch_still_breaks()

    print("\n" + "=" * 60)
    print("OVERALL: epoch-based self-healing fix " + ("WORKS AS DESIGNED" if recovered else "FAILED -- DO NOT TRUST"))
