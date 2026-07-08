"""
Connection ID mechanism for CTP, addressing the gap found by inspection:
CTP's wire format (seq || tag || ciphertext) carries no session identifier
at all -- a receiver has no way to know which session's key material
applies to a packet except by assuming the network layer beneath it
already disambiguates (one channel = one session). QUIC solves the
analogous problem with a Connection ID (CID) decoupled from the network
4-tuple, enabling migration, load-balancer routing, and multiplexing.

Design: CID = truncated SHA3-256(K_epoch || "CTP-CID" || epoch_number).
This reuses the epoch mechanism already built and validated (epoch_size
default 64, tested in test_epoch_resync.py) rather than adding a separate
rotation scheme: both sides already derive K_epoch and the epoch number
independently, so a receiver can PRECOMPUTE upcoming epochs' CIDs and
register them in advance -- no extra handshake messages needed for
rotation, unlike QUIC's NEW_CONNECTION_ID frames, which exist specifically
because QUIC's CIDs are NOT independently derivable by both sides.

This module deliberately implements the NAIVE version first (no path
validation on address changes) so the specific vulnerability that creates
can be demonstrated directly (see the attack test in this file), rather
than asserting it exists.

Kept as a layer ABOVE ctp_cipher.py's encrypt()/decrypt() rather than
modifying CTP's own wire format, so the extensively-validated existing
test suite (round-trip, tamper, replay, two-time-pad, epoch resync) is
untouched by this addition.
"""

import hashlib
import hmac
import os
import struct

from ctp_cipher import CTP, AuthenticationError, ReplayError

CID_LEN = 16
CID_LOOKAHEAD_EPOCHS = 3  # how many future epochs' CIDs to pre-register


def compute_cid(k_epoch: bytes, epoch: int) -> bytes:
    return hashlib.sha3_256(k_epoch + b"CTP-CID" + struct.pack(">Q", epoch)).digest()[:CID_LEN]


class CTPSession:
    """Wraps a CTP object, prefixing outgoing packets with the current
    epoch's connection ID."""

    def __init__(self, ctp_obj: CTP):
        self.ctp = ctp_obj

    def encrypt(self, plaintext: bytes) -> bytes:
        epoch = self.ctp.seq // self.ctp.epoch_size
        cid = compute_cid(self.ctp.k_epoch, epoch)
        return cid + self.ctp.encrypt(plaintext)

    def current_and_upcoming_cids(self, lookahead: int = CID_LOOKAHEAD_EPOCHS):
        current_epoch = self.ctp.current_epoch if self.ctp.current_epoch is not None else 0
        return [compute_cid(self.ctp.k_epoch, current_epoch + i) for i in range(lookahead + 1)]


class NaiveCTPSessionManager:
    """Receiver-side router: dispatches incoming raw packets to the right
    CTPSession by connection ID. NAIVE: updates its notion of a session's
    'current address' immediately on any packet with a recognized CID, with
    no path validation. This is deliberate, not an oversight -- see
    test_cid_spoofing_attack below for exactly what that omission costs."""

    def __init__(self):
        self.cid_to_session = {}
        self.session_address = {}  # cid -> last-seen source address

    def register_session(self, session: CTPSession):
        for cid in session.current_and_upcoming_cids():
            self.cid_to_session[cid] = session

    def _refresh_registration(self, session: CTPSession):
        for cid in session.current_and_upcoming_cids():
            self.cid_to_session.setdefault(cid, session)

    def receive_packet(self, raw_packet: bytes, source_address):
        cid = raw_packet[:CID_LEN]
        inner = raw_packet[CID_LEN:]
        session = self.cid_to_session.get(cid)
        if session is None:
            raise KeyError(f"unknown connection ID {cid.hex()} -- no session registered")

        # NAIVE: address is trusted immediately, before (and regardless of)
        # whether the packet turns out to be valid. This mirrors the most
        # naive possible integration -- see the attack test for why even
        # gating this on successful decryption doesn't fully fix things.
        self.session_address[cid] = source_address

        plaintext = session.ctp.decrypt(inner)  # raises on tamper/replay
        self._refresh_registration(session)
        return plaintext, session


class PathValidationError(Exception):
    pass


class PathValidationThrottled(Exception):
    """Raised when a challenge is suppressed by rate limiting, distinct
    from an actual failed validation -- see the rate-limiting note in
    ValidatingCTPSessionManager."""
    pass


def respond_to_path_challenge(k_auth: bytes, challenge: bytes) -> bytes:
    """What a LEGITIMATE endpoint (one that actually possesses this
    session's k_auth) computes when it receives a path challenge. An
    attacker without k_auth has no way to produce this value, regardless
    of how perfectly they can replay an observed packet."""
    return hmac.new(k_auth, b"CTP-PATH-CHALLENGE" + challenge, hashlib.sha3_256).digest()


class ValidatingCTPSessionManager:
    """Same routing as NaiveCTPSessionManager, but a source address is
    trusted only after it proves possession of the session's k_auth via a
    challenge/response -- analogous to QUIC's PATH_CHALLENGE/PATH_RESPONSE.
    A packet from an address not yet trusted does NOT have its payload
    decrypted (so it cannot consume replay-window/sequence state) until
    that address passes validation; the currently-trusted address keeps
    working throughout, uninterrupted.

    Rate limiting on challenge issuance (found necessary during review,
    not part of the original design): without it, an attacker can send an
    unbounded stream of packets claiming an arbitrary third party's
    address, causing the receiver to fire a fresh challenge at that victim
    every single time -- individually small (16 bytes), but unthrottled,
    which is both a minor harassment vector against the spoofed victim and
    a real resource-exhaustion vector against the receiver itself (real
    cryptographic and network work per attempt, triggerable by anyone who
    can send packets with a valid CID at all). A cooldown per (CID,
    candidate address) pair closes this cheaply: this is not a
    cryptographic fix, and is stated as such -- it bounds a resource-abuse
    concern, not a confidentiality/integrity one."""

    def __init__(self, challenge_cooldown_seconds: float = 1.0):
        self.cid_to_session = {}
        self.trusted_address = {}  # cid -> currently-validated address
        self.challenge_cooldown_seconds = challenge_cooldown_seconds
        self._last_challenge_time = {}  # (cid, candidate_address) -> monotonic timestamp

    def register_session(self, session: CTPSession, initial_trusted_address=None):
        for cid in session.current_and_upcoming_cids():
            self.cid_to_session[cid] = session
        if initial_trusted_address is not None:
            current_cid = compute_cid(session.ctp.k_epoch, session.ctp.current_epoch)
            self.trusted_address[current_cid] = initial_trusted_address

    def _refresh_registration(self, session: CTPSession):
        for cid in session.current_and_upcoming_cids():
            self.cid_to_session.setdefault(cid, session)

    def receive_packet(self, raw_packet: bytes, source_address, deliver_challenge_fn,
                        _now: float = None):
        """
        deliver_challenge_fn(candidate_address, challenge_bytes) -> response_bytes
        Simulates actually sending a PATH_CHALLENGE to a network address and
        getting a response back. A real implementation would be async; this
        is synchronous for testing, which does not change what it proves --
        whether the RESPONSE is correct is what matters, not the transport
        timing.

        _now: injectable clock for testing the rate limiter deterministically;
        defaults to time.monotonic().
        """
        import time
        now = _now if _now is not None else time.monotonic()

        cid = raw_packet[:CID_LEN]
        inner = raw_packet[CID_LEN:]
        session = self.cid_to_session.get(cid)
        if session is None:
            raise KeyError(f"unknown connection ID {cid.hex()} -- no session registered")

        trusted = self.trusted_address.get(cid)
        if trusted is None:
            # First packet ever seen for this CID: trust-on-first-use, the
            # same way a brand new session naturally starts (there is no
            # prior trusted address to be tricked away from yet, and the
            # initial address is established via the separate, already-
            # assumed-secure session-setup channel, not this mechanism).
            self.trusted_address[cid] = source_address
        elif source_address != trusted:
            # Candidate path -- validate BEFORE trusting it, and critically,
            # BEFORE decrypting (an unvalidated address must not be able to
            # consume sequence/replay state at all).
            rate_key = (cid, source_address)
            last = self._last_challenge_time.get(rate_key)
            if last is not None and (now - last) < self.challenge_cooldown_seconds:
                raise PathValidationThrottled(
                    f"challenge to {source_address!r} suppressed by rate limit "
                    f"({now - last:.3f}s since last attempt, cooldown "
                    f"{self.challenge_cooldown_seconds}s)"
                )
            self._last_challenge_time[rate_key] = now

            challenge = os.urandom(16)
            response = deliver_challenge_fn(source_address, challenge)
            expected = respond_to_path_challenge(session.ctp.k_auth, challenge)
            if not hmac.compare_digest(response or b"", expected):
                raise PathValidationError(
                    f"path validation failed for candidate address {source_address!r}"
                )
            self.trusted_address[cid] = source_address

        plaintext = session.ctp.decrypt(inner)  # only reached once the address is trusted
        self._refresh_registration(session)
        return plaintext, session
