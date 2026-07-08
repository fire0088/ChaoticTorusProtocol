"""
Reference implementation of CTP v0.10 (Chaotic Torus Protocol).

This is a RESEARCH PROTOTYPE for benchmarking and comparison purposes only.
It has not been cryptanalyzed. Do not use it to protect real data.

Design implemented here:
  - 64x64x64 boolean lattice, updated each generation by a Feistel-structured
    round function (v0.5; replaces the v0.1-v0.4 Carter Bays 4555 outer-
    totalistic rule, retained below as CarterBaysLattice for reference/
    comparison only and no longer used by default).
  - 1024-generation burn-in (configurable; default reduced for benchmark
    turnaround, see NOTE in CTP.__init__).
  - Keystream extraction: SHAKE-256 sponge absorbing ratchet state + packed
    lattice bits, squeezed to the needed number of keystream bytes.
  - Ratchet: S_i = SHA3-256(S_{i-1} || SHA3-256(M_{i-1}) || seq_i)
  - AEAD: encrypt-then-MAC via HMAC-SHA3-256 over (seq || ciphertext), keyed
    by a key independently derived (HKDF) from the diffusion key.

v0.5 change -- why the lattice rule was replaced, not just patched:
  Empirical testing (see TrippleTorusSymmetricEncryption_v4.tex) found the
  Carter Bays 4555 rule collapses to the all-zero absorbing state in ~90%
  of random trials within a handful of generations, after which every
  colliding session (regardless of key) produces the identical, public,
  key-independent lattice output. v0.4 patched this with a secret-derived
  perturbation XORed in on every step. That patch worked, but it worked by
  masking the CA's output with independent secret noise, not by fixing the
  CA -- it didn't address the root cause.

  v0.5 replaces the rule itself with a Feistel-structured construction
  (FeistelLattice, below) built from a linear diffusion step and a
  nonlinear step modeled on Keccak-f's theta/chi, wrapped in a Feistel
  network so the whole transform is PROVABLY invertible (verified by
  round-trip testing at full n=64 scale) regardless of whether the inner
  round function is itself a bijection -- which, per invertibility_test.py,
  it isn't at this lattice size (chi's specific offset structure is only
  bijective for odd cycle lengths, and 64 is even). Because the lattice
  update is now a true permutation of the state space, two different keys
  provably cannot collide onto the same lattice trajectory -- not just
  empirically, as a mathematical consequence of injectivity. Measured
  density holds at 50% +/- 0.5% over 100 generations and avalanche reaches
  the ideal 50% bit-difference within a single generation (see
  feistel_lattice.py for the standalone tests).

  The v0.4 secret perturbation is KEPT as defense-in-depth, layered on top
  of the new lattice. Its original specific justification (preventing
  convergence to a public, key-independent constant) is now handled by
  bijectivity instead. Its continued justification is generic: this rule
  still has no differential/linear cryptanalysis bounds and no third-party
  review, so an additional, independent secret-derived input costs little
  and protects against classes of weakness bijectivity alone would not
  catch.
"""

import hashlib
import hmac
import os
import struct
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve1d
from Crypto.Hash import KangarooTwelve

LATTICE_N = 64
_BOX_1D = np.ones(3, dtype=np.int32)
NUM_FEISTEL_ROUNDS = 8


def hkdf_sha3_256(key: bytes, label: bytes, length: int = 32) -> bytes:
    """Minimal single-block HKDF-Expand using SHA3-256 as the PRF, with an
    implicit HKDF-Extract step using a fixed salt (adequate for deriving
    two independent subkeys from a uniformly random master key)."""
    prk = hmac.new(b"CTP-HKDF-salt", key, hashlib.sha3_256).digest()
    t = hmac.new(prk, label + b"\x01", hashlib.sha3_256).digest()
    return t[:length]


def _theta(half: np.ndarray) -> np.ndarray:
    """Linear diffusion: XOR in all 6 axis-neighbors (toroidal wrap on
    each axis at that half's own length). Confirmed invertible as a
    standalone GF(2)-linear operator at n=4, n=8 (power-of-2 sizes); see
    invertibility_test.py. Not required to be invertible here since it's
    used inside a Feistel round, but it is anyway -- extra margin."""
    out = half.copy()
    for axis in range(3):
        out ^= np.roll(half, 1, axis=axis)
        out ^= np.roll(half, -1, axis=axis)
    return out


def _chi(half: np.ndarray) -> np.ndarray:
    """
    Nonlinear step, v0.6: algebraic degree raised from 2 to 3.

    v0.5 used cell XOR (NOT a AND b) -- a single degree-2 monomial, modeled
    directly on Keccak-f's chi. Low algebraic degree is precisely the
    property that made the original Carter Bays threshold rule vulnerable
    to algebraic/SAT attacks (Section on algebraic resistance, Revision 2);
    a degree-2 nonlinear step reintroduces a milder version of the same
    concern one layer down. Since chi is wrapped in a Feistel network
    (feistel_step below), it does NOT need to be bijective on its own --
    confirmed in invertibility_test.py -- which leaves complete freedom to
    raise its degree without re-deriving invertibility.

    Current form: cell XOR (a AND b) XOR (c AND d AND e) -- degree 3, using
    5 distinct neighbor offsets across two axes and two monomials of
    different degree, rather than a single pure-AND term (a single high-
    degree monomial is a narrower, more structurally regular target than a
    mixed ANF with terms of different degree and support).

    Tested empirically (chi_degree_test.py) against the original degree-2
    form and two higher-degree single-monomial alternatives: density and
    avalanche are statistically indistinguishable across all variants
    (theta's diffusion absorbs the bias each monomial introduces on its
    own), so raising the degree was verified to cost nothing on the
    properties already established, not merely assumed to be free.
    """
    a = np.roll(half, 1, axis=0)
    b = np.roll(half, 2, axis=0)
    c = np.roll(half, 1, axis=1)
    d = np.roll(half, 2, axis=1)
    e = np.roll(half, 1, axis=2)
    return half ^ (a & b) ^ (c & d & e)


def _round_constant(shape, generation: int, round_idx: int) -> np.ndarray:
    """Public (non-secret) round constant, a deterministic function of the
    generation and round index only. Purpose is to break the round
    function's translational symmetry across generations and Feistel
    rounds (the same purpose Keccak's iota step serves) -- this is NOT a
    source of security by itself and does not need to be secret."""
    nbits = int(np.prod(shape))
    nbytes = (nbits + 7) // 8
    tag = f"CTP-round-const-{generation}-{round_idx}".encode()
    raw = hashlib.shake_256(tag).digest(nbytes)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
    return bits.reshape(shape).astype(np.uint8)


def _feistel_F(half: np.ndarray, generation: int, round_idx: int) -> np.ndarray:
    rc = _round_constant(half.shape, generation, round_idx)
    return _chi(_theta(half)) ^ rc


def feistel_step(grid: np.ndarray, generation: int) -> np.ndarray:
    """
    Advance the lattice one generation via a Feistel network:
        L' = R
        R' = L XOR F(R, generation, round)
    repeated for NUM_FEISTEL_ROUNDS rounds. Provably invertible regardless
    of F's own bijectivity (feistel_step_inverse below is the exact
    inverse; round-trip-verified at full n=64 scale in feistel_lattice.py).
    """
    n = grid.shape[0]
    half = n // 2
    L, R = grid[:half].copy(), grid[half:].copy()
    for r in range(NUM_FEISTEL_ROUNDS):
        L, R = R, L ^ _feistel_F(R, generation, r)
    return np.concatenate([L, R], axis=0)


def feistel_step_inverse(grid: np.ndarray, generation: int) -> np.ndarray:
    """Exact inverse of feistel_step. Not used by CTP itself (the cipher
    only ever runs the lattice forward), but included and tested because a
    passing round-trip test is the concrete evidence that this construction
    is genuinely bijective and the implementation matches the math."""
    n = grid.shape[0]
    half = n // 2
    L, R = grid[:half].copy(), grid[half:].copy()
    for r in reversed(range(NUM_FEISTEL_ROUNDS)):
        L, R = R ^ _feistel_F(L, generation, r), L
    return np.concatenate([L, R], axis=0)


class FeistelLattice:
    """
    3D lattice updated by the Feistel-structured round function above.
    Default lattice class as of v0.5; see module docstring for why this
    replaced CarterBaysLattice rather than patching it further.
    """

    def __init__(self, seed: bytes, n: int = LATTICE_N):
        assert n % 2 == 0, "FeistelLattice requires an even lattice size"
        self.n = n
        self.generation = 0
        nbits = n * n * n
        nbytes = (nbits + 7) // 8
        raw = hashlib.shake_256(seed).digest(nbytes)
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
        self.grid = bits.reshape(n, n, n).astype(np.uint8)

    def step(self, perturbation_seed: bytes = None):
        """Advance one generation. perturbation_seed: optional secret-
        derived material XORed in after the round (defense-in-depth; see
        module docstring -- no longer required to prevent public collapse,
        since this construction is bijective by design, but retained as a
        cheap additional protection against unknown structural weaknesses
        in the round function itself)."""
        self.grid = feistel_step(self.grid, self.generation)
        self.generation += 1

        if perturbation_seed is not None:
            nbits = self.n ** 3
            nbytes = (nbits + 7) // 8
            raw = hashlib.shake_256(perturbation_seed).digest(nbytes)
            mask_bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
            mask = mask_bits.reshape(self.n, self.n, self.n).astype(np.uint8)
            self.grid = self.grid ^ mask

    def packed_bytes(self) -> bytes:
        return np.packbits(self.grid.reshape(-1)).tobytes()

    def population(self) -> int:
        return int(self.grid.sum())


class CarterBaysLattice:
    """3D outer-totalistic Life-like cellular automaton on a toroidal grid.

    RETAINED FOR REFERENCE/COMPARISON ONLY -- no longer used by default as
    of v0.5. Empirical testing found this rule collapses to the all-zero
    absorbing state in ~90% of random trials (see the v0.4 draft). See
    FeistelLattice above for the current default lattice class."""

    def __init__(self, seed: bytes, birth=(4,), survive=(5,), n=LATTICE_N):
        self.n = n
        self.birth = set(birth)
        self.survive = set(survive)
        # Expand the seed deterministically into n^3 bits via SHAKE-256.
        nbits = n * n * n
        nbytes = (nbits + 7) // 8
        raw = hashlib.shake_256(seed).digest(nbytes)
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
        self.grid = bits.reshape(n, n, n).astype(np.uint8)

    def step(self, perturbation_seed: bytes = None):
        """
        Advance one generation under the configured rule.

        perturbation_seed: if provided, fresh pseudorandom bits derived from
        this value (expected to be secret, session-specific ratchet material)
        are XORed into the grid after the rule is applied. This is a v0.4
        security addition -- see CTP._advance for the rationale: without it,
        an outer-totalistic rule that reaches an absorbing state (e.g. total
        extinction) produces the *same* public, key-independent constant for
        every session that collapses, regardless of key. Binding each step to
        secret material ensures the lattice can never fully decouple from the
        key, even in a degenerate/absorbing regime.
        """
        g = self.grid.astype(np.int32)
        g = convolve1d(g, _BOX_1D, axis=0, mode="wrap")
        g = convolve1d(g, _BOX_1D, axis=1, mode="wrap")
        g = convolve1d(g, _BOX_1D, axis=2, mode="wrap")
        neighbor_sum = g - self.grid  # exclude the center cell itself
        born = np.isin(neighbor_sum, list(self.birth)) & (self.grid == 0)
        survives = np.isin(neighbor_sum, list(self.survive)) & (self.grid == 1)
        new_grid = (born | survives).astype(np.uint8)

        if perturbation_seed is not None:
            nbits = self.n ** 3
            nbytes = (nbits + 7) // 8
            raw = hashlib.shake_256(perturbation_seed).digest(nbytes)
            mask_bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
            mask = mask_bits.reshape(self.n, self.n, self.n).astype(np.uint8)
            new_grid = new_grid ^ mask

        self.grid = new_grid

    def packed_bytes(self) -> bytes:
        return np.packbits(self.grid.reshape(-1)).tobytes()

    def population(self) -> int:
        return int(self.grid.sum())


class ReplayError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class CTP:
    NONCE_LEN = 16

    def __init__(self, master_key: bytes, nonce: bytes = None, burn_in: int = 64,
                 lattice_n: int = LATTICE_N, evolve_every: int = 16,
                 epoch_size: int = 64, epoch_burn_in: int = 1):
        """
        NOTE ON burn_in: the spec calls for 1024 generations. That is
        expensive to run per-session in a pure-Python/numpy benchmark
        (~seconds per session at n=64). Default here is reduced to 64 for
        benchmark turnaround; pass burn_in=1024 to match the spec exactly.
        This only affects lattice mixing, not the AEAD security properties,
        which per the spec do not depend on the lattice (Section 3.2 of the
        v0.2 draft).

        NOTE ON evolve_every: the original spec evolves the lattice once per
        packet. Profiling (profile_ctp.py) shows lattice.step() is the
        dominant per-packet cost, and it is a FIXED cost independent of
        payload size. Per-packet keystream freshness is already guaranteed
        by the SHA3-256 ratchet chain S_i, which incorporates seq_i and a
        hash of the previous plaintext regardless of how often the lattice
        itself steps -- the sponge's security argument (Section 3.2) never
        depended on the lattice changing every packet. Setting evolve_every
        > 1 amortizes the lattice-step cost across multiple packets without
        reintroducing keystream repetition, at the cost of the lattice
        contributing even less fresh entropy per packet than it already
        optionally does. evolve_every=1 reproduces the original per-packet
        schedule exactly.

        NOTE ON nonce: nonce is mixed into every HKDF derivation and the
        ratchet-init state. If not supplied, a fresh random one is
        generated (the sender's role) and must then be communicated to the
        receiver once, out-of-band or as part of session setup -- the
        receiver constructs CTP(master_key, nonce=received_nonce) with the
        SAME nonce to derive matching keys.

        NOTE ON epoch_size / epoch_burn_in (this revision -- fixes a real
        defect found via direct testing): the ratchet chain S_i =
        SHA3-256(S_{i-1} || H(M_{i-1}) || seq_i) is a hash chain with no
        gap tolerance. Testing confirmed that losing a single packet
        desynchronizes the chain PERMANENTLY -- every subsequent packet in
        the session decrypts to silent garbage, with no error, because the
        HMAC tag (keyed by the fixed K_auth) verifies correctly regardless
        of ratchet sync; only decryption depends on the chain, and
        authentication cannot detect its desync. A second, related issue:
        the lattice's evolution schedule was driven by a locally-counted
        number of successfully-processed packets, which also diverges
        after any loss, independent of the ratchet issue.

        Both are fixed the same way: every epoch_size packets (default 64,
        matching the replay window granularity), the ratchet state and the
        lattice are reset to values derived deterministically from
        (K_epoch, K_diff, epoch_number) -- NOT chained from any prior
        packet's plaintext or lattice history. Since the epoch number is
        seq // epoch_size and seq is visible in the cleartext packet
        header, either side can independently compute the correct epoch
        start state without needing to have correctly processed any packet
        from a prior epoch. This bounds the blast radius of a single loss
        event to at most epoch_size-1 packets (the remainder of the epoch
        in which the loss occurred) rather than the rest of the session.

        This does NOT fix packet reordering within an epoch -- the chain
        still assumes in-order delivery between epoch boundaries, and a
        reordered (but not lost) packet within one epoch will still
        desync decryption until the next epoch boundary. Fixing that would
        require abandoning within-epoch chaining entirely (deriving every
        packet's ratchet contribution as a pure function of its own
        position rather than of prior packets), which was not what was
        agreed on here and is not implemented.

        epoch_burn_in defaults to 1, not the full burn_in used at session
        start: measured avalanche testing (see the security battery and
        Feistel construction validation) shows this lattice reaches ideal
        ~50% avalanche within a single generation, unlike the original
        Carter Bays rule burn_in was originally sized for. Re-running a
        large burn_in on every epoch reset would add real, unnecessary
        per-epoch cost; 1 generation is sized to the lattice's actual
        measured mixing speed, not to the original (much larger) default.
        """
        assert len(master_key) == 32, "CTP requires a 256-bit master key"
        if nonce is None:
            nonce = os.urandom(self.NONCE_LEN)
        assert len(nonce) == self.NONCE_LEN, f"nonce must be {self.NONCE_LEN} bytes"
        self.nonce = nonce

        self.k_diff = hkdf_sha3_256(master_key, b"CTP-diffusion" + nonce)
        self.k_auth = hkdf_sha3_256(master_key, b"CTP-auth" + nonce)
        self.k_perturb = hkdf_sha3_256(master_key, b"CTP-perturb" + nonce)
        self.k_epoch = hkdf_sha3_256(master_key, b"CTP-epoch" + nonce)

        self.lattice_n = lattice_n
        self.evolve_every = max(1, evolve_every)
        self.epoch_size = max(1, epoch_size)
        self.epoch_burn_in = epoch_burn_in

        self.seq = 0
        self.replay_window_size = 64  # matches the window size the design has
        # described since its first revision; the implementation of it,
        # found during this security check, did not: it used an ever-
        # growing Python set() that never pruned old entries, meaning
        # memory grew without bound over a long-lived session (and never
        # actually implemented "sliding" or "bitwise masks" as claimed).
        # Fixed to an actual bounded two-integer sliding window below.
        self._replay_highest = -1
        self._replay_bitmask = 0
        self.current_epoch = None  # forces a reset before the first packet
        self._epoch_reset(epoch=0, burn_in=burn_in)

    def _epoch_reset(self, epoch: int, burn_in: int = None):
        """Deterministically (re)derive ratchet state and lattice for the
        given epoch, from session keys and the epoch number alone -- no
        dependency on any prior packet's plaintext or lattice history. See
        the epoch_size docstring note in __init__ for why this exists."""
        if burn_in is None:
            burn_in = self.epoch_burn_in
        self.state = hashlib.sha3_256(self.k_epoch + struct.pack(">Q", epoch)).digest()
        lattice_seed = hashlib.sha3_256(
            self.k_diff + b"epoch" + struct.pack(">Q", epoch)
        ).digest()
        self.lattice = FeistelLattice(lattice_seed, n=self.lattice_n)
        for i in range(burn_in):
            self.lattice.step(
                perturbation_seed=self._perturb_seed(b"burnin-epoch" + struct.pack(">Q", epoch), i)
            )
        self.current_epoch = epoch
        self._epoch_packet_count = 0

    def _ensure_epoch(self, seq: int):
        epoch = seq // self.epoch_size
        if epoch != self.current_epoch:
            self._epoch_reset(epoch)

    def _check_replay(self, seq: int) -> bool:
        """Read-only check against the CURRENTLY COMMITTED window state --
        does not mutate anything. Returns True if seq is definitely a
        replay (already committed) or too old to track. Mutation is
        deferred to _commit_replay, called only after tag verification
        succeeds: if checking and marking happened together before
        verification, an attacker could pre-emptively 'burn' a legitimate
        sequence number with a forged packet that fails verification,
        causing a real retransmission using that same seq to be wrongly
        rejected later. Deferring the mutation until after a packet is
        confirmed authentic closes that -- a flood of forged packets can
        never shift the window or mark anything as seen."""
        if seq > self._replay_highest:
            return False
        diff = self._replay_highest - seq
        if diff >= self.replay_window_size:
            return True
        bit = 1 << diff
        return bool(self._replay_bitmask & bit)

    def _commit_replay(self, seq: int):
        """Actually advance/mark the sliding window. Call ONLY after tag
        verification has succeeded (see _check_replay's rationale)."""
        if seq > self._replay_highest:
            shift = seq - self._replay_highest
            if shift >= self.replay_window_size:
                self._replay_bitmask = 0
            else:
                self._replay_bitmask <<= shift
                self._replay_bitmask &= (1 << self.replay_window_size) - 1
            self._replay_highest = seq
            self._replay_bitmask |= 1
        else:
            diff = self._replay_highest - seq
            self._replay_bitmask |= (1 << diff)

    def _perturb_seed(self, tag: bytes, counter: int) -> bytes:
        """Domain-separated, secret-derived material used to keep the lattice
        from ever settling into a state that is independent of the key. See
        CTP._advance and CarterBaysLattice.step() docstring for rationale."""
        return hashlib.sha3_256(self.k_perturb + tag + struct.pack(">Q", counter)).digest()

    # ---- internal helpers ----
    def _keystream(self, seq: int, nbytes: int) -> bytes:
        """
        Keystream extraction, v0.8: SHAKE-256 -> KangarooTwelve (K12).

        Measured (not assumed) at CTP's actual per-packet absorb size
        (~32KB: ratchet state + lattice bytes + seq): K12 and SHAKE-256
        perform identically, because pycryptodome's K12 is sequential
        software with no parallel chunk execution -- the "thousands of
        chunks in parallel" framing describes what K12's tree structure
        permits, not what any available library does by default, and our
        current per-call input (~4 chunks at K12's 8KB leaf size) is far
        too small for that structure to matter regardless. At larger
        input sizes (1-10MB, tested standalone), K12 measured ~30% faster
        than SHAKE-256 -- from its reduced round count (12-round Keccak-p
        vs. 24-round Keccak-f), not from parallelism.

        The tradeoff this brings, stated plainly rather than glossed over:
        12 rounds is a real, deliberate reduction in security margin
        versus SHAKE-256's 24, publicly analyzed and defended by the
        Keccak team as retaining a comfortable margin against best-known
        attacks, but a smaller margin nonetheless. K12 is otherwise a
        better-provenance primitive than most of this project's own
        constructions (IETF-standardized, independently reviewed), which
        is the actual reason to prefer it here -- not the parallelism
        claim, which measured evidence does not support at this input size.
        """
        material = self.state + self.lattice.packed_bytes() + struct.pack(">Q", seq)
        return KangarooTwelve.new(data=material).read(nbytes)

    def _keystream_shake256(self, seq: int, nbytes: int) -> bytes:
        """Retained for comparison/rollback: the pre-v0.8 SHAKE-256
        extraction, unchanged."""
        material = self.state + self.lattice.packed_bytes() + struct.pack(">Q", seq)
        return hashlib.shake_256(material).digest(nbytes)

    def _advance(self, seq: int, plaintext: bytes):
        h_m = hashlib.sha3_256(plaintext).digest()
        self.state = hashlib.sha3_256(
            self.state + h_m + struct.pack(">Q", seq)
        ).digest()
        # Epoch-local count (reset to 0 by _epoch_reset), not a running
        # total -- a running total driven by "packets successfully
        # processed" would itself diverge between sender and receiver
        # after any loss, independent of the ratchet chain issue this
        # epoch mechanism fixes. See __init__'s epoch_size docstring note.
        self._epoch_packet_count += 1
        if self._epoch_packet_count % self.evolve_every == 0:
            # Bind the lattice to secret, per-call material (self.state has
            # just been updated above, so this is fresh and unique per call).
            # This closes the empirical defect found in testing: this rule
            # collapses to the all-zero fixed point in ~90% of random trials,
            # and once there, EVERY session (regardless of key) that collapses
            # produces the identical, public, key-independent lattice output
            # for the remainder of the session. XORing in secret-derived bits
            # on every step means the lattice output can never be independent
            # of the key, even in a degenerate/absorbing regime. Note this
            # also means, once perturbed, the lattice's output is statistically
            # dominated by this secret mask rather than by the automaton's own
            # structure -- see the paper (Section on this fix) for the honest
            # implication of that.
            self.lattice.step(perturbation_seed=self._perturb_seed(b"step", seq))

    def _tag(self, seq: int, ciphertext: bytes) -> bytes:
        mac_input = struct.pack(">Q", seq) + ciphertext
        return hmac.new(self.k_auth, mac_input, hashlib.sha3_256).digest()

    # ---- public API ----
    def encrypt(self, plaintext: bytes) -> bytes:
        seq = self.seq
        self._ensure_epoch(seq)
        keystream = self._keystream(seq, len(plaintext))
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, keystream))
        tag = self._tag(seq, ciphertext)
        self._advance(seq, plaintext)
        self.seq += 1
        return struct.pack(">Q", seq) + tag + ciphertext

    def decrypt(self, packet: bytes) -> bytes:
        seq = struct.unpack(">Q", packet[:8])[0]
        tag = packet[8:40]
        ciphertext = packet[40:]

        # Replay check happens on metadata only, before trusting the tag result.
        if self._check_replay(seq):
            raise ReplayError(f"sequence {seq} rejected (replay or outside window)")

        expected_tag = self._tag(seq, ciphertext)
        if not hmac.compare_digest(tag, expected_tag):
            raise AuthenticationError("tag verification failed")

        # Only after authentication succeeds do we trust ciphertext/seq.
        self._commit_replay(seq)
        self._ensure_epoch(seq)
        keystream = self._keystream(seq, len(ciphertext))
        plaintext = bytes(a ^ b for a, b in zip(ciphertext, keystream))
        self._advance(seq, plaintext)
        return plaintext


if __name__ == "__main__":
    # Quick self-test: round trip + tamper detection.
    # NOTE: nonce must now be shared between sender and receiver explicitly
    # (in the real protocol, carried in the KEM handshake transcript or the
    # first packet) -- alice generates one, bob is constructed with the
    # same value, exactly as two real endpoints would coordinate.
    key = os.urandom(32)
    alice = CTP(key, burn_in=64)
    bob = CTP(key, nonce=alice.nonce, burn_in=64)

    msg = b"the quick brown fox jumps over the lazy dog" * 10
    pkt = alice.encrypt(msg)
    recovered = bob.decrypt(pkt)
    assert recovered == msg, "round-trip failed"
    print("[OK] round-trip correctness")

    # Tamper with ciphertext -> must fail authentication. Use a fresh
    # sender/receiver pair so this isn't conflated with replay detection.
    alice_t = CTP(key, burn_in=64)
    bob_t = CTP(key, nonce=alice_t.nonce, burn_in=64)
    pkt_t = alice_t.encrypt(msg)
    tampered = bytearray(pkt_t)
    tampered[-1] ^= 0x01
    try:
        bob_t.decrypt(bytes(tampered))
        print("[FAIL] tampered ciphertext was accepted!")
    except AuthenticationError:
        print("[OK] tamper detection (bit-flip rejected)")

    # Replay -> must fail.
    alice2 = CTP(key, burn_in=64)
    bob2 = CTP(key, nonce=alice2.nonce, burn_in=64)
    pkt2 = alice2.encrypt(msg)
    bob2.decrypt(pkt2)
    try:
        bob2.decrypt(pkt2)
        print("[FAIL] replayed packet was accepted!")
    except ReplayError:
        print("[OK] replay detection")

    # The fix this revision adds: reusing the SAME master_key across two
    # sessions no longer collides, as long as the (now mandatory, and by
    # default randomly generated) nonces differ. This directly tests the
    # vulnerability demonstrated in pipeline_followup_test.py.
    same_key = os.urandom(32)
    session_a = CTP(same_key, burn_in=64)  # random nonce A
    session_b = CTP(same_key, burn_in=64)  # random nonce B (independent)
    assert session_a.nonce != session_b.nonce, "nonces collided (astronomically unlikely)"
    zeros = b"\x00" * 256
    ks_a = session_a.encrypt(zeros)[40:]
    ks_b = session_b.encrypt(zeros)[40:]
    if ks_a == ks_b:
        print("[FAIL] same master_key + different nonces still collided!")
    else:
        diff_bits = sum(bin(a ^ b).count("1") for a, b in zip(ks_a, ks_b))
        pct = 100.0 * diff_bits / (len(ks_a) * 8)
        print(f"[OK] same master_key, different nonces: keystreams differ "
              f"({pct:.1f}% of bits, expected ~50% -- no collision)")
