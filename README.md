# The Chaotic Torus Protocol (CTP)

**A research prototype — not for production use, not for protecting real data.**

This repository contains the reference implementation, GPU acceleration work, security
testing, and formal proofs for CTP: a proof-of-concept symmetric cipher built to explore
two questions — can a cipher's diffusion layer be designed around GPU hardware from the
outset rather than ported to it afterward, and what does a genuinely rigorous,
verify-before-trust validation discipline look like when applied to a novel cryptographic
construction end to end?

Read `docs/ctp_tifs_paper.pdf` for the full, formal account (formatted for IEEE
Transactions on Information Forensics and Security). `docs/ctp_whitepaper.pdf` is a less
formal running account of the same work. This README is the practical, "how do I run
this" companion to both.

## What CTP actually is

CTP is a session-based symmetric transport cipher with two planes:

- **Data Plane**: payload encryption. A Feistel-structured 3D lattice diffusion layer
  (a 64×64×64 boolean grid evolved by a local, uniform, GPU-parallel round function)
  feeds a KangarooTwelve-based sponge that extracts the keystream. Encryption is
  encrypt-then-MAC (HMAC-SHA3-256), with the tag verified before any decryption is
  attempted.
- **Control Plane**: a standard AES-GCM channel used for session setup. An earlier design
  also gave it a role in recovering from packet loss; testing found that role turned out
  to be unnecessary — see the epoch mechanism below.

On top of this sits a **connection-identification layer** (added later, described below)
supporting multiplexing multiple sessions over one channel and surviving address changes
(e.g., a mobile device switching networks), analogous to QUIC's Connection ID.

### The diffusion layer

Two lattice halves `L, R` evolve via a Feistel round: `L' = R`, `R' = L ⊕ F(R)`, where `F`
combines a linear step (`theta`: XOR of six axis-neighbors) and a nonlinear step
(`chi(x) = x ⊕ (a∧b) ⊕ (c∧d∧e)`, algebraic degree 3, confirmed by brute-force computation
of its full truth table, not just asserted). The Feistel wrapper makes the whole
construction invertible regardless of whether `chi` itself is a bijection (it isn't, at
this lattice size) — this is the standard Luby-Rackoff guarantee, applied deliberately.

### Keystream extraction and the epoch mechanism

Keystream is `K12(ratchet_state ‖ lattice_bytes ‖ seq, length)`. The ratchet chains
through a one-way hash of every prior plaintext, which has zero tolerance for packet
loss by construction — losing one packet corrupts every subsequent one, silently, for
the rest of the session (confirmed by direct testing, not assumed). The fix: every
`epoch_size` packets (64 by default), both the ratchet and the lattice reset to a state
derived only from a dedicated epoch key and the epoch number — not chained from
anything — so either endpoint can independently recompute the correct state without
needing to have processed anything from the previous epoch. No resync message is ever
sent; there's nothing for the mechanism to get stuck waiting on.

### Connection identification and path validation

CTP's wire format originally carried nothing but a sequence number — no way to tell
which session's keys apply to an incoming packet except by assuming the network layer
already handles that. A connection ID, derived per-epoch from the same epoch key already
in use (`CID = SHA3-256(K_epoch ‖ "CTP-CID" ‖ epoch)`, truncated), fixes this without
needing new handshake messages, since both sides can already compute future epochs' CIDs
in advance.

A naive first version of this — trusting a new source address immediately on any packet
with a recognized CID — was tested directly against the attack it invites: an off-path
attacker with no key material at all can intercept one valid packet, relay it from a
spoofed address, and hijack the session's address association, causing the real sender's
next packet to be rejected as a replay. This worked exactly as predicted. The fix is a
QUIC-style challenge/response: an address change isn't trusted until it proves possession
of the session's MAC key, and packets from an unvalidated address aren't decrypted at all
(so they can't consume replay-window state either). Re-tested against the identical
attack, the fix holds; a separate test confirms genuine migration still works.

### What's proven, and what's tested but not proven

Two formal, reduction-based results exist:

1. **Theorem (nonce-based authenticated encryption security)**: CTP's Data Plane is
   secure, given that the keystream generator behaves as a PRF and HMAC-SHA3-256 is a
   secure MAC — via the standard encrypt-then-MAC composition theorem
   (Bellare–Namprempre 2000).
2. **Theorem (path validation soundness)**: an adversary without the session's MAC key
   succeeds at spoofing a validated address with probability bounded by HMAC's own PRF
   advantage plus 2⁻²⁵⁶.

The keystream generator's PRF assumption is the weak point — it's specific to this
project's own novel construction, unreviewed by anyone outside it. Two targeted attacks
probe it directly rather than leaving it as an unexamined assumption: a **cube tester**
(checking for low algebraic degree surviving to the output) and an **exact differential
distribution table analysis** of the only nonlinear component (checking for a
high-probability differential trail). Neither found anything at the scale tested. This is
real evidence, not proof — no third-party cryptanalysis has been done, and neither test
is a substitute for one.

### Performance, briefly

Both GPU-accelerated components (lattice diffusion, keystream extraction) now measure
faster than an optimized sequential CPU baseline — roughly 88× for the lattice under
concurrent load, and a margin growing past 2× for extraction at the largest batch size
tested. Getting the second number right took six rounds of profiling-driven fixes; an
early, incomplete version measured over 1000× *slower*, and a later "fixed" version's
flat performance ceiling turned out to be 99%+ unaccelerated Python/numpy overhead, not
GPU hardware saturation. Despite the internal speedup, CTP's absolute throughput remains
one to two orders of magnitude below a single CPU core of contemporary hardware-
accelerated AES-256-GCM or ChaCha20-Poly1305. Read `docs/ctp_tifs_paper.pdf` Section VI
for the full, honest account of that debugging process — it's as much a finding as the
final number.

## Requirements

```
pip install numpy scipy pycryptodome taichi --break-system-packages
```

GPU scripts run on CPU if no GPU backend is available (Taichi prints which backend it
selected) — correctness is identical either way, only throughput differs. All scripts in
`scripts/` are flat and import from each other directly; keep them together in one
folder.

## Quick start

```bash
cd scripts

# Basic correctness: round-trip, tamper detection, replay detection, nonce fix
python3 ctp_cipher.py

# Full statistical security battery (NIST-style tests, algebraic degree, avalanche)
python3 security_battery.py

# The formal proof's supporting evidence
python3 cube_tester.py             # algebraic-degree attack attempt
python3 differential_analysis.py   # differential cryptanalysis attempt

# Defects found and fixed, demonstrated directly
python3 pipeline_followup_test.py  # two-time-pad
python3 test_epoch_resync.py       # packet-loss recovery
python3 test_replay_window.py      # bounded replay window + pre-emptive-burn fix
python3 test_cid_routing.py        # off-path address-spoofing attack + fix

# Throughput vs. AES-256-GCM and ChaCha20-Poly1305
python3 benchmark_compare.py

# GPU kernels and their validation
python3 ctp_gpu.py                       # lattice diffusion layer
python3 keccak_p12_gpu.py                # Keccak-p permutation
python3 batched_feistel_lattice.py       # multi-session batched lattice
python3 batched_k12_sessions.py          # multi-session batched keystream (early version)
python3 keccak_fused_gpu.py              # fully-fused single-kernel K12 (Stage A/B/C validation)
python3 benchmark_fused_k12_vectorized.py  # the version that actually beats pycryptodome
python3 profile_fused_k12_phases.py      # phase-by-phase timing breakdown

# Exact hardware/software versions, for reproducing the paper's numbers
python3 capture_environment.py
```

## What's in `scripts/`

**Core**
- `ctp_cipher.py` — the reference implementation (lattice, ratchet, epoch mechanism,
  replay window, encrypt/decrypt). Start here.
- `ctp_session_manager.py` — the connection-ID and path-validation layer, built as a
  wrapper above `ctp_cipher.py` rather than a wire-format change, so the core's
  extensively-tested behavior is untouched by it.

**Validation** (independent-reference cross-checking, the discipline the whole project
is built around)
- `keccak_reference.py` — from-scratch Keccak-f[1600]/Keccak-p[1600,12], validated
  against `hashlib` and TurboSHAKE128.
- `k12_full.py` — full KangarooTwelve tree construction, validated against pycryptodome.
- `invertibility_test.py` — proves the diffusion layer's linear step is invertible and
  finds exactly where its nonlinear step is not, motivating the Feistel wrapper.
- `feistel_lattice.py` — the Feistel-wrapped round function and its own round-trip test.
- `security_battery.py` — NIST SP 800-22-style statistical battery, avalanche testing,
  exact algebraic-degree verification.
- `cube_tester.py` — algebraic-degree attack against the full composed keystream
  generator (not just the nonlinear step in isolation).
- `differential_analysis.py` — exact differential distribution table for the nonlinear
  step, plus empirical multi-round trail tracking against the real construction.
- `pipeline_followup_test.py` — the two-time-pad defect this project found (an unused
  nonce field) and its fix, demonstrated directly.
- `test_epoch_resync.py` — the packet-loss/permanent-desync defect and its fix.
- `test_replay_window.py` — the unbounded-memory replay-tracking defect, its fix, and the
  pre-emptive sequence-number-burn condition found while fixing it.
- `test_cid_routing.py` — the off-path address-spoofing attack against the connection-ID
  layer, its fix, the fix's own rate-limiting gap, and confirmation that legitimate
  migration still works.

**GPU acceleration**
- `gpu_backend.py` — crash-safe backend detection (naive init can segfault a system with
  no GPU rather than raising a catchable exception).
- `ctp_gpu.py` — Taichi kernel for the lattice diffusion layer.
- `keccak_p12_gpu.py` — Taichi kernel for the Keccak-p permutation.
- `batched_feistel_lattice.py` / `batched_k12_sessions.py` — early multi-session batched
  versions (the latter has the round-trip-per-block bottleneck the later fused version
  fixes — kept for the historical comparison).
- `keccak_fused_gpu.py` — the fully-fused, single-kernel version: one upload, one launch,
  one download for an entire batch's multi-block absorption. Includes its own Stage
  A/B/C correctness validation.
- `benchmark_fused_k12.py` / `benchmark_fused_k12_vectorized.py` /
  `benchmark_fused_k12_highN.py` — successive rounds of the GPU-vs-CPU benchmark; the
  `_vectorized` version is the one that actually beats pycryptodome.
- `profile_fused_k12_phases.py` — the phase-by-phase breakdown that found the GPU kernel
  was under 1% of runtime while unaccelerated Python orchestration was the rest.

**Exploration** (design-process scripts documenting negative results, not the final
construction — kept because the paper's rigor depends on showing what didn't work, too)
- `sweep_rules.py` — parameter sweep showing no simple threshold-counting cellular
  automaton rule avoids structural collapse.
- `chi_theta_test.py` — the first structurally different round function tried,
  motivating the Feistel construction.
- `chi_degree_test.py` — comparison of nonlinear-step variants by algebraic degree.

**Benchmarks / reproducibility**
- `benchmark_compare.py` — throughput vs. AES-256-GCM and ChaCha20-Poly1305.
- `profile_ctp.py` — per-component cost breakdown of the full pipeline.
- `capture_environment.py` — prints exact GPU model, driver, CUDA version, and package
  versions, for anyone trying to reproduce the papers' measured numbers.

## What's in `docs/`

- `ctp_tifs_paper.pdf` / `.tex` — the full, formal writeup (IEEE TIFS format): design,
  security evaluation, formal proofs, GPU results, related work, honest limitations.
- `ctp_whitepaper.pdf` / `.tex` — a less formal running account of the same project.

## Before you take any of this seriously as "secure"

- No third-party cryptanalysis has been performed. Everything here is the authors'
  own testing.
- The formal proofs are conditional on the keystream generator behaving as a PRF — an
  assumption specific to this project, probed by two targeted attacks but not proven,
  and fundamentally not provable the way no symmetric primitive's security ever is.
- CTP provides no forward secrecy against master-key compromise.
- GPU measurements are from one hardware configuration (NVIDIA GeForce RTX 3070,
  CUDA 13.1); generality across other GPUs is untested.
- CTP's absolute throughput, even after the GPU work, remains one to two orders of
  magnitude below a single CPU core of contemporary hardware-accelerated encryption.

This is a research vehicle for studying GPU-parallel cipher design and validation
methodology. It is not a deployment candidate.
