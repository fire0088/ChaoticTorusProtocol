"""
Batched multi-session Feistel lattice CA update. This is the reframing
from the last round of testing: within ONE session, CTP's pipeline is
mostly sequential (ratchet -> lattice step -> extraction -> XOR -> tag,
each depending on the last). The real parallelism opportunity is ACROSS
sessions -- a server handling many concurrent connections has thousands of
completely independent lattice evolutions, keystream extractions, and tag
computations happening at once. This module batches the lattice CA update
across N sessions (assumed to be advancing in lockstep, e.g. a server
processing one tick of incoming packets across all active connections),
reusing the same theta/chi/iota round function validated in ctp_cipher.py
and ctp_gpu.py, extended with a leading batch (session) dimension.

Validated against running N independent single-session FeistelLattice
steps (ctp_cipher.py) before being trusted, same discipline as everywhere
else in this project. NO GPU IS AVAILABLE HERE -- this runs on Taichi's
CPU backend via gpu_backend.py, exactly like the prior GPU work; it proves
correctness and demonstrates the batching structure, not GPU throughput.
"""

import hashlib
import numpy as np
import taichi as ti

from gpu_backend import detect_best_backend
from ctp_cipher import FeistelLattice, LATTICE_N, NUM_FEISTEL_ROUNDS

_BACKEND_NAME = detect_best_backend()
ti.init(arch=getattr(ti, _BACKEND_NAME))


@ti.kernel
def _theta_batched(inp: ti.template(), outp: ti.template(), half: ti.i32, n: ti.i32):
    for b, i, j, k in inp:
        v = inp[b, i, j, k]
        v ^= inp[b, (i + 1) % half, j, k]
        v ^= inp[b, (i - 1) % half, j, k]
        v ^= inp[b, i, (j + 1) % n, k]
        v ^= inp[b, i, (j - 1) % n, k]
        v ^= inp[b, i, j, (k + 1) % n]
        v ^= inp[b, i, j, (k - 1) % n]
        outp[b, i, j, k] = v


@ti.kernel
def _chi_iota_combine_batched(
    theta_out: ti.template(),
    L: ti.template(),
    new_R: ti.template(),
    round_const: ti.template(),  # NOT batched: round constants are public,
    half: ti.i32,                # generation/round-dependent only, shared
    n: ti.i32,                   # across all sessions at the same tick
):
    for b, i, j, k in theta_out:
        a = theta_out[b, (i - 1) % half, j, k]
        b2 = theta_out[b, (i - 2) % half, j, k]
        c = theta_out[b, i, (j - 1) % n, k]
        d = theta_out[b, i, (j - 2) % n, k]
        e = theta_out[b, i, j, (k - 1) % n]
        x = theta_out[b, i, j, k]
        chi_val = x ^ (a & b2) ^ (c & d & e)
        new_R[b, i, j, k] = L[b, i, j, k] ^ (chi_val ^ round_const[i, j, k])


def _round_constant_bits(half, n, generation, round_idx):
    """Identical to ctp_cipher._round_constant -- public, deterministic."""
    nbits = half * n * n
    nbytes = (nbits + 7) // 8
    tag = f"CTP-round-const-{generation}-{round_idx}".encode()
    raw = hashlib.shake_256(tag).digest(nbytes)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
    return bits.reshape(half, n, n).astype(np.uint8)


class BatchedFeistelLattice:
    """N independent session lattices, advanced together each tick.
    Same interface shape as FeistelLattice, but step() advances ALL N
    sessions in one batch of kernel calls instead of N separate ones."""

    def __init__(self, seeds: list, n: int = LATTICE_N):
        assert n % 2 == 0
        self.n = n
        self.half = n // 2
        self.N = len(seeds)
        self.generation = 0

        full_grids = []
        for seed in seeds:
            nbits = n * n * n
            nbytes = (nbits + 7) // 8
            raw = hashlib.shake_256(seed).digest(nbytes)
            bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
            full_grids.append(bits.reshape(n, n, n).astype(np.uint8))
        full = np.stack(full_grids, axis=0)  # (N, n, n, n)

        self.L = ti.field(dtype=ti.u8, shape=(self.N, self.half, n, n))
        self.R = ti.field(dtype=ti.u8, shape=(self.N, self.half, n, n))
        self.L.from_numpy(full[:, : self.half])
        self.R.from_numpy(full[:, self.half :])

        self._theta_scratch = ti.field(dtype=ti.u8, shape=(self.N, self.half, n, n))
        self._extra = ti.field(dtype=ti.u8, shape=(self.N, self.half, n, n))
        self._round_const_field = ti.field(dtype=ti.u8, shape=(self.half, n, n))

    def step(self):
        half, n = self.half, self.n
        for r in range(NUM_FEISTEL_ROUNDS):
            _theta_batched(self.R, self._theta_scratch, half, n)
            rc_bits = _round_constant_bits(half, n, self.generation, r)
            self._round_const_field.from_numpy(rc_bits)
            _chi_iota_combine_batched(
                self._theta_scratch, self.L, self._extra, self._round_const_field, half, n
            )
            self.L, self.R, self._extra = self.R, self._extra, self.L
        self.generation += 1

    def grid_numpy(self, session_idx: int) -> np.ndarray:
        return np.concatenate(
            [self.L.to_numpy()[session_idx], self.R.to_numpy()[session_idx]], axis=0
        )


if __name__ == "__main__":
    import os
    import time

    _gpu_note = "" if _BACKEND_NAME != "cpu" else " (no GPU backend found -- correctness only, no throughput conclusion)"
    print(f"Taichi backend in use: {_BACKEND_NAME}{_gpu_note}")

    N_SESSIONS = 16
    seeds = [os.urandom(32) for _ in range(N_SESSIONS)]

    print(f"\n=== Correctness: batched ({N_SESSIONS} sessions) vs per-session CPU reference ===")
    batched = BatchedFeistelLattice(seeds, n=LATTICE_N)
    references = [FeistelLattice(seed, n=LATTICE_N) for seed in seeds]

    all_match = True
    for gen in range(3):
        batched.step()
        for ref in references:
            ref.step()
        for i, ref in enumerate(references):
            match = np.array_equal(batched.grid_numpy(i), ref.grid)
            all_match &= match
            if not match:
                print(f"  generation {gen+1}, session {i}: MISMATCH")
    print("ALL SESSIONS MATCH ACROSS ALL GENERATIONS" if all_match else "FAILURES DETECTED")

    print(f"\n=== Timing: batched step vs N sequential single-session steps (structural overhead only) ===")
    for N_SESSIONS in [1, 8, 32, 128]:
        seeds = [os.urandom(32) for _ in range(N_SESSIONS)]

        batched = BatchedFeistelLattice(seeds, n=LATTICE_N)
        batched.step()  # warm up / JIT
        t0 = time.perf_counter()
        batched.step()
        t1 = time.perf_counter()
        batched_time = t1 - t0

        references = [FeistelLattice(seed, n=LATTICE_N) for seed in seeds]
        references[0].step()  # warm up
        t0 = time.perf_counter()
        for ref in references:
            ref.step()
        t1 = time.perf_counter()
        sequential_time = t1 - t0

        print(f"  N={N_SESSIONS:>4}: batched={batched_time*1000:8.2f}ms  "
              f"sequential={sequential_time*1000:8.2f}ms  "
              f"ratio={sequential_time/batched_time:5.2f}x")
