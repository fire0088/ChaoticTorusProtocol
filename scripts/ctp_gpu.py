"""
GPU (Taichi compute shader) implementation of the Feistel-structured
lattice from ctp_cipher.py. Same construction, same offsets, same degree-3
chi, same round constants -- ported to run as parallel GPU kernels instead
of vectorized numpy, and cross-validated against the CPU reference for
bit-for-bit agreement before being trusted for anything.

Backend selection goes through gpu_backend.detect_best_backend(), which
isolates the crash risk of probing GPU backends in a subprocess (see that
module's docstring for why an in-process try/except is not sufficient here).
"""

import hashlib
import struct

import numpy as np
import taichi as ti

from gpu_backend import detect_best_backend

_BACKEND_NAME = detect_best_backend()
ti.init(arch=getattr(ti, _BACKEND_NAME))

LATTICE_N = 64
NUM_FEISTEL_ROUNDS = 8


@ti.kernel
def _theta_kernel(inp: ti.template(), outp: ti.template(), half: ti.i32, n: ti.i32):
    for i, j, k in inp:
        v = inp[i, j, k]
        v ^= inp[(i + 1) % half, j, k]
        v ^= inp[(i - 1) % half, j, k]
        v ^= inp[i, (j + 1) % n, k]
        v ^= inp[i, (j - 1) % n, k]
        v ^= inp[i, j, (k + 1) % n]
        v ^= inp[i, j, (k - 1) % n]
        outp[i, j, k] = v


@ti.kernel
def _chi_iota_combine_kernel(
    theta_out: ti.template(),
    L: ti.template(),
    new_R: ti.template(),
    round_const: ti.template(),
    half: ti.i32,
    n: ti.i32,
):
    """Applies chi (degree-3 mixed ANF, matching ctp_cipher._chi exactly)
    to theta_out, XORs in the round constant, then XORs with L to produce
    the new R half -- i.e. new_R = L ^ (chi(theta_out) ^ round_const)."""
    for i, j, k in theta_out:
        a = theta_out[(i - 1) % half, j, k]
        b = theta_out[(i - 2) % half, j, k]
        c = theta_out[i, (j - 1) % n, k]
        d = theta_out[i, (j - 2) % n, k]
        e = theta_out[i, j, (k - 1) % n]
        x = theta_out[i, j, k]
        chi_val = x ^ (a & b) ^ (c & d & e)
        new_R[i, j, k] = L[i, j, k] ^ (chi_val ^ round_const[i, j, k])


def _round_constant_bits(half, n, generation, round_idx):
    """Identical to ctp_cipher._round_constant -- public, deterministic,
    not secret."""
    nbits = half * n * n
    nbytes = (nbits + 7) // 8
    tag = f"CTP-round-const-{generation}-{round_idx}".encode()
    raw = hashlib.shake_256(tag).digest(nbytes)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
    return bits.reshape(half, n, n).astype(np.uint8)


class GPUFeistelLattice:
    """Same interface as ctp_cipher.FeistelLattice (step/packed_bytes/
    population), backed by Taichi GPU fields instead of numpy arrays."""

    def __init__(self, seed: bytes, n: int = LATTICE_N):
        assert n % 2 == 0, "requires an even lattice size"
        self.n = n
        self.half = n // 2
        self.generation = 0

        nbits = n * n * n
        nbytes = (nbits + 7) // 8
        raw = hashlib.shake_256(seed).digest(nbytes)
        bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
        full = bits.reshape(n, n, n).astype(np.uint8)

        self.L = ti.field(dtype=ti.u8, shape=(self.half, n, n))
        self.R = ti.field(dtype=ti.u8, shape=(self.half, n, n))
        self.L.from_numpy(full[: self.half])
        self.R.from_numpy(full[self.half :])

        # Persistent scratch buffers, allocated once and reused every round
        # via reference rotation (see step()) rather than reallocated.
        self._theta_scratch = ti.field(dtype=ti.u8, shape=(self.half, n, n))
        self._extra = ti.field(dtype=ti.u8, shape=(self.half, n, n))
        self._round_const_field = ti.field(dtype=ti.u8, shape=(self.half, n, n))

    def step(self, perturbation_seed: bytes = None):
        half, n = self.half, self.n
        for r in range(NUM_FEISTEL_ROUNDS):
            _theta_kernel(self.R, self._theta_scratch, half, n)
            rc_bits = _round_constant_bits(half, n, self.generation, r)
            self._round_const_field.from_numpy(rc_bits)
            # writes new_R = L ^ (chi(theta(R)) ^ round_const) into self._extra
            _chi_iota_combine_kernel(
                self._theta_scratch, self.L, self._extra,
                self._round_const_field, half, n,
            )
            # 3-way rotation: new L = old R, new R = computed value (_extra),
            # and old L becomes the free scratch buffer for the next round.
            self.L, self.R, self._extra = self.R, self._extra, self.L
        self.generation += 1

        if perturbation_seed is not None:
            nbits = n ** 3
            nbytes = (nbits + 7) // 8
            raw = hashlib.shake_256(perturbation_seed).digest(nbytes)
            mask_bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
            mask = mask_bits.reshape(n, n, n).astype(np.uint8)
            full = np.concatenate([self.L.to_numpy(), self.R.to_numpy()], axis=0)
            full ^= mask
            self.L.from_numpy(full[: self.half])
            self.R.from_numpy(full[self.half :])

    def packed_bytes(self) -> bytes:
        full = np.concatenate([self.L.to_numpy(), self.R.to_numpy()], axis=0)
        return np.packbits(full.reshape(-1)).tobytes()

    def population(self) -> int:
        return int(self.L.to_numpy().sum() + self.R.to_numpy().sum())

    def grid_numpy(self) -> np.ndarray:
        return np.concatenate([self.L.to_numpy(), self.R.to_numpy()], axis=0)


if __name__ == "__main__":
    import os
    print(f"Taichi backend in use: {_BACKEND_NAME}")

    print("\n=== Cross-validation against CPU reference ===")
    from ctp_cipher import feistel_step

    seed = os.urandom(32)
    n = 64

    nbits = n ** 3
    nbytes = (nbits + 7) // 8
    raw = hashlib.shake_256(seed).digest(nbytes)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:nbits]
    cpu_grid = bits.reshape(n, n, n).astype(np.uint8)

    gpu_lattice = GPUFeistelLattice(seed, n=n)

    all_match = True
    for gen in range(5):
        cpu_grid = feistel_step(cpu_grid, gen)
        gpu_lattice.step()
        gpu_grid = gpu_lattice.grid_numpy()
        match = np.array_equal(cpu_grid, gpu_grid)
        all_match &= match
        print(f"generation {gen+1}: CPU/GPU match = {match}  "
              f"(population cpu={cpu_grid.sum()} gpu={gpu_grid.sum()})")

    print("\nALL GENERATIONS MATCH" if all_match else "MISMATCH DETECTED")
