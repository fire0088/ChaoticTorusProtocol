"""
Taichi implementation of Keccak-p[1600,12], structured for the parallelism
that actually exists in this problem: many INDEPENDENT permutation
instances computed simultaneously (one per K12 chunk), not parallelism
within a single instance's 12 sequential rounds (which cannot be
parallelized -- each round depends on the previous one).

This is checked against keccak_reference.py (itself already validated
against hashlib and pycryptodome) before being trusted for anything, the
same discipline used for the lattice's GPU port (ctp_gpu.py).

NO GPU IS AVAILABLE IN THIS ENVIRONMENT. Backend selection goes through
gpu_backend.py, which will resolve to the CPU backend here. This validates
CORRECTNESS ONLY -- it says nothing about GPU throughput, exactly as
Section "What This Does and Does Not Demonstrate" of the paper already
states for the lattice port, and for the same reason.
"""

import numpy as np
import taichi as ti

from gpu_backend import detect_best_backend
import keccak_reference as ref

_BACKEND_NAME = detect_best_backend()
ti.init(arch=getattr(ti, _BACKEND_NAME))

MASK64 = np.uint64((1 << 64) - 1)

RC_NP = np.array(ref.RC, dtype=np.uint64)
R_OFFSETS_NP = np.array(ref.R_OFFSETS, dtype=np.int32)  # R_OFFSETS[x][y]

rc_field = ti.field(dtype=ti.u64, shape=24)
rc_field.from_numpy(RC_NP)

r_offsets_field = ti.field(dtype=ti.i32, shape=(5, 5))
r_offsets_field.from_numpy(R_OFFSETS_NP)


@ti.func
def rol64(x: ti.u64, n: ti.i32) -> ti.u64:
    result = x
    if n != 0:
        result = (x << ti.u64(n)) | (x >> ti.u64(64 - n))
    return result


@ti.kernel
def keccak_p12_batch(states: ti.template(), num_rounds: ti.i32):
    """states: ti.field(dtype=ti.u64, shape=(N, 5, 5)). Each of the N
    states is permuted independently and in parallel (Taichi auto-
    parallelizes the outer 'for b in range(N)' loop); the num_rounds
    rounds within each instance run sequentially, as they must."""
    N = states.shape[0]
    for b in range(N):
        for rnd in range(24 - num_rounds, 24):
            # theta
            C0 = states[b, 0, 0] ^ states[b, 0, 1] ^ states[b, 0, 2] ^ states[b, 0, 3] ^ states[b, 0, 4]
            C1 = states[b, 1, 0] ^ states[b, 1, 1] ^ states[b, 1, 2] ^ states[b, 1, 3] ^ states[b, 1, 4]
            C2 = states[b, 2, 0] ^ states[b, 2, 1] ^ states[b, 2, 2] ^ states[b, 2, 3] ^ states[b, 2, 4]
            C3 = states[b, 3, 0] ^ states[b, 3, 1] ^ states[b, 3, 2] ^ states[b, 3, 3] ^ states[b, 3, 4]
            C4 = states[b, 4, 0] ^ states[b, 4, 1] ^ states[b, 4, 2] ^ states[b, 4, 3] ^ states[b, 4, 4]

            D0 = C4 ^ rol64(C1, 1)
            D1 = C0 ^ rol64(C2, 1)
            D2 = C1 ^ rol64(C3, 1)
            D3 = C2 ^ rol64(C4, 1)
            D4 = C3 ^ rol64(C0, 1)

            for y in ti.static(range(5)):
                states[b, 0, y] ^= D0
                states[b, 1, y] ^= D1
                states[b, 2, y] ^= D2
                states[b, 3, y] ^= D3
                states[b, 4, y] ^= D4

            # rho + pi: new[y][(2x+3y) mod 5] = rot(old[x][y], R[x][y])
            tmp = ti.Matrix.zero(ti.u64, 5, 5)
            for x in ti.static(range(5)):
                for y in ti.static(range(5)):
                    nx = y
                    ny = (2 * x + 3 * y) % 5
                    tmp[nx, ny] = rol64(states[b, x, y], r_offsets_field[x, y])

            # chi
            newstate = ti.Matrix.zero(ti.u64, 5, 5)
            for x in ti.static(range(5)):
                for y in ti.static(range(5)):
                    newstate[x, y] = tmp[x, y] ^ ((~tmp[(x + 1) % 5, y]) & tmp[(x + 2) % 5, y])

            for x in ti.static(range(5)):
                for y in ti.static(range(5)):
                    states[b, x, y] = newstate[x, y]

            # iota
            states[b, 0, 0] ^= rc_field[rnd]


def numpy_states_to_taichi_layout(states_list):
    """states_list: list of 5x5 numpy uint64 arrays, state[x][y]."""
    arr = np.zeros((len(states_list), 5, 5), dtype=np.uint64)
    for i, s in enumerate(states_list):
        for x in range(5):
            for y in range(5):
                arr[i, x, y] = np.uint64(s[x][y])
    return arr


if __name__ == "__main__":
    import os

    _gpu_note = "" if _BACKEND_NAME != "cpu" else " (no GPU backend found -- correctness only, no throughput conclusion)"
    print(f"Taichi backend in use: {_BACKEND_NAME}{_gpu_note}")

    print("\n=== Cross-validation: Taichi Keccak-p[1600,12] vs CPU reference ===")
    N = 64
    random_states = []
    for _ in range(N):
        s = [[int(np.random.randint(0, 2**63, dtype=np.int64)) ^ (int(np.random.randint(0, 2, dtype=np.int64)) << 62)
              for _ in range(5)] for _ in range(5)]
        random_states.append(s)

    # Reference: run each through the validated CPU implementation
    expected = []
    for s in random_states:
        s_copy = [row[:] for row in s]
        out = ref.keccak_f(s_copy, num_rounds=12)
        expected.append(out)

    # Taichi: batch all N states and run in parallel
    ti_states = numpy_states_to_taichi_layout(random_states)
    field = ti.field(dtype=ti.u64, shape=(N, 5, 5))
    field.from_numpy(ti_states)
    keccak_p12_batch(field, 12)
    got = field.to_numpy()

    all_match = True
    for i in range(N):
        for x in range(5):
            for y in range(5):
                if int(got[i, x, y]) != (expected[i][x][y] & 0xFFFFFFFFFFFFFFFF):
                    all_match = False

    print(f"Batch of {N} independent Keccak-p[1600,12] permutations: "
          f"{'ALL MATCH' if all_match else 'MISMATCH DETECTED'}")

    print("\n=== Repeat with 10 independent trials of fresh random batches ===")
    all_trials_ok = True
    for trial in range(10):
        random_states = []
        for _ in range(N):
            s = [[int(np.random.randint(-(2**63), 2**63, dtype=np.int64)) for _ in range(5)] for _ in range(5)]
            s = [[v & 0xFFFFFFFFFFFFFFFF for v in row] for row in s]
            random_states.append(s)
        expected = [ref.keccak_f([row[:] for row in s], num_rounds=12) for s in random_states]
        ti_states = numpy_states_to_taichi_layout(random_states)
        field2 = ti.field(dtype=ti.u64, shape=(N, 5, 5))
        field2.from_numpy(ti_states)
        keccak_p12_batch(field2, 12)
        got2 = field2.to_numpy()
        trial_ok = all(
            int(got2[i, x, y]) == (expected[i][x][y] & 0xFFFFFFFFFFFFFFFF)
            for i in range(N) for x in range(5) for y in range(5)
        )
        all_trials_ok &= trial_ok
        print(f"  trial {trial}: {'OK' if trial_ok else 'MISMATCH'}")

    print("\nALL TRIALS PASSED" if all_trials_ok else "FAILURES DETECTED -- DO NOT TRUST THIS IMPLEMENTATION")
