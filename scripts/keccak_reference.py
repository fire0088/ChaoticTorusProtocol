"""
Stage 1: a from-scratch Keccak-f[1600] permutation + sponge construction,
validated bit-for-bit against Python's own hashlib (which wraps a mature,
independent Keccak implementation) before anything downstream is allowed
to depend on it.

This is deliberately built and checked in stages rather than written once
and trusted:
  1. Full 24-round Keccak-f[1600] + SHA3/SHAKE padding, checked against
     hashlib.sha3_256 / hashlib.shake_256 for many random inputs and known
     test vectors. This validates the permutation itself (rotation offsets,
     round constants, lane indexing) is exactly correct, since the sponge
     wrapper around it is simple and hashlib is trusted ground truth.
  2. (keccak_p12_test.py) Reduce to 12 rounds using the standard "last N
     constants" convention and validate against pycryptodome's
     KangarooTwelve, which uses Keccak-p[1600,12] internally.
Only after both stages pass does porting to a Taichi kernel happen
(keccak_p12_gpu.py), at which point it's checked against *this* module,
not assumed correct by construction.
"""

MASK64 = (1 << 64) - 1

RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

# Rotation offsets, indexed R[x][y]
R_OFFSETS = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]


def rol64(x: int, n: int) -> int:
    n %= 64
    if n == 0:
        return x & MASK64
    return ((x << n) | (x >> (64 - n))) & MASK64


def keccak_f(state, num_rounds: int = 24):
    """state: 5x5 nested list, state[x][y], 64-bit lanes. Modified in place
    and also returned. Uses the last `num_rounds` round constants (the
    standard convention for reduced-round Keccak-p variants)."""
    for rnd in range(24 - num_rounds, 24):
        # theta
        C = [state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4] for x in range(5)]
        D = [C[(x - 1) % 5] ^ rol64(C[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                state[x][y] ^= D[x]

        # rho + pi combined: new[y][2x+3y mod 5] = rot(old[x][y], R[x][y])
        newstate = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                newstate[y][(2 * x + 3 * y) % 5] = rol64(state[x][y], R_OFFSETS[x][y])
        state = newstate

        # chi
        newstate = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                newstate[x][y] = state[x][y] ^ ((~state[(x + 1) % 5][y]) & state[(x + 2) % 5][y] & MASK64)
        state = newstate

        # iota
        state[0][0] ^= RC[rnd]

    return state


def bytes_to_state(b: bytes):
    """200 bytes -> 5x5 state of 64-bit lanes, little-endian, lane[x][y]
    at byte offset 8*(x + 5*y) per the Keccak spec's lane ordering."""
    state = [[0] * 5 for _ in range(5)]
    for y in range(5):
        for x in range(5):
            offset = 8 * (x + 5 * y)
            lane = int.from_bytes(b[offset:offset + 8], "little")
            state[x][y] = lane
    return state


def state_to_bytes(state) -> bytes:
    out = bytearray(200)
    for y in range(5):
        for x in range(5):
            offset = 8 * (x + 5 * y)
            out[offset:offset + 8] = state[x][y].to_bytes(8, "little")
    return bytes(out)


def pad10star1(rate_bytes: int, msg_len: int, domain: int) -> bytes:
    pad_len = rate_bytes - (msg_len % rate_bytes)
    if pad_len == 1:
        return bytes([domain | 0x80])
    return bytes([domain]) + b"\x00" * (pad_len - 2) + bytes([0x80])


def keccak_sponge(message: bytes, rate_bytes: int, domain: int, out_len: int, num_rounds: int = 24) -> bytes:
    padded = message + pad10star1(rate_bytes, len(message), domain)
    state_bytes = bytearray(200)

    # absorb
    for i in range(0, len(padded), rate_bytes):
        block = padded[i:i + rate_bytes]
        for j in range(len(block)):
            state_bytes[j] ^= block[j]
        state = bytes_to_state(bytes(state_bytes))
        state = keccak_f(state, num_rounds)
        state_bytes = bytearray(state_to_bytes(state))

    # squeeze
    out = bytearray()
    state = bytes_to_state(bytes(state_bytes))
    while len(out) < out_len:
        sb = state_to_bytes(state)
        out.extend(sb[:rate_bytes])
        if len(out) < out_len:
            state = keccak_f(state, num_rounds)
    return bytes(out[:out_len])


def sha3_256(message: bytes) -> bytes:
    return keccak_sponge(message, rate_bytes=136, domain=0x06, out_len=32, num_rounds=24)


def shake_256(message: bytes, out_len: int) -> bytes:
    return keccak_sponge(message, rate_bytes=136, domain=0x1F, out_len=out_len, num_rounds=24)


def turboshake128(message: bytes, domain: int, out_len: int) -> bytes:
    """TurboSHAKE128: same sponge, rate=168 bytes (capacity=256 bits),
    12-round Keccak-p[1600,12], customizable domain byte. This is the
    simple (non-tree) sponge KangarooTwelve builds its tree-hash mode on
    top of -- validating this first isolates the reduced-round permutation
    itself from K12's additional Sakura tree-encoding complexity."""
    return keccak_sponge(message, rate_bytes=168, domain=domain, out_len=out_len, num_rounds=12)


if __name__ == "__main__":
    import hashlib
    import os

    print("=== Known-answer test: SHA3-256('') ===")
    expected = "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"[:64]
    got = sha3_256(b"").hex()
    print(f"expected: {expected}")
    print(f"got:      {got}")
    print("MATCH" if got == expected else "MISMATCH")

    print("\n=== SHA3-256 vs hashlib, random inputs ===")
    all_ok = True
    for length in [0, 1, 32, 135, 136, 137, 1000, 32800]:
        msg = os.urandom(length)
        ours = sha3_256(msg)
        theirs = hashlib.sha3_256(msg).digest()
        ok = ours == theirs
        all_ok &= ok
        print(f"len={length:>6}: {'OK' if ok else 'MISMATCH'}")
    print("ALL SHA3-256 TESTS PASSED" if all_ok else "FAILURES DETECTED")

    print("\n=== SHAKE-256 vs hashlib, random inputs and output lengths ===")
    all_ok2 = True
    for length in [0, 1, 32, 135, 136, 137, 1000, 32800]:
        for out_len in [16, 32, 64, 1000]:
            msg = os.urandom(length)
            ours = shake_256(msg, out_len)
            theirs = hashlib.shake_256(msg).digest(out_len)
            ok = ours == theirs
            all_ok2 &= ok
            if not ok:
                print(f"len={length}, out_len={out_len}: MISMATCH")
    print("ALL SHAKE-256 TESTS PASSED" if all_ok2 else "FAILURES DETECTED")

    print("\n=== Stage 2: Keccak-p[1600,12] via TurboSHAKE128 vs pycryptodome ===")
    from Crypto.Hash import TurboSHAKE128
    all_ok3 = True
    for length in [0, 1, 32, 167, 168, 169, 1000, 32800]:
        for out_len in [16, 32, 64, 1000]:
            for domain in [0x01, 0x1F, 0x7F]:
                msg = os.urandom(length)
                ours = turboshake128(msg, domain, out_len)
                theirs_h = TurboSHAKE128.new(domain=domain)
                theirs_h.update(msg)
                theirs = theirs_h.read(out_len)
                ok = ours == theirs
                all_ok3 &= ok
                if not ok:
                    print(f"len={length}, out_len={out_len}, domain={hex(domain)}: MISMATCH")
    print("ALL KECCAK-P[1600,12] (TurboSHAKE128) TESTS PASSED" if all_ok3 else "FAILURES DETECTED")
