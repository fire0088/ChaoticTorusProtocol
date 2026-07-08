"""
Full KangarooTwelve (Sakura tree-hash mode) built on the validated
Keccak-p[1600,12] sponge from keccak_reference.py. Checked against
pycryptodome's actual KangarooTwelve (not just the underlying permutation)
across the critical chunk-boundary sizes (8192 bytes) before being trusted.

Construction (K12, C assumed empty -- CTP has no use for a customization
string, it's using K12 purely as a keystream extraction XOF):
  S = M || length_encode(0)                      (since C = b"")
  split S into chunks of 8192 bytes: S_0, S_1, ..., S_(n-1)
  if n == 1 (|S| <= 8192): output = F(S_0 || 0x07, L)
  else:
    CV_i = F(S_i || 0x0B, 32)  for i = 1 .. n-1   (independent -> parallelizable)
    final_input = S_0 || 0x03 || 0x00*7 || CV_1 || .. || CV_(n-1)
                  || length_encode(n-1) || 0xFF || 0xFF
    output = F(final_input || 0x06, L)
where F is the sponge over Keccak-p[1600,12], rate 168 bytes (capacity 256
bits, matching TurboSHAKE128), exactly as validated in keccak_reference.py.
"""

import keccak_reference as ref

CHUNK_SIZE = 8192


def length_encode(x: int) -> bytes:
    """NIST SP 800-185 'right_encode': big-endian bytes of x (TRUE minimal
    length -- x=0 needs ZERO bytes, not one, since the minimal big-endian
    representation of 0 is the empty string), followed by a final byte
    giving the number of bytes used. Verified against pycryptodome's
    KangarooTwelve(b"") below; the x=0 special case was wrong on the first
    attempt (assumed one zero byte) and caught by that check, not assumed
    correct."""
    if x == 0:
        enc = b""
    else:
        enc = x.to_bytes((x.bit_length() + 7) // 8, "big")
    return enc + bytes([len(enc)])


def _f_with_domain(data_without_domain: bytes, domain: int, out_len: int) -> bytes:
    """F(X || domain_byte, out_len) -- the domain byte IS the pad10*1
    domain-separation byte consumed by keccak_sponge's `domain` parameter,
    not literally appended before a separate padding step (keccak_sponge
    already folds the domain byte into its own padding, matching how
    keccak_reference.turboshake128 is used)."""
    return ref.keccak_sponge(data_without_domain, rate_bytes=168, domain=domain, out_len=out_len, num_rounds=12)


def kangarootwelve(message: bytes, out_len: int, customization: bytes = b"") -> bytes:
    S = message + customization + length_encode(len(customization))

    if len(S) <= CHUNK_SIZE:
        return _f_with_domain(S, domain=0x07, out_len=out_len)

    S0 = S[:CHUNK_SIZE]
    chunks = [S[i:i + CHUNK_SIZE] for i in range(CHUNK_SIZE, len(S), CHUNK_SIZE)]
    n_minus_1 = len(chunks)

    # Chaining values: independent per chunk -> the actual parallelism
    # opportunity K12 is designed around (see keccak_p12_gpu.py for the
    # batched-independent-instance GPU structure this maps onto).
    cvs = [_f_with_domain(chunk, domain=0x0B, out_len=32) for chunk in chunks]

    final_input = S0 + b"\x03" + b"\x00" * 7 + b"".join(cvs) + length_encode(n_minus_1) + b"\xFF\xFF"
    return _f_with_domain(final_input, domain=0x06, out_len=out_len)


if __name__ == "__main__":
    import os
    from Crypto.Hash import KangarooTwelve as PyCryptoK12

    def reference_k12(message: bytes, out_len: int) -> bytes:
        h = PyCryptoK12.new(data=message)
        return h.read(out_len)

    print("=== KangarooTwelve vs pycryptodome, across the chunk boundary ===")
    test_sizes = [
        0, 1, 100,
        CHUNK_SIZE - 1, CHUNK_SIZE, CHUNK_SIZE + 1,   # the critical boundary
        2 * CHUNK_SIZE, 2 * CHUNK_SIZE + 1,
        3 * CHUNK_SIZE + 500,
        32800,   # CTP's actual real-world per-packet absorb size
        1_000_000,
    ]
    all_ok = True
    for size in test_sizes:
        msg = os.urandom(size)
        for out_len in [16, 32, 64, 1000]:
            ours = kangarootwelve(msg, out_len)
            theirs = reference_k12(msg, out_len)
            ok = ours == theirs
            all_ok &= ok
            status = "OK" if ok else "MISMATCH"
            if not ok or out_len == 32:  # print one line per size, flag any failure
                print(f"  size={size:>9}  out_len={out_len:>5}: {status}")

    print("\nALL KANGAROOTWELVE TESTS PASSED" if all_ok else "FAILURES DETECTED -- DO NOT TRUST THIS IMPLEMENTATION")
