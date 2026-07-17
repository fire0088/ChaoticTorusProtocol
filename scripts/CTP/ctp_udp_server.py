"""
CTP UDP server. Waits for a client's INIT packet, establishes two
directional CTP streams (one for encrypting outgoing messages, one for
decrypting the client's incoming messages -- see ctp_udp_common.py's
docstring for why these must be separate objects), then exchanges real
CTP-encrypted messages with the client over a real UDP socket.

Shows a live, slowly-rotating 3D view of this side's SEND lattice (the one
that evolves as this process encrypts outgoing messages) plus diagnostic
text. Type a message and press Enter to send it -- the torus updates
immediately.

DEMO SIMPLIFICATION: the shared master key comes from a passphrase both
sides provide, not from CTP's real PQ-KEM handshake. See ctp_udp_common.py.

Usage:
    python3 ctp_udp_server.py --port 9999 --secret "shared demo passphrase"
"""

import argparse
import queue
import socket
import sys
import threading

from ctp_cipher import CTP, AuthenticationError, ReplayError
from ctp_udp_common import (
    PKT_INIT, PKT_ACK, PKT_DATA,
    derive_master_key, parse_init_packet, build_ack_packet, build_data_packet,
    lattice_snapshot, SharedDiagnostics, TorusVisualizer,
)


def main():
    ap = argparse.ArgumentParser(description="CTP UDP demo server")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--secret", required=True,
                     help="Shared passphrase -- MUST match the client's --secret. "
                          "Demo-only stand-in for CTP's real KEM handshake.")
    ap.add_argument("--viz-stride", type=int, default=4,
                     help="Subsampling stride for the displayed point cloud "
                          "(the real lattice used for crypto is always full resolution)")
    ap.add_argument("--evolve-every", type=int, default=1,
                     help="How often the lattice evolves, in packets. 1 = every "
                          "packet, for a visually responsive demo (CTP's default "
                          "for efficiency is 16; see ctp_cipher.py's docstring)")
    ap.add_argument("--burn-in", type=int, default=8,
                     help="Lattice burn-in generations at startup/epoch reset "
                          "(reduced from CTP's default for fast demo startup)")
    args = ap.parse_args()

    master_key = derive_master_key(args.secret)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.bind, args.port))
    print(f"[server] listening on {args.bind}:{args.port} -- waiting for a client...")

    # Block here until the client's INIT arrives -- this is the only
    # single-threaded, blocking part; everything after is threaded.
    while True:
        data, client_addr = sock.recvfrom(65535)
        if len(data) >= 1 + 2 * CTP.NONCE_LEN and data[0] == PKT_INIT:
            nonce_c2s, nonce_s2c = parse_init_packet(data)
            break
        print(f"[server] ignoring unexpected packet from {client_addr} before handshake")

    sock.sendto(build_ack_packet(), client_addr)
    print(f"[server] handshake complete with {client_addr}")

    # recv_ctp mirrors the CLIENT's send stream (client encrypts with
    # nonce_c2s, so we decrypt with the same nonce); send_ctp is OUR send
    # stream, using the other nonce, which the client will mirror to decrypt.
    recv_ctp = CTP(master_key, nonce=nonce_c2s, burn_in=args.burn_in,
                   evolve_every=args.evolve_every)
    send_ctp = CTP(master_key, nonce=nonce_s2c, burn_in=args.burn_in,
                   evolve_every=args.evolve_every)

    diag = SharedDiagnostics(role="server")
    diag.connected = True
    diag.note(f"handshake complete with {client_addr}")

    outgoing = queue.Queue()

    def network_thread():
        while True:
            try:
                data, addr = sock.recvfrom(65535)
            except OSError:
                return
            if not data:
                continue
            if data[0] == PKT_DATA:
                try:
                    plaintext = recv_ctp.decrypt(data[1:])
                    with diag.lock:
                        diag.received += 1
                        diag.last_decrypted = plaintext.decode("utf-8", errors="replace")
                        import time as _time
                        diag.last_decrypted_time = _time.strftime("%H:%M:%S")
                    diag.note(f"recv: {plaintext[:50]!r}")
                except AuthenticationError:
                    with diag.lock:
                        diag.rejected_auth += 1
                    diag.note("REJECTED: authentication failed (tampered or wrong key)")
                except ReplayError:
                    with diag.lock:
                        diag.rejected_replay += 1
                    diag.note("REJECTED: replay/out-of-window packet")

    def console_thread():
        print("Type a message and press Enter to send it to the client. Ctrl-D to stop typing.")
        for line in sys.stdin:
            line = line.rstrip("\n")
            if not line:
                continue
            pkt = send_ctp.encrypt(line.encode("utf-8"))
            sock.sendto(build_data_packet(pkt), client_addr)
            with diag.lock:
                diag.sent += 1
                diag.last_sent_preview = line
            diag.note(f"sent: {line[:50]!r}")

    threading.Thread(target=network_thread, daemon=True).start()
    threading.Thread(target=console_thread, daemon=True).start()

    viz = TorusVisualizer("CTP server -- local send-side torus", viz_stride=args.viz_stride)
    from matplotlib.animation import FuncAnimation

    def update(_frame):
        grid = lattice_snapshot(send_ctp)
        text = diag.snapshot_text(send_ctp, recv_ctp)
        viz.render(grid, text)
        return []

    anim = FuncAnimation(viz.fig, update, interval=150, cache_frame_data=False)
    viz.plt.show()


if __name__ == "__main__":
    main()
