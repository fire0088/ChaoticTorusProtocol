"""
CTP UDP client. Sends the handshake INIT packet (with retry until the
server ACKs), then exchanges real CTP-encrypted messages with the server
over a real UDP socket. See ctp_udp_server.py and ctp_udp_common.py for
the protocol and the important demo-simplification notes (passphrase-
derived shared key instead of CTP's real KEM handshake; two directional
CTP objects per side).

Shows a live, slowly-rotating 3D view of this side's SEND lattice plus
diagnostic text. Type a message and press Enter to send it.

Usage:
    python3 ctp_udp_client.py --host 127.0.0.1 --port 9999 --secret "shared demo passphrase"
"""

import argparse
import os
import queue
import socket
import sys
import threading
import time

from ctp_cipher import CTP, AuthenticationError, ReplayError
from ctp_udp_common import (
    PKT_ACK, PKT_DATA,
    derive_master_key, build_init_packet, build_data_packet,
    lattice_snapshot, SharedDiagnostics, TorusVisualizer,
)


def main():
    ap = argparse.ArgumentParser(description="CTP UDP demo client")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--secret", required=True,
                     help="Shared passphrase -- MUST match the server's --secret.")
    ap.add_argument("--block-size", type=int, default=4,
                     help="Cluster cells into block_size^3 blocks for display "
                          "(the real lattice used for crypto is always full resolution)")
    ap.add_argument("--activity-threshold", type=float, default=0.6,
                     help="Only draw a block if its active-cell fraction exceeds this "
                          "(0-1). See ctp_udp_server.py for why the default isn't 0.")
    ap.add_argument("--evolve-every", type=int, default=1)
    ap.add_argument("--burn-in", type=int, default=8)
    ap.add_argument("--handshake-retries", type=int, default=10)
    ap.add_argument("--handshake-timeout", type=float, default=1.0)
    args = ap.parse_args()

    master_key = derive_master_key(args.secret)
    server_addr = (args.host, args.port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(args.handshake_timeout)

    nonce_c2s = os.urandom(CTP.NONCE_LEN)  # this client's send stream
    nonce_s2c = os.urandom(CTP.NONCE_LEN)  # mirrors the server's send stream

    init_pkt = build_init_packet(nonce_c2s, nonce_s2c)
    acked = False
    for attempt in range(1, args.handshake_retries + 1):
        sock.sendto(init_pkt, server_addr)
        print(f"[client] sent handshake attempt {attempt}/{args.handshake_retries} to {server_addr}")
        try:
            data, addr = sock.recvfrom(65535)
            if data and data[0] == PKT_ACK:
                acked = True
                break
        except socket.timeout:
            continue
    if not acked:
        print("[client] no ACK received after all retries -- is the server running? "
              "Proceeding anyway in case the ACK was simply lost.")

    sock.settimeout(None)
    print(f"[client] handshake complete with {server_addr}")

    send_ctp = CTP(master_key, nonce=nonce_c2s, burn_in=args.burn_in,
                   evolve_every=args.evolve_every)
    recv_ctp = CTP(master_key, nonce=nonce_s2c, burn_in=args.burn_in,
                   evolve_every=args.evolve_every)

    diag = SharedDiagnostics(role="client")
    diag.connected = True
    diag.note(f"handshake complete with {server_addr}")

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
                        diag.last_decrypted_time = time.strftime("%H:%M:%S")
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
        print("Type a message and press Enter to send it to the server. Ctrl-D to stop typing.")
        for line in sys.stdin:
            line = line.rstrip("\n")
            if not line:
                continue
            pkt = send_ctp.encrypt(line.encode("utf-8"))
            sock.sendto(build_data_packet(pkt), server_addr)
            with diag.lock:
                diag.sent += 1
                diag.last_sent_preview = line
            diag.note(f"sent: {line[:50]!r}")

    threading.Thread(target=network_thread, daemon=True).start()
    threading.Thread(target=console_thread, daemon=True).start()

    viz = TorusVisualizer("CTP client -- local send-side torus", block_size=args.block_size,
                          activity_threshold=args.activity_threshold)
    from matplotlib.animation import FuncAnimation

    def update(_frame):
        grid = lattice_snapshot(send_ctp)
        text = diag.snapshot_text(send_ctp, recv_ctp)
        viz.render(grid, text)
        return []

    # 150ms, not 600ms: the visualizer now only rebuilds voxel geometry
    # (~400ms) when the lattice actually changed since the last frame --
    # most frames are pure rotation on unchanged data (~110-120ms,
    # measured directly), so the interval can target that common case
    # instead of the occasional expensive one. A frame right after a sent
    # or received message will still take longer; that's fine; it's rare.
    anim = FuncAnimation(viz.fig, update, interval=150, cache_frame_data=False)
    viz.plt.show()


if __name__ == "__main__":
    main()
