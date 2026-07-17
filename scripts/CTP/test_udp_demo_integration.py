"""
Integration test for the CTP UDP demo's actual protocol logic: real
loopback UDP sockets, the real handshake, and real bidirectional CTP
encryption/decryption -- exactly what ctp_udp_server.py and
ctp_udp_client.py do, minus the matplotlib GUI (which needs a real display
to verify visually; the rendering pipeline itself was already checked
headlessly via TorusVisualizer.render_to_file).
"""

import os
import socket
import threading
import time

from ctp_cipher import CTP, AuthenticationError, ReplayError
from ctp_udp_common import (
    PKT_INIT, PKT_ACK, PKT_DATA,
    derive_master_key, build_init_packet, parse_init_packet,
    build_ack_packet, build_data_packet,
)

PORT = 58123


def run_server(master_key, received_messages, ready_event, stop_event):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", PORT))
    sock.settimeout(0.5)

    client_addr = None
    recv_ctp = send_ctp = None

    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        if recv_ctp is None and data[0] == PKT_INIT:
            nonce_c2s, nonce_s2c = parse_init_packet(data)
            client_addr = addr
            recv_ctp = CTP(master_key, nonce=nonce_c2s, burn_in=2, evolve_every=1)
            send_ctp = CTP(master_key, nonce=nonce_s2c, burn_in=2, evolve_every=1)
            sock.sendto(build_ack_packet(), client_addr)
            ready_event.set()
        elif data[0] == PKT_DATA and recv_ctp is not None:
            try:
                pt = recv_ctp.decrypt(data[1:])
                received_messages.append(("server_received", pt))
                # echo back, encrypted with the server's own send stream
                reply = send_ctp.encrypt(b"ack: " + pt)
                sock.sendto(build_data_packet(reply), client_addr)
            except (AuthenticationError, ReplayError) as e:
                received_messages.append(("server_rejected", type(e).__name__))
    sock.close()


def run_client(master_key, received_messages, messages_to_send, done_event):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0)

    nonce_c2s = os.urandom(CTP.NONCE_LEN)
    nonce_s2c = os.urandom(CTP.NONCE_LEN)
    server_addr = ("127.0.0.1", PORT)

    init_pkt = build_init_packet(nonce_c2s, nonce_s2c)
    acked = False
    for _ in range(10):
        sock.sendto(init_pkt, server_addr)
        try:
            data, addr = sock.recvfrom(65535)
            if data[0] == PKT_ACK:
                acked = True
                break
        except socket.timeout:
            continue
    assert acked, "client never received ACK from server"

    send_ctp = CTP(master_key, nonce=nonce_c2s, burn_in=2, evolve_every=1)
    recv_ctp = CTP(master_key, nonce=nonce_s2c, burn_in=2, evolve_every=1)

    for msg in messages_to_send:
        pkt = send_ctp.encrypt(msg)
        sock.sendto(build_data_packet(pkt), server_addr)
        try:
            data, addr = sock.recvfrom(65535)
            if data[0] == PKT_DATA:
                pt = recv_ctp.decrypt(data[1:])
                received_messages.append(("client_received", pt))
        except socket.timeout:
            received_messages.append(("client_timeout", msg))

    sock.close()
    done_event.set()


if __name__ == "__main__":
    master_key = derive_master_key("integration-test-passphrase")
    server_received = []
    client_received = []
    ready = threading.Event()
    stop = threading.Event()
    done = threading.Event()

    server_thread = threading.Thread(
        target=run_server, args=(master_key, server_received, ready, stop), daemon=True
    )
    server_thread.start()

    messages = [b"hello from client", b"second message", b"a third one, just to be sure"]
    client_thread = threading.Thread(
        target=run_client, args=(master_key, client_received, messages, done), daemon=True
    )
    client_thread.start()

    done.wait(timeout=10)
    time.sleep(0.3)  # let any last server-side processing land
    stop.set()
    server_thread.join(timeout=2)

    print("=== Server received ===")
    for kind, val in server_received:
        print(f"  {kind}: {val}")
    print("=== Client received ===")
    for kind, val in client_received:
        print(f"  {kind}: {val}")

    server_ok = [v for k, v in server_received if k == "server_received"] == messages
    client_ok = all(
        v == b"ack: " + m for (k, v), m in zip(
            [(k, v) for k, v in client_received if k == "client_received"], messages
        )
    ) and len([k for k, v in client_received if k == "client_received"]) == len(messages)

    print()
    print(f"[{'OK' if server_ok else 'FAIL'}] server correctly decrypted all client messages")
    print(f"[{'OK' if client_ok else 'FAIL'}] client correctly decrypted all server echo replies")
    print(f"\nOVERALL: {'PASS' if (server_ok and client_ok) else 'FAIL'}")
