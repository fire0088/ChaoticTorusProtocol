"""
Shared code for the CTP UDP demo (ctp_udp_server.py / ctp_udp_client.py):
packet framing for a minimal session handshake, and a live, rotating 3D
visualization of the lattice ("the torus block") showing only active (1)
cells.

IMPORTANT DEMO SIMPLIFICATION, stated up front: CTP's real design gets its
master key from a PQ-KEM handshake (see the papers, Section on Architecture
Overview) -- that handshake is not implemented here. This demo instead
derives a shared master key from a passphrase both sides provide on the
command line. This is fine for a local demo and is NOT how CTP is meant to
establish keys in the design the papers describe.

Also note: a single ctp_cipher.CTP object is DIRECTIONAL -- encrypt() and
decrypt() both mutate the same internal ratchet/lattice state, so one
object is for either "the stream I send" or "the stream I receive", never
both. A bidirectional session therefore needs two CTP objects per side: one
mirrors the other side's send stream (for decrypting) and one is this
side's own send stream (for encrypting). This file's handshake sets up
exactly that.
"""

import hashlib
import struct
import threading
import time
from collections import deque

import numpy as np

from ctp_cipher import CTP, AuthenticationError, ReplayError

# --- Wire framing for this demo's tiny session-setup protocol ---
# (Separate from CTP's own Data Plane packet format, which starts right
# after these framing bytes for PKT_DATA.)
PKT_INIT = 0x00  # client -> server: "here are the two directions' nonces"
PKT_ACK = 0x01   # server -> client: "got it, ready"
PKT_DATA = 0x02  # either direction: real CTP-encrypted payload follows


def derive_master_key(passphrase: str) -> bytes:
    """DEMO ONLY -- see module docstring. Real CTP gets this from a KEM
    handshake, not a passphrase."""
    return hashlib.sha3_256(passphrase.encode("utf-8")).digest()


def build_init_packet(nonce_c2s: bytes, nonce_s2c: bytes) -> bytes:
    assert len(nonce_c2s) == CTP.NONCE_LEN and len(nonce_s2c) == CTP.NONCE_LEN
    return bytes([PKT_INIT]) + nonce_c2s + nonce_s2c


def parse_init_packet(data: bytes):
    assert data[0] == PKT_INIT
    n = CTP.NONCE_LEN
    nonce_c2s = data[1:1 + n]
    nonce_s2c = data[1 + n:1 + 2 * n]
    return nonce_c2s, nonce_s2c


def build_ack_packet() -> bytes:
    return bytes([PKT_ACK])


def build_data_packet(ctp_wire_bytes: bytes) -> bytes:
    return bytes([PKT_DATA]) + ctp_wire_bytes


def lattice_snapshot(ctp_obj: CTP) -> np.ndarray:
    """The full LATTICE_N x LATTICE_N x LATTICE_N boolean grid -- 'the
    torus block' -- read directly from the live CTP object. This is the
    ACTUAL cryptographic state, not a separate simulation of it."""
    return ctp_obj.lattice.grid


class SharedDiagnostics:
    """Thread-safe holder for the info the visualizer displays, updated by
    the network-receive thread and the console-input/send thread, read by
    the main (matplotlib) thread."""

    def __init__(self, role: str):
        self.lock = threading.Lock()
        self.role = role
        self.sent = 0
        self.received = 0
        self.rejected_auth = 0
        self.rejected_replay = 0
        self.last_decrypted = ""
        self.last_decrypted_time = None
        self.last_sent_preview = ""
        self.connected = False
        self.log = deque(maxlen=8)

    def note(self, line: str):
        with self.lock:
            self.log.append(f"{time.strftime('%H:%M:%S')}  {line}")

    def snapshot_text(self, send_ctp: CTP, recv_ctp: CTP) -> str:
        with self.lock:
            lines = [
                f"role: {self.role}   connected: {self.connected}",
                f"send seq={send_ctp.seq:<6} epoch={send_ctp.current_epoch:<4} "
                f"lattice pop={send_ctp.lattice.population()}/{send_ctp.lattice_n**3} "
                f"({100*send_ctp.lattice.population()/send_ctp.lattice_n**3:.1f}%)",
                f"recv seq={recv_ctp.seq:<6} epoch={recv_ctp.current_epoch:<4} "
                f"lattice pop={recv_ctp.lattice.population()}/{recv_ctp.lattice_n**3} "
                f"({100*recv_ctp.lattice.population()/recv_ctp.lattice_n**3:.1f}%)",
                f"sent={self.sent}  received={self.received}  "
                f"rejected(auth)={self.rejected_auth}  rejected(replay)={self.rejected_replay}",
                f"last decrypted [{self.last_decrypted_time or '--:--:--'}]: "
                f"{self.last_decrypted[:60]!r}",
                "",
            ] + list(self.log)
            return "\n".join(lines)


class TorusVisualizer:
    """Live, slowly-rotating 3D point-cloud view of a lattice snapshot.
    Only active (1) cells are drawn -- 0 cells are simply absent, not drawn
    as anything. The full 64^3 grid (~131k active cells at ~50% density)
    is far too many points for smooth real-time matplotlib rendering, so
    the DISPLAYED point cloud is subsampled by `viz_stride` along each
    axis; the actual cryptographic lattice used for encryption is always
    full resolution regardless of this setting -- only the picture is
    thinned out."""

    def __init__(self, title: str, viz_stride: int = 4, rotate_deg_per_frame: float = 0.6):
        import matplotlib
        import matplotlib.pyplot as plt

        self.plt = plt
        self.viz_stride = viz_stride
        self.rotate_deg_per_frame = rotate_deg_per_frame
        self._azim = 0.0

        self.fig = plt.figure(figsize=(9, 7))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.suptitle(title)
        self.scatter = None
        self.text_artist = self.fig.text(
            0.02, 0.02, "", family="monospace", fontsize=8, va="bottom", ha="left"
        )

    def _subsampled_points(self, grid: np.ndarray):
        s = self.viz_stride
        sub = grid[::s, ::s, ::s]
        xs, ys, zs = np.nonzero(sub)
        return xs, ys, zs

    def render(self, grid: np.ndarray, diagnostics_text: str):
        xs, ys, zs = self._subsampled_points(grid)

        self.ax.clear()
        n = grid.shape[0] // self.viz_stride
        self.ax.set_xlim(0, n)
        self.ax.set_ylim(0, n)
        self.ax.set_zlim(0, n)
        self.ax.set_axis_off()

        if len(xs) > 0:
            # Color by a simple function of position for visual depth cues,
            # not for any cryptographic meaning.
            colors = (xs + ys + zs) % n
            self.ax.scatter(xs, ys, zs, c=colors, cmap="plasma", s=4, alpha=0.6, linewidths=0)

        self._azim = (self._azim + self.rotate_deg_per_frame) % 360
        self.ax.view_init(elev=22, azim=self._azim)

        self.text_artist.set_text(diagnostics_text)
        self.fig.canvas.draw_idle()

    def render_to_file(self, grid: np.ndarray, diagnostics_text: str, path: str):
        """Non-interactive single-frame render, for headless verification."""
        self.render(grid, diagnostics_text)
        self.fig.savefig(path, dpi=110)
