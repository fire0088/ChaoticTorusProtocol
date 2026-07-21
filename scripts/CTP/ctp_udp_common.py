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
    """Live, slowly-rotating 3D view of a lattice snapshot, clustered into
    block_size^3 blocks rather than drawn cell-by-cell. Each block is one
    marker, sized and colored by how many of its cells are active; a block
    with zero active cells is not drawn at all, same rule as before, just
    applied per-block instead of per-cell. This is both a performance
    choice (a 64^3 grid at block_size=4 is 16^3=4096 candidate markers
    instead of up to ~131k individual points) and the requested visual
    one -- fewer, bigger, chunkier shapes instead of a fine point cloud.
    The actual cryptographic lattice used for encryption is always full
    resolution regardless of this setting; only the picture is
    coarsened."""

    def __init__(self, title: str, block_size: int = 4, rotate_deg_per_frame: float = 0.6,
                 activity_threshold: float = 0.6, pulse_period: float = 4.0):
        import time as _time
        import matplotlib
        import matplotlib.pyplot as plt

        self.plt = plt
        self.block_size = block_size
        self.rotate_deg_per_frame = rotate_deg_per_frame
        # A block is only drawn if its activity exceeds this FRACTION of
        # block_size^3, not merely if it has any active cell at all. This
        # matters more than it looks: at block_size=4 (64 cells/block) and
        # the lattice's normal ~50% cell density, a block being entirely
        # empty has probability ~5x10^-20 -- essentially every block
        # registers as "occupied" under an any-nonzero rule, so ALL 4096
        # blocks would render solid, permanently, regardless of what the
        # lattice is actually doing. A threshold picks out the blocks that
        # are meaningfully ABOVE the ~50% baseline, which is what actually
        # varies as the lattice evolves and gives a real, changing pattern
        # instead of a static solid cube.
        self.activity_threshold = activity_threshold
        self.pulse_period = pulse_period
        self._start_time = _time.time()
        self._azim = 0.0
        self._elev = 22.0
        # Change detection: the lattice only actually changes when a packet
        # is sent or received, not on every animation timer tick. Rebuilding
        # the voxel geometry (ax.voxels()) costs ~500-600ms; re-aiming the
        # camera on the SAME already-built geometry costs ~150ms (measured
        # directly, roughly 4x cheaper). Most frames are pure rotation with
        # no new data, so skipping the rebuild on those frames is where the
        # actual smoothness gain comes from -- not from making voxels()
        # itself faster, which wasn't found to be meaningfully improvable
        # (edge rendering, e.g., was checked and isn't the bottleneck).
        self._last_grid = None
        self._n_blocks = None
        # Populated on rebuild, read on every frame (rebuild or not) to
        # apply pulse/depth-fade without redoing the expensive geometry
        # build -- per-voxel color updates on EXISTING artists cost about
        # the same as the rotation-only redraw already being done
        # (measured directly: ~176ms for ~190 voxels vs. ~198ms for a
        # rotation with no color changes at all), so animating color is
        # effectively free on top of what every frame already pays for.
        self._voxel_artists = None    # dict: (i,j,k) -> Poly3DCollection
        self._voxel_indices = None    # (M,3) array of filled block indices
        self._base_colors = None      # (M,4) RGBA before pulse/depth modulation

        self.fig = plt.figure(figsize=(9, 7), facecolor="black")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("black")
        # Terminal green throughout -- title, diagnostics text, and a
        # bordered "HUD box" behind the text instead of plain floating text.
        term_green = "#33FF66"
        self.fig.suptitle(title, color=term_green, family="monospace")
        self.scatter = None
        self.text_artist = self.fig.text(
            0.02, 0.02, "", family="monospace", fontsize=8, va="bottom", ha="left",
            color=term_green,
            bbox=dict(facecolor="black", edgecolor=term_green, alpha=0.55,
                      boxstyle="round,pad=0.4", linewidth=0.8),
        )

    def _block_counts(self, grid: np.ndarray):
        """Partition grid into block_size^3 cubes; return the full
        (n_blocks, n_blocks, n_blocks) array of each block's active-cell
        count (0..block_size^3), not just the occupied ones -- ax.voxels()
        needs the full grid shape to know which cells to draw at all."""
        b = self.block_size
        n = grid.shape[0]
        n_blocks = n // b
        trimmed = grid[:n_blocks * b, :n_blocks * b, :n_blocks * b]
        reshaped = trimmed.reshape(n_blocks, b, n_blocks, b, n_blocks, b)
        counts = reshaped.sum(axis=(1, 3, 5))  # (n_blocks, n_blocks, n_blocks)
        return counts, n_blocks

    def _glow_pulse(self) -> float:
        """Returns 0..1, a smooth single-lobe rise and fall across the
        full pulse_period -- a slow 'breathing' shape (raised cosine: 0
        at the start, smoothly up to 1 at the midpoint, smoothly back to
        0), not the sharp double-spike EKG shape used before. That
        earlier shape was two narrow gaussian bumps close together with
        a long quiet gap -- realistic as an actual heartbeat trace, but
        it stays "quick" no matter how long the overall period is made,
        since each spike itself is brief. A gentle, comforting pulse
        needs the RISE and FALL themselves to be slow, not just spaced
        further apart, which is what a single wide raised-cosine lobe
        gives instead. Uses wall-clock time, not frame count, so the
        rhythm stays consistent regardless of how often render() actually
        gets called."""
        import time as _time
        t = ((_time.time() - self._start_time) % self.pulse_period) / self.pulse_period
        return (1 - np.cos(2 * np.pi * t)) / 2

    def _depth_fade_factors(self, indices: np.ndarray, n_blocks: int) -> np.ndarray:
        """Per-voxel alpha multiplier from ~0.35 (far from camera) to 1.0
        (near camera), based on the CURRENT camera angle -- recomputed
        every frame since rotation changes which blocks face the camera."""
        elev_rad = np.radians(self._elev)
        azim_rad = np.radians(self._azim)
        view_dir = np.array([
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ])
        center = n_blocks / 2.0
        centered_positions = indices + 0.5 - center
        depths = centered_positions @ view_dir
        d_min, d_max = depths.min(), depths.max()
        if d_max - d_min < 1e-9:
            return np.ones(len(indices))
        normalized = (depths - d_min) / (d_max - d_min)
        return 0.35 + 0.65 * normalized

    def _apply_dynamic_color(self):
        """Combine each voxel's base activity color with the current
        heartbeat pulse (global, same phase for every block) and depth
        fade (per-block, camera-dependent), then push the result onto the
        EXISTING Poly3DCollection artists -- no geometry rebuild.

        The glow is MULTIPLICATIVE per RGB channel, not a flat amount
        added to all three -- checked directly against the actual colors
        this produces (hot colormap, count 1..64): adding a flat constant
        pushes every color's zero-valued channels up by the identical
        absolute amount, which is exactly what made it look like one
        uniform wash regardless of a block's own color (a dark red block
        and a bright orange one both picked up the same flat pink/white
        tint). Scaling each channel by a factor instead leaves a channel
        that's already 0 at 0, and only brightens channels with real
        intensity -- a dim red block glows into a more vivid red/orange,
        not pink, matching its own hue rather than a shared overlay
        color. A small saturation nudge (via HSV, S only, not V) is
        layered on top for the near-white/high-activity blocks
        specifically -- V turned out to be a poor lever on its own,
        since HSV's Value is just max(R,G,B) and the hot colormap
        already reaches R=1 very early in its range, so boosting V
        alone is a no-op for most blocks; only saturation has real
        headroom left there."""
        if self._voxel_artists is None:
            return
        pulse = self._glow_pulse()
        depth_factors = self._depth_fade_factors(self._voxel_indices, self._n_blocks)

        base_rgb = self._base_colors[:, :3]
        base_alpha = self._base_colors[:, 3]

        glow_factor = 1.0 + 0.4 * pulse
        boosted_rgb = np.clip(base_rgb * glow_factor, 0.0, 1.0)

        import matplotlib.colors as mcolors
        hsv = mcolors.rgb_to_hsv(boosted_rgb)
        hsv[:, 1] = np.clip(hsv[:, 1] + 0.15 * pulse, 0.0, 1.0)
        boosted_rgb = mcolors.hsv_to_rgb(hsv)

        alpha_mult = (0.55 + 0.45 * pulse) * depth_factors
        colors = np.empty_like(self._base_colors)
        colors[:, :3] = boosted_rgb
        colors[:, 3] = np.clip(base_alpha * alpha_mult, 0.05, 1.0)

        for (i, j, k), color in zip(self._voxel_indices, colors):
            artist = self._voxel_artists.get((i, j, k))
            if artist is not None:
                artist.set_facecolor(color)

    def render(self, grid: np.ndarray, diagnostics_text: str):
        data_changed = (self._last_grid is None) or not np.array_equal(grid, self._last_grid)

        if data_changed:
            self._rebuild_geometry(grid)
            self._last_grid = grid.copy()  # a copy, not a reference -- grid may be
            # the SAME live array object the CTP object keeps mutating in place;
            # comparing against a bare reference next time would compare the
            # array to itself post-mutation and never detect a real change.

        # Rotation, pulse/depth-fade color update, and diagnostics text all
        # happen on every call, cheaply, whether or not geometry was rebuilt.
        self._azim = (self._azim + self.rotate_deg_per_frame) % 360
        self.ax.view_init(elev=self._elev, azim=self._azim)
        self._apply_dynamic_color()
        self.text_artist.set_text(diagnostics_text)
        self.fig.canvas.draw_idle()

    def _rebuild_geometry(self, grid: np.ndarray):
        """The expensive path (~500-600ms, measured): only run when the
        lattice has actually changed since the last render."""
        counts, n_blocks = self._block_counts(grid)
        self._n_blocks = n_blocks
        max_count = self.block_size ** 3
        filled = counts > (self.activity_threshold * max_count)

        self.ax.clear()
        self.ax.set_facecolor("black")
        self.ax.set_xlim(0, n_blocks)
        self.ax.set_ylim(0, n_blocks)
        self.ax.set_zlim(0, n_blocks)
        self.ax.set_axis_off()
        self.ax.grid(False)
        # Defensive: matplotlib's 3D axes panes can render a visible
        # light-colored "box" behind the data even with axis_off() set,
        # depending on version -- make them fully transparent explicitly
        # rather than rely on axis_off() alone, so an empty region truly
        # renders as nothing, not a ghost outline that could be mistaken
        # for content.
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.pane.set_facecolor((0, 0, 0, 0))
            axis.pane.set_edgecolor((0, 0, 0, 0))

        self._voxel_artists = None
        self._voxel_indices = None
        self._base_colors = None

        if filled.any():
            # scatter's marker="s" only ever draws a flat, camera-facing
            # square in 3D -- it can't give real cube faces or edges no
            # matter the marker choice, since mplot3d markers are always
            # 2D glyphs placed at a projected point, not 3D geometry.
            # ax.voxels() draws actual cuboid geometry per cell (six faces,
            # correct perspective/occlusion, real edges), which is what
            # "cubes, like in Minecraft" actually requires.
            # Color range starts at the threshold itself, not 0 -- every
            # drawn block is by definition above threshold, so anchoring
            # the color scale there (rather than at 0, or even at
            # max_count/2) uses the full color gradient across the blocks
            # actually being shown instead of compressing them into a
            # narrow slice of it.
            threshold_count = self.activity_threshold * max_count
            norm = self.plt.Normalize(vmin=threshold_count, vmax=max_count)
            cmap = self.plt.get_cmap("hot")
            facecolors = cmap(norm(counts))
            facecolors[..., 3] = 0.95  # uniform alpha; color itself still varies by activity

            artists = self.ax.voxels(filled, facecolors=facecolors,
                                     edgecolor=(0.25, 0.25, 0.25, 0.6), linewidth=0.4)
            self._voxel_artists = artists
            indices = np.argwhere(filled)
            self._voxel_indices = indices
            self._base_colors = facecolors[tuple(indices.T)]

    def render_to_file(self, grid: np.ndarray, diagnostics_text: str, path: str):
        """Non-interactive single-frame render, for headless verification."""
        self.render(grid, diagnostics_text)
        self.fig.savefig(path, dpi=110, facecolor=self.fig.get_facecolor())
