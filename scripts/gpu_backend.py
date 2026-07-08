"""
Backend detection for Taichi that survives a segfaulting backend probe.

Earlier testing (this project, prior revision) found that ti.init(arch=ti.gpu)
does not fail gracefully on a system without a working GPU backend: it
segfaults the process outright ("RHI Error: Can not create Vulkan instance"
followed by a hard crash), not a catchable Python exception. A try/except
around ti.init() in-process cannot recover from that -- the process is
already gone. This module isolates each backend probe in a disposable
subprocess so a crash there does not take down the caller, and only
initializes Taichi in the actual process once a backend has been confirmed
to work.
"""

import subprocess
import sys


_PROBE_CODE = """
import taichi as ti
import sys
try:
    ti.init(arch=ti.{arch}, log_level=ti.ERROR)
    resolved = ti.lang.impl.current_cfg().arch
    expected = ti.{arch}
    if "{arch}" != "cpu" and resolved != expected:
        # Taichi silently substituted a different backend (commonly CPU)
        # rather than raising -- this is NOT the backend we asked for.
        print("PROBE_FAIL_SILENT_FALLBACK")
        sys.exit(1)
    x = ti.field(dtype=ti.u8, shape=(4, 4, 4))
    x.fill(1)
    _ = x.to_numpy().sum()  # force actual device execution, not just init
    print("PROBE_OK")
except Exception:
    print("PROBE_FAIL")
    sys.exit(1)
"""


def _probe_backend(arch_name: str, timeout: float = 20.0) -> bool:
    """Run a throwaway subprocess that tries to init + actually use the
    given Taichi backend. Returns True only if it exits cleanly AND prints
    the success sentinel -- a segfault in the subprocess shows up as a
    non-zero/abnormal exit code here, which we can safely detect from the
    parent without the parent itself crashing."""
    code = _PROBE_CODE.format(arch=arch_name)
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, timeout=timeout, text=True,
        )
        return result.returncode == 0 and "PROBE_OK" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def detect_best_backend(preference=("cuda", "vulkan", "metal", "cpu")) -> str:
    """Return the name of the first backend in `preference` that actually
    works on this machine, probed safely out-of-process. Raises RuntimeError
    if none do (should not happen -- 'cpu' is always last and should always
    succeed)."""
    for name in preference:
        if _probe_backend(name):
            return name
    raise RuntimeError(
        "No usable Taichi backend found, including cpu fallback. "
        "Check the Taichi installation."
    )


if __name__ == "__main__":
    print("Probing Taichi backends (each isolated in a subprocess)...")
    for name in ("cuda", "vulkan", "metal", "cpu"):
        ok = _probe_backend(name)
        print(f"  {name:8s}: {'AVAILABLE' if ok else 'not available'}")
    best = detect_best_backend()
    print(f"\nSelected backend: {best}")
