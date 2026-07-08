"""
Capture exact hardware/software versions for reproducibility. Run this on
the same machine used for the benchmark numbers reported in the papers --
currently those numbers just say "an NVIDIA CUDA device," which isn't
enough for anyone to interpret or attempt to reproduce them.
"""

import platform
import subprocess
import sys


def get_gpu_info():
    info = {}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            info["gpu_name"] = parts[0] if len(parts) > 0 else "unknown"
            info["driver_version"] = parts[1] if len(parts) > 1 else "unknown"
            info["gpu_memory"] = parts[2] if len(parts) > 2 else "unknown"
            info["compute_capability"] = parts[3] if len(parts) > 3 else "unknown"
        else:
            info["gpu_name"] = "nvidia-smi returned no data"
    except FileNotFoundError:
        info["gpu_name"] = "nvidia-smi not found (no NVIDIA GPU, or not on PATH)"
    except Exception as e:
        info["gpu_name"] = f"error querying GPU: {e}"

    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "release" in line.lower():
                    info["cuda_toolkit_version"] = line.strip()
    except FileNotFoundError:
        info["cuda_toolkit_version"] = "nvcc not found"
    except Exception as e:
        info["cuda_toolkit_version"] = f"error: {e}"

    return info


def get_software_versions():
    versions = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    for pkg in ["taichi", "numpy", "scipy", "pycryptodome"]:
        try:
            mod = __import__("Crypto" if pkg == "pycryptodome" else pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


if __name__ == "__main__":
    print("=== GPU / Hardware ===")
    gpu_info = get_gpu_info()
    for k, v in gpu_info.items():
        print(f"  {k:24s}: {v}")

    print("\n=== Software Versions ===")
    sw_info = get_software_versions()
    for k, v in sw_info.items():
        print(f"  {k:24s}: {v}")

    print("\n=== Taichi backend detail ===")
    try:
        import taichi as ti
        from gpu_backend import detect_best_backend
        backend = detect_best_backend()
        ti.init(arch=getattr(ti, backend))
        print(f"  {'selected_backend':24s}: {backend}")
        print(f"  {'resolved_arch':24s}: {ti.lang.impl.current_cfg().arch}")
    except Exception as e:
        print(f"  Could not initialize Taichi: {e}")

    print("\n--- Copy the block above into the paper's experimental setup / reproducibility section ---")
