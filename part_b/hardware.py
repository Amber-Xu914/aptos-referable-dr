"""5. Runtime and hardware notes.

hardware_info() records what a run used (GPU, CUDA, library versions, CPU, RAM); every
training run stores it in its summary JSON, so the runtime table in the report comes from
the runs themselves rather than from memory.

    python part_b/hardware.py          prints the info and writes results/part_b/hardware.json
"""
import os
import platform
import sys
import time
from pathlib import Path
from contextlib import contextmanager


def hardware_info():
    info = dict(python=sys.version.split()[0], platform=platform.platform(), cpu_count=os.cpu_count(),
                colab="COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ)
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        try:
            info["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
        except (ValueError, AttributeError, OSError):
            pass
    try:
        import torch
        import torchvision
        info.update(torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.cuda.is_available(),
                    cpu=platform.processor() or platform.machine())
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            info.update(gpu=prop.name, gpu_mem_gb=round(prop.total_memory / 1e9, 1),
                        cuda_version=torch.version.cuda, cudnn=torch.backends.cudnn.version())
        elif _mps_ok():
            info.update(gpu=f"Apple MPS ({platform.machine()})", mps=True)
        else:
            info["gpu"] = "none (CPU only)"
    except ImportError:
        info["torch"] = "not installed"
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    return info


def _mps_ok():
    import torch
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


_DEVICE = None


def device():
    """GPU first, CPU as fallback: NVIDIA (CUDA), then Apple silicon (MPS), then CPU.

    The GPU is only used after a small test calculation succeeds on it, so a GPU that PyTorch
    reports but cannot actually run on (e.g. too old for this PyTorch build) falls back to the
    CPU with a warning instead of crashing in the middle of training.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    import torch
    for kind, ok in (("cuda", torch.cuda.is_available), ("mps", _mps_ok)):
        if not ok():
            continue
        try:
            x = torch.ones(8, 8, device=kind)
            float((x @ x).sum())                        # forces the work to run on the device
            _DEVICE = torch.device(kind)
            return _DEVICE
        except Exception as e:                          # noqa: BLE001 - any failure means "do not use it"
            print(f"WARNING: PyTorch sees a {kind} GPU but cannot run on it ({type(e).__name__}: "
                  f"{str(e).splitlines()[0][:150]}). Using the CPU instead.", flush=True)
    _DEVICE = torch.device("cpu")
    return _DEVICE


def nvidia_smi():
    """(GPU name, driver's highest supported CUDA version) from nvidia-smi, or None if no NVIDIA driver."""
    import re
    import shutil
    import subprocess
    exe = shutil.which("nvidia-smi")
    if exe is None and sys.platform.startswith("win"):
        cand = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
        exe = str(cand) if cand.exists() else None
    if exe is None:
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=15).stdout
        name = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    return (name[0] if name else "NVIDIA GPU"), (m.group(1) if m else None)


def python_cmd():
    """This Python as a command to paste in a terminal (PowerShell, VS Code's Windows default, needs '&')."""
    return f'& "{sys.executable}"' if sys.platform.startswith("win") else f'"{sys.executable}"'


def gpu_advice():
    """If an NVIDIA GPU exists but PyTorch is not using it, explain why and how to fix it (else None)."""
    import torch
    if torch.cuda.is_available() or _mps_ok():
        return None
    smi = nvidia_smi()
    py = python_cmd()
    if smi is None:
        return ("No NVIDIA GPU driver found (nvidia-smi is missing), so training runs on the CPU. If the "
                "computer has an NVIDIA GPU, install the latest driver from nvidia.com/drivers and run again.")
    gpu, driver_cuda = smi
    if torch.version.cuda is None:
        # CPU-only PyTorch build. Suggest the newest wheel the driver can run (pytorch.org lists the current ones).
        suggest = None
        if driver_cuda:
            major, minor = (int(v) for v in driver_cuda.split(".")[:2])
            for tag, need in (("cu128", (12, 8)), ("cu126", (12, 6)), ("cu124", (12, 4)), ("cu121", (12, 1)),
                              ("cu118", (11, 8))):
                if (major, minor) >= need:
                    suggest = tag
                    break
        lines = [f"Found {gpu} (driver supports CUDA {driver_cuda or '?'}), but the installed PyTorch "
                 f"({torch.__version__}) is the CPU-only build, so it cannot use the GPU.",
                 "Installing the CUDA toolkit does not change this: the GPU build of PyTorch brings its own CUDA.",
                 "Replace PyTorch with the GPU build (run in the VS Code terminal):",
                 f'  {py} -m pip uninstall -y torch torchvision']
        if suggest:
            lines.append(f'  {py} -m pip install torch torchvision --index-url https://download.pytorch.org/whl/{suggest}')
        lines += ["Check https://pytorch.org/get-started/locally/ for the current command: choose a CUDA version",
                  f"no higher than {driver_cuda or 'the one nvidia-smi shows'} and replace 'pip3' with "
                  f"{py} -m pip.",
                  f'Then check:  {py} -c "import torch; print(torch.cuda.is_available())"   -> True']
        if not suggest and driver_cuda:
            lines.insert(3, f"Your driver (CUDA {driver_cuda}) is old; update it from nvidia.com/drivers first.")
        return "\n".join(lines)
    return (f"Found {gpu} and PyTorch is built for CUDA {torch.version.cuda}, but CUDA is not available. "
            f"The NVIDIA driver supports CUDA {driver_cuda or '?'}; if that is lower than {torch.version.cuda}, "
            "update the driver from nvidia.com/drivers (or install a PyTorch build for an older CUDA), then restart.")


def amp(dev):
    """Mixed-precision context: fp16 autocast on CUDA only (MPS/CPU run in fp32)."""
    import contextlib
    import torch
    return torch.autocast("cuda") if dev.type == "cuda" else contextlib.nullcontext()


def default_workers():
    """DataLoader workers. Windows/macOS start workers by 'spawn', which copies the preloaded
    images into every worker, so augmentation runs in the main process there."""
    return 2 if sys.platform.startswith("linux") else 0


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif _mps_ok():
        torch.mps.synchronize()


@contextmanager
def timer(store, key):
    """with timer(d, "train_s"): ...  -> d["train_s"] = elapsed seconds (GPU work included)."""
    sync(); t0 = time.perf_counter()
    yield
    sync(); store[key] = round(time.perf_counter() - t0, 2)


def peak_gpu_mem_gb():
    import torch
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None


if __name__ == "__main__":
    import argparse
    import json
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from part_b.common import add_path_args, get_paths, save_json
    ap = argparse.ArgumentParser(description="Print and save hardware/software info.")
    add_path_args(ap)
    a = ap.parse_args()
    info = hardware_info()
    print(json.dumps(info, indent=2))
    save_json(info, get_paths(a.root, a.cache).results / "hardware.json")
    dev = device()
    print(f"\ntraining device: {dev.type}")
    advice = gpu_advice()
    if advice:
        print("\n" + advice)
