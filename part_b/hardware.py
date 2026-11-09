""" Runtime and hardware notes.

hardware_info() records what a run used (GPU, CUDA, library versions, CPU, RAM); every
training run stores it in its summary JSON, so the runtime table in the report comes from
the runs themselves rather than from memory.

prints the info and writes results/part_b/hardware.json
"""
import os
import platform
import sys
import time
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
        info.update(torch=torch.__version__, torchvision=torchvision.__version__, cuda=torch.cuda.is_available())
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            info.update(gpu=prop.name, gpu_mem_gb=round(prop.total_memory / 1e9, 1),
                        cuda_version=torch.version.cuda, cudnn=torch.backends.cudnn.version())
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


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
    import json
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from part_b.common import get_paths, save_json
    info = hardware_info()
    print(json.dumps(info, indent=2))
    save_json(info, get_paths().results / "hardware.json")