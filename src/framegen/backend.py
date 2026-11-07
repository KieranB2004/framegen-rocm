"""Device/backend abstraction for framegen.

The key insight for running on AMD:

    The ROCm *build* of PyTorch exposes the standard ``torch.cuda`` API.
    HIP is hidden below it. So the identical code path runs on both
    NVIDIA (CUDA) and AMD (ROCm); this module simply labels the vendor so
    that logs/tests can distinguish them and the CLI can give useful messages.

Selection precedence (for ``--device auto``):
    1. ``--device cpu`` / ``--device cuda:0`` / ``--device rocm:0`` (explicit)
    2. ``torch.version.rocm`` set + ``torch.cuda.is_available()`` -> ROCm
    3. ``torch.cuda.is_available()`` -> CUDA
    4. CPU
"""

import os
import torch


def _env_rocm():
    for k in ("ROCm", "ROCM_VERSION", "HIP_VERSION"):
        v = os.environ.get(k)
        if v:
            return v
    return None


def detect():
    """Return (vendor, device_name, details) without creating a device.

    vendor in {\'rocm\', \'cuda\', \'cpu\'}.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        details = {
            "device_id": 0,
            "torch_version": torch.__version__,
            "cuda_api_version": getattr(torch, "version", {}).cuda,
            "rocm_version": getattr(torch, "version", {}).rocm or _env_rocm(),
            "gpu_count": torch.cuda.device_count(),
        }
        if getattr(torch, "version", {}).rocm:
            return "rocm", name, details
        low = name.lower()
        if any(tok in low for tok in ("radeon", "mi ", "instinct", "amdgpu", "gpu")):
            return "rocm", name, details
        return "cuda", name, details
    return "cpu", "cpu", {"torch_version": torch.__version__}


def get_device(prefer=None, device_id=0):
    """Return a torch.device.

    prefer in {None, \'auto\', \'cpu\', \'rocm\', \'cuda\', \'gpu\', \'cuda:N\', \'rocm:N\'}.
    """
    if prefer is None or prefer == "auto":
        prefer = "auto"
    prefer = prefer.lower()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer.startswith("cuda:") or prefer.startswith("rocm:"):
        return torch.device(prefer.replace("rocm:", "cuda:"))
    if prefer in ("rocm", "cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_id}")
        if prefer == "rocm":
            raise RuntimeError(
                "--device rocm requested but no AMD ROCm GPU is visible. "
                "Install the ROCm driver + the ROCm wheel of PyTorch, then "
                "verify with ``python -c 'import torch; print(torch.cuda.is_available(), torch.version.rocm)'``."
            )
        raise RuntimeError(
            "--device cuda requested but no CUDA GPU is visible. "
            "Install a CUDA driver + the CUDA wheel of PyTorch, or use ``--device cpu``."
        )
    raise ValueError(f"Unknown device: {prefer!r}")


def backend_summary():
    """Convenience: one dict summarising what we will run on."""
    vendor, name, details = detect()
    device = torch.device("cuda:0") if vendor in ("rocm", "cuda") else torch.device("cpu")
    return {"backend": vendor, "device": device, "device_name": name, "details": details}
