"""Device selection: CUDA -> MPS -> CPU, with explicit fallback reporting.

The project's members are mostly scikit-learn estimators, which are CPU-only.
What *can* run on an accelerator is:

* ``torch_linear`` member  -- linear softmax on the sparse stylegram features
  (CUDA / MPS when a CUDA-enabled or Apple-silicon torch is installed);
* LightGBM's stylometry member with ``device_type='gpu'`` (when LightGBM is
  installed and built with GPU support).

Everything else is CPU-bound by nature, so the policy is: use the accelerator
where it exists, report the decision loudly, and never fail silently.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

PREFERENCE_CHOICES = ("auto", "cuda", "mps", "cpu")


def _omp_guard() -> None:
    """Allow torch + MKL to coexist (Windows conda frequently needs this)."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def import_torch():
    """Import torch lazily, tolerating a broken / CPU-only installation."""
    _omp_guard()
    try:
        import torch  # noqa: F401

        return torch
    except Exception:
        return None


def _nvidia_smi() -> Optional[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return lines[0] if lines else None
    except Exception:
        return None


@dataclass
class DeviceInfo:
    kind: str = "cpu"                 # cuda | mps | cpu
    name: str = "cpu"
    preference: str = "auto"
    torch_available: bool = False
    torch_version: str = ""
    torch_cuda_build: Optional[str] = None
    backends: Dict[str, bool] = field(default_factory=dict)
    gpu_present: Optional[str] = None
    fallback_reason: str = ""
    index: int = 0

    @property
    def is_accelerator(self) -> bool:
        return self.kind in ("cuda", "mps")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "preference": self.preference,
            "torch_available": self.torch_available,
            "torch_version": self.torch_version,
            "torch_cuda_build": self.torch_cuda_build,
            "backends": self.backends,
            "gpu_present": self.gpu_present,
            "fallback_reason": self.fallback_reason,
            "index": self.index,
        }

    def summary(self) -> str:
        head = f"device={self.kind} ({self.name})"
        if self.fallback_reason:
            head += f" | fallback: {self.fallback_reason}"
        return head

    def describe(self) -> str:
        lines = [
            f"preference      : {self.preference}",
            f"selected device : {self.kind} ({self.name})",
            f"hardware        : {self.gpu_present or 'no NVIDIA GPU detected'}",
            f"torch           : {self.torch_version or 'not importable'}"
            + (f" (cuda build {self.torch_cuda_build})" if self.torch_cuda_build else ""),
            "backends        : " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in self.backends.items()),
        ]
        if self.fallback_reason:
            lines.append(f"fallback reason : {self.fallback_reason}")
        return "\n".join(lines)

    def torch_device(self):
        torch = import_torch()
        if torch is None or self.kind == "cpu":
            return None
        if self.kind == "cuda":
            return torch.device(f"cuda:{self.index}")
        return torch.device(self.kind)


def detect_device(cfg=None, verbose: bool = False) -> DeviceInfo:
    """Resolve the compute device following CUDA -> MPS -> CPU."""
    preference = str(cfg.get("device.preference", "auto") if cfg else "auto").lower()
    if preference not in PREFERENCE_CHOICES:
        preference = "auto"
    index = int(cfg.get("device.gpu_index", 0) if cfg else 0)

    info = DeviceInfo(preference=preference, index=index, gpu_present=_nvidia_smi())
    torch = import_torch()

    if preference == "cpu":
        info.kind, info.name = "cpu", "cpu"
        info.fallback_reason = "forced by device.preference=cpu"
        info.torch_available = torch is not None
        if torch is not None:
            info.torch_version = getattr(torch, "__version__", "")
            info.torch_cuda_build = getattr(getattr(torch, "version", None), "cuda", None)
        info.backends = _backend_flags(info)
        return info

    if torch is None:
        info.kind, info.name = "cpu", "cpu"
        info.fallback_reason = "torch is not importable"
        info.backends = _backend_flags(info)
        if verbose:
            print("[device] " + info.summary())
        return info

    info.torch_available = True
    info.torch_version = getattr(torch, "__version__", "")
    info.torch_cuda_build = getattr(getattr(torch, "version", None), "cuda", None)

    cuda_ok = False
    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False

    mps_ok = False
    try:
        mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        mps_ok = False

    if preference in ("auto", "cuda") and cuda_ok:
        try:
            info.name = torch.cuda.get_device_name(index)
        except Exception:
            info.name = f"cuda:{index}"
        info.kind = "cuda"
    elif preference in ("auto", "mps") and mps_ok:
        info.kind, info.name = "mps", "apple-mps"
    else:
        info.kind, info.name = "cpu", "cpu"
        if preference == "cuda":
            info.fallback_reason = "device.preference=cuda but torch.cuda.is_available() is False"
        elif preference == "mps":
            info.fallback_reason = "device.preference=mps but torch.backends.mps is unavailable"
        elif info.torch_cuda_build is None:
            info.fallback_reason = (
                f"torch {info.torch_version} is a CPU-only build"
                + (f" although an NVIDIA GPU is present ({info.gpu_present})" if info.gpu_present else "")
            )
        else:
            info.fallback_reason = "no CUDA/MPS device reported by torch"

    info.backends = _backend_flags(info)
    if verbose:
        print("[device] " + info.summary())
    return info


def _backend_flags(info: DeviceInfo) -> Dict[str, bool]:
    flags = {"torch": info.torch_available}
    if info.torch_available:
        try:
            import torch

            flags["torch_cuda"] = bool(torch.cuda.is_available())
            flags["torch_mps"] = bool(
                getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            )
        except Exception:
            flags["torch_cuda"] = flags["torch_mps"] = False
    else:
        flags["torch_cuda"] = flags["torch_mps"] = False

    for module, key in (("cuml", "cuml"), ("lightgbm", "lightgbm"), ("xgboost", "xgboost")):
        try:
            __import__(module)
            flags[key] = True
        except Exception:
            flags[key] = False
    return flags


def enforce_policy(info: DeviceInfo, cfg) -> None:
    """Abort when the caller demanded a GPU and we could not provide one."""
    if bool(cfg.get("device.require_gpu", False)) and not info.is_accelerator:
        raise RuntimeError(
            "device.require_gpu=true but no accelerator is usable: "
            f"{info.fallback_reason or 'unknown reason'}. "
            "Install a CUDA build of torch (see README section 'GPU') or set "
            "device.require_gpu=false."
        )
