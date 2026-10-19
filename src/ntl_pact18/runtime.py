"""Execution-environment profiles: one algorithm, two hosts.

The local laptop (Windows, RTX 4060 Laptop 8 GB) and Kaggle (T4 x2, offline,
read-only ``/kaggle/input``) differ in exactly four operational facts:

1. where the competition CSV lives,
2. where artifacts may be written,
3. which Qwen checkpoint is reachable (HF cache id vs an attached Kaggle model),
4. whether a GPU is mandatory.

Nothing that changes what the model *is* - member list, n-gram ranges,
regularisation, folds, calibration, family experts, shield, PRISM lambda - is
touched by a profile. That is what keeps the two hosts comparable.

Detection is deliberately boring: Kaggle sets ``KAGGLE_KERNEL_RUN_TYPE`` inside
a notebook/kernel, and a local checkout never has ``/kaggle/input``. A profile
can also be forced (``profile="local"`` / ``"kaggle"``) so a run is reproducible
even if detection changes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
COMPETITION_SLUG = "name-that-llm"
LOCAL_QWEN_ID = "Qwen/Qwen3-8B-Base"

PLATFORMS = ("auto", "local", "kaggle", "off")


@dataclass(frozen=True)
class RuntimeProfile:
    """Resolved operational settings for one host."""

    name: str
    reason: str
    data_dir: str
    artifacts_dir: str
    qwen_model: Optional[str]
    qwen_enabled: bool
    require_gpu: bool
    device_preference: str

    def as_overrides(self) -> Dict[str, Any]:
        """Config fragment merged into the YAML (before ``--set`` and env)."""
        return {
            "paths": {
                "data_dir": self.data_dir,
                "artifacts_dir": self.artifacts_dir,
            },
            "device": {
                "preference": self.device_preference,
                "require_gpu": self.require_gpu,
            },
            "features": {
                "qwen": {
                    "enabled": self.qwen_enabled,
                    "model_id": self.qwen_model or LOCAL_QWEN_ID,
                }
            },
        }


def detect_platform(environ: Optional[Mapping[str, str]] = None) -> Tuple[str, str]:
    """Return ``(platform, reason)`` using cheap, deterministic markers."""
    env = environ if environ is not None else os.environ
    if env.get("KAGGLE_KERNEL_RUN_TYPE") or env.get("KAGGLE_URL_BASE"):
        return "kaggle", "KAGGLE_KERNEL_RUN_TYPE/KAGGLE_URL_BASE is set"
    if KAGGLE_INPUT.is_dir():
        return "kaggle", "/kaggle/input exists"
    return "local", "no Kaggle markers"


def find_competition_dir(root: Path = KAGGLE_INPUT) -> Optional[Path]:
    """Locate the mounted competition folder (layout differs by Kaggle era)."""
    for cand in (root / COMPETITION_SLUG, root / "competitions" / COMPETITION_SLUG):
        if (cand / "test.csv").exists():
            return cand
    return None


def find_qwen_model(root: Path = KAGGLE_INPUT) -> Optional[str]:
    """Return the directory of an attached Qwen3 checkpoint, if any.

    Kaggle model mounts nest the weights a few levels deep, so we look for a
    ``config.json`` whose ``model_type`` starts with ``qwen3`` instead of
    hard-coding a path that changes with every mirror.
    """
    try:
        candidates = Path(root).rglob("config.json")
    except OSError:  # pragma: no cover - unreadable mount
        return None
    for cfg in candidates:
        try:
            meta = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(meta.get("model_type", "")).startswith("qwen3"):
            return cfg.parent.as_posix()
    return None


def gpu_inventory() -> Dict[str, Any]:
    """Best-effort GPU description; never raises."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch absent
        return {"cuda": False, "error": repr(exc)}
    try:
        if not torch.cuda.is_available():
            return {"cuda": False, "count": 0}
        props = torch.cuda.get_device_properties(0)
        return {
            "cuda": True,
            "count": int(torch.cuda.device_count()),
            "name": str(props.name),
            "vram_gb": round(props.total_memory / 2**30, 1),
            "compute_capability": f"{props.major}.{props.minor}",
            "bf16": bool(torch.cuda.is_bf16_supported()),
        }
    except Exception as exc:  # pragma: no cover - driver quirks
        return {"cuda": False, "error": repr(exc)}


def build_profile(
    name: str = "auto",
    environ: Optional[Mapping[str, str]] = None,
    root: Path = KAGGLE_INPUT,
) -> RuntimeProfile:
    """Resolve a profile. ``auto`` inspects the host, the rest are explicit."""
    name = (name or "auto").lower()
    if name not in PLATFORMS:
        raise ValueError(f"unknown profile {name!r}; choose from {PLATFORMS}")

    detected, reason = detect_platform(environ)
    resolved = detected if name in ("auto", "off") else name
    if name in ("local", "kaggle"):
        reason = f"forced by profile={name}"

    if resolved == "kaggle":
        comp = find_competition_dir(root)
        qwen = find_qwen_model(root)
        return RuntimeProfile(
            name="kaggle",
            reason=reason,
            data_dir=(comp or (root / COMPETITION_SLUG)).as_posix(),
            artifacts_dir=(KAGGLE_WORKING / "artifacts").as_posix(),
            qwen_model=qwen,
            qwen_enabled=qwen is not None,
            require_gpu=True,
            device_preference="cuda",
        )

    return RuntimeProfile(
        name="local",
        reason=reason,
        data_dir=(REPO_ROOT / "data" / "real").as_posix(),
        artifacts_dir=(REPO_ROOT / "artifacts" / "real").as_posix(),
        qwen_model=LOCAL_QWEN_ID,
        qwen_enabled=True,
        require_gpu=False,
        device_preference="auto",
    )


def describe(profile: RuntimeProfile) -> str:
    """Human-readable preamble written to logs."""
    gpu = gpu_inventory()
    gpu_line = (
        f"{gpu.get('name')} ({gpu.get('vram_gb')} GB, cc {gpu.get('compute_capability')}, "
        f"bf16={gpu.get('bf16')}, x{gpu.get('count')})"
        if gpu.get("cuda")
        else f"none ({gpu.get('error', 'cuda unavailable')})"
    )
    lines = [
        f"profile      : {profile.name} ({profile.reason})",
        f"data_dir     : {profile.data_dir}",
        f"artifacts_dir: {profile.artifacts_dir}",
        f"qwen         : {profile.qwen_model or 'not found'} | enabled={profile.qwen_enabled}",
        f"device       : preference={profile.device_preference} require_gpu={profile.require_gpu}",
        f"gpu          : {gpu_line}",
    ]
    return "\n".join(lines)


def apply_profile(cfg, name: str = "auto") -> RuntimeProfile:
    """Merge a profile into an already-loaded ``Config`` in place."""
    profile = build_profile(name)
    for section, values in profile.as_overrides().items():
        for key, value in values.items():
            cfg.set(f"{section}.{key}", value)
    return profile
