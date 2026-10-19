import numpy as np

from ntl_pact18.config import load_config
from ntl_pact18.device import detect_device, enforce_policy


def test_device_detection_returns_valid_kind():
    info = detect_device(load_config())
    assert info.kind in {"cuda", "mps", "cpu"}
    assert isinstance(info.name, str) and info.name
    payload = info.as_dict()
    assert set(payload) >= {"kind", "name", "preference", "backends", "fallback_reason"}
    assert "selected device" in info.describe()


def test_forced_cpu_never_reports_accelerator():
    cfg = load_config(overrides=["device.preference=cpu"])
    info = detect_device(cfg)
    assert info.kind == "cpu"
    assert info.is_accelerator is False
    assert "forced" in info.fallback_reason


def test_require_gpu_raises_when_no_accelerator(monkeypatch):
    cfg = load_config(overrides=["device.preference=cpu", "device.require_gpu=true"])
    info = detect_device(cfg)
    try:
        enforce_policy(info, cfg)
    except RuntimeError as exc:
        assert "require_gpu" in str(exc)
    else:  # pragma: no cover - only when a GPU is actually selected
        assert info.is_accelerator


def test_device_report_mentions_backends():
    info = detect_device(load_config())
    assert isinstance(info.backends, dict)
    assert "torch" in info.backends
