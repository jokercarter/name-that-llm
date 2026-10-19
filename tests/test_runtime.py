"""Runtime profile detection and merge order."""

from __future__ import annotations

import json

import pytest

from ntl_pact18.config import load_config
from ntl_pact18.runtime import (
    LOCAL_QWEN_ID,
    build_profile,
    detect_platform,
    find_competition_dir,
    find_qwen_model,
)


def test_detect_platform_reads_kaggle_env(monkeypatch):
    monkeypatch.setattr("ntl_pact18.runtime.KAGGLE_INPUT", __import__("pathlib").Path("/nonexistent"))
    platform, reason = detect_platform({"KAGGLE_KERNEL_RUN_TYPE": "Batch"})
    assert platform == "kaggle"
    assert "KAGGLE" in reason


def test_detect_platform_defaults_to_local(monkeypatch):
    monkeypatch.setattr("ntl_pact18.runtime.KAGGLE_INPUT", __import__("pathlib").Path("/nonexistent"))
    platform, _ = detect_platform({})
    assert platform == "local"


def test_local_profile_is_operational_only():
    profile = build_profile("local")
    overrides = profile.as_overrides()
    assert profile.name == "local"
    assert overrides["paths"]["data_dir"].endswith("data/real")
    assert overrides["paths"]["artifacts_dir"].endswith("artifacts/real")
    assert overrides["features"]["qwen"]["model_id"] == LOCAL_QWEN_ID
    assert overrides["device"]["require_gpu"] is False
    assert overrides["device"]["preference"] == "auto"


def _fake_kaggle_root(tmp_path, with_qwen: bool = True):
    comp = tmp_path / "competitions" / "name-that-llm"
    comp.mkdir(parents=True)
    (comp / "test.csv").write_text("id,text\n", encoding="utf-8")
    if with_qwen:
        model = tmp_path / "models" / "qwen3-8b" / "transformers" / "default" / "1"
        model.mkdir(parents=True)
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    return tmp_path


def test_kaggle_profile_finds_competition_and_qwen(tmp_path):
    root = _fake_kaggle_root(tmp_path)
    profile = build_profile("kaggle", root=root)
    assert profile.data_dir.endswith("name-that-llm")
    assert profile.qwen_enabled is True
    assert profile.qwen_model.endswith("qwen3-8b/transformers/default/1")
    assert profile.require_gpu is True
    assert profile.device_preference == "cuda"
    assert profile.artifacts_dir.endswith("artifacts")


def test_kaggle_profile_without_qwen_disables_member(tmp_path):
    root = _fake_kaggle_root(tmp_path, with_qwen=False)
    profile = build_profile("kaggle", root=root)
    assert profile.qwen_enabled is False
    assert profile.as_overrides()["features"]["qwen"]["enabled"] is False


def test_finders_are_tolerant_of_missing_dirs(tmp_path):
    assert find_competition_dir(tmp_path) is None
    assert find_qwen_model(tmp_path) is None


def test_config_profile_merge_order_is_overrideable():
    cfg = load_config("configs/real.yaml", profile="local")
    assert cfg.get("paths.data_dir").endswith("data/real")
    assert cfg.get("features.qwen.enabled") is True

    cfg2 = load_config(
        "configs/real.yaml",
        overrides=["paths.data_dir=/tmp/custom"],
        profile="local",
    )
    assert cfg2.get("paths.data_dir") == "/tmp/custom"


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        build_profile("colab")


def test_off_profile_leaves_config_untouched():
    cfg = load_config("configs/real.yaml", profile="off")
    assert cfg.get("paths.data_dir") == "data/real"
