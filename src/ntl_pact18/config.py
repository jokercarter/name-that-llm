"""Configuration loading: YAML + dotted overrides + ``NTL_*`` environment variables.

Precedence (low to high):
    configs/default.yaml  <  --config file  <  --set key=value  <  environment

Environment variable forms (both accepted, later wins):
    NTL_PRISM__LAM=1.5      double underscore separates nesting levels
    NTL_PRISM_LAM=1.5       single underscore, resolved by longest-prefix match
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

_MISSING = object()


class Config:
    """Thin, forgiving wrapper around a nested dict."""

    def __init__(self, raw: Dict[str, Any], source: Optional[Path] = None):
        self.raw = raw
        self.source = source

    # ---------------------------------------------------------------- access
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted, _MISSING)
        if value is _MISSING:
            raise KeyError(f"missing required config key: {dotted}")
        return value

    def section(self, dotted: str) -> Dict[str, Any]:
        value = self.get(dotted, {})
        return copy.deepcopy(value) if isinstance(value, Mapping) else {}

    def path(self, dotted: str, default: str = "") -> Path:
        raw = self.get(dotted, default)
        p = Path(str(raw))
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    # ----------------------------------------------------------------- mutation
    def set(self, dotted: str, value: Any) -> "Config":
        parts = dotted.split(".")
        node = self.raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise TypeError(f"cannot descend into non-dict at {part!r}")
        node[parts[-1]] = value
        return self

    def update(self, other: Mapping[str, Any]) -> "Config":
        self.raw = deep_merge(self.raw, other)
        return self

    def copy(self) -> "Config":
        return Config(copy.deepcopy(self.raw), self.source)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Config(source={self.source}, keys={sorted(self.raw)})"


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    """Parse an env/CLI value into bool/int/float/list/None when possible."""
    lowered = text.strip()
    low = lowered.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"none", "null"}:
        return None
    try:
        return int(lowered)
    except ValueError:
        pass
    try:
        return float(lowered)
    except ValueError:
        pass
    if lowered.startswith("[") and lowered.endswith("]"):
        body = lowered[1:-1].strip()
        if not body:
            return []
        return [_coerce(p) for p in body.split(",")]
    return lowered


def parse_set_overrides(items: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Turn ``["prism.lam=1.5", "clustering.method=tfidf"]`` into a nested dict."""
    overrides: Dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        parts = key.strip().split(".")
        node = overrides
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return overrides


def _env_overrides(environ: Mapping[str, str], known_prefixes: Iterable[str]) -> Dict[str, Any]:
    """Resolve ``NTL_*`` variables against the known top-level config keys."""
    prefixes = sorted({p.lower() for p in known_prefixes}, key=len, reverse=True)
    result: Dict[str, Any] = {}
    for name, raw_value in environ.items():
        if not name.upper().startswith("NTL_"):
            continue
        body = name[4:]
        if not body:
            continue
        lowered = body.lower()
        matched = None
        for prefix in prefixes:
            for sep in ("__", "_"):
                candidate = prefix + sep
                if lowered.startswith(candidate):
                    matched = (prefix, body[len(candidate):])
                    break
            if matched:
                break
        if not matched:
            continue
        section, rest = matched
        rest = rest.replace("__", ".")
        node = result.setdefault(section, {})
        parts = [p for p in rest.split(".") if p]
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        if parts:
            node[parts[-1]] = _coerce(raw_value)
    return result


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Iterable[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
    base_path: Optional[str | Path] = None,
    profile: Optional[str] = None,
) -> Config:
    """Load the layered configuration.

    ``profile`` (``auto`` / ``local`` / ``kaggle``) injects only operational
    settings - paths, device policy, Qwen checkpoint - between the YAML file and
    the ``--set`` overrides, so explicit flags always win.
    """
    base_file = Path(base_path) if base_path else DEFAULT_CONFIG
    raw: Dict[str, Any] = {}
    if base_file.exists():
        raw = yaml.safe_load(base_file.read_text(encoding="utf-8")) or {}

    source = base_file
    if path:
        cfg_file = Path(path)
        if not cfg_file.is_absolute() and not cfg_file.exists():
            cfg_file = PROJECT_ROOT / "configs" / cfg_file
        if not cfg_file.exists():
            raise FileNotFoundError(f"config not found: {path}")
        extra = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
        raw = deep_merge(raw, extra)
        source = cfg_file

    if profile and profile != "off":
        from .runtime import build_profile

        raw = deep_merge(raw, build_profile(profile, environ=environ).as_overrides())
    raw = deep_merge(raw, parse_set_overrides(overrides))
    raw = deep_merge(raw, _env_overrides(environ or os.environ, raw.keys()))
    return Config(raw, source)


def dump_config(cfg: Config) -> str:
    return yaml.safe_dump(cfg.raw, sort_keys=False, allow_unicode=True)
