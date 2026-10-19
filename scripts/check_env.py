#!/usr/bin/env python
"""Print the runtime environment and check the required dependencies."""

from __future__ import annotations

import importlib
import sys

REQUIRED = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("yaml", "PyYAML"),
    ("joblib", "joblib"),
]
OPTIONAL = [("lightgbm", "lightgbm (optional boosting backend)"), ("pytest", "pytest (dev)")]


def main() -> int:
    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    missing = 0
    for module, label in REQUIRED:
        try:
            mod = importlib.import_module(module)
            print(f"  [ok]   {label:14s} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # pragma: no cover
            missing += 1
            print(f"  [MISS] {label:14s} {exc}")
    for module, label in OPTIONAL:
        try:
            mod = importlib.import_module(module)
            print(f"  [ok]   {label:34s} {getattr(mod, '__version__', '?')}")
        except Exception:
            print(f"  [--]   {label:34s} not installed (graceful fallback)")
    if missing:
        print("\nmissing required packages. install with:\n  python -m pip install -r requirements.txt")
        return 1
    print("\nenvironment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
