"""Package the runnable code tree into ``ntl-pact18-code.zip`` for the Kaggle
dataset ``jokercarter/ntl-pact18-verify``.

The zip carries a single top-level folder ``ntl-pact18-code/`` so the kernel
can unzip it into ``/kaggle/working`` and find ``src/``, ``configs/`` and
``scripts/`` at known paths. Only files the runtime needs are included; caches,
virtualenvs and data are deliberately excluded (the data ships as separate
dataset files).

Usage:
    python scripts/package_code.py [--out .tmp/ds_verify/ntl-pact18-code.zip]
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# top-level entries copied verbatim into the zip
INCLUDE_FILES = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-gpu.txt",
    "run.py",
]
INCLUDE_DIRS = ["src", "scripts", "configs", "tests"]

# never ship these (build artefacts, local state, data)
SKIP_DIR_NAMES = {
    "__pycache__", ".pytest_cache", ".venv", "venv", ".git", ".tmp",
    "artifacts", "data", "docs", ".workbuddy", ".ruff_cache", ".mypy_cache",
}
SKIP_SUFFIXES = {".pyc", ".pyo", ".log", ".parquet", ".csv", ".zip", ".npy", ".joblib"}


def build(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in INCLUDE_FILES:
            p = ROOT / rel
            if p.is_file():
                z.write(p, (Path("ntl-pact18-code") / rel).as_posix())
                n += 1
        for d in INCLUDE_DIRS:
            base = ROOT / d
            if not base.is_dir():
                continue
            for f in sorted(base.rglob("*")):
                if not f.is_file():
                    continue
                if set(f.parts) & SKIP_DIR_NAMES or f.suffix in SKIP_SUFFIXES:
                    continue
                z.write(f, (Path("ntl-pact18-code") / f.relative_to(ROOT)).as_posix())
                n += 1
    print(f"packed {n} files | {out.stat().st_size / 1024:.1f} KB -> {out}")
    for key in [
        "scripts/train_parallel.py",
        "src/ntl_pact18/dist_utils.py",
        "src/ntl_pact18/features/qwen_member.py",
        "src/ntl_pact18/members.py",
        "configs/kaggle_2stage_8b.yaml",
        "tests/test_two_stage.py",
    ]:
        with zipfile.ZipFile(out) as z:
            names = set(z.namelist())
        print(f"  has {key}: {str((Path('ntl-pact18-code') / key).as_posix()) in names}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".tmp/ds_verify/ntl-pact18-code.zip")
    args = ap.parse_args()
    build(Path(args.out))
