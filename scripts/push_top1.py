"""Push the vendored top1 notebook (Public 0.636) to Kaggle as our own kernel.

The notebook itself is unmodified; only the kernel identity changes. All 16
datasets, 8 kernel outputs and 7 base models come from
``vendor/top1_fleongg/kernel-metadata.original.json`` so the input set stays
byte-for-byte identical to the public 0.636 run.

Usage
-----
    python scripts/push_top1.py                # build + push
    python scripts/push_top1.py --build-only   # just stage .tmp/kpush_top1
    python scripts/push_top1.py --slug my-slug # override the kernel slug
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "top1_fleongg"
STAGE = ROOT / ".tmp" / "kpush_top1"
KAGGLE_PY = ROOT / ".tmp" / "kaggle-cli" / "Scripts" / "python.exe"
NB_NAME = "ntl-part-two-complement.ipynb"


def build(slug: str, owner: str) -> Path:
    src_nb = VENDOR / NB_NAME
    src_meta = VENDOR / "kernel-metadata.original.json"
    for p in (src_nb, src_meta):
        if not p.exists():
            raise SystemExit(f"missing vendored file: {p}")

    original = json.loads(src_meta.read_text(encoding="utf-8"))
    meta = {
        "id": f"{owner}/{slug}",
        "title": slug,
        "code_file": NB_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": original.get("dataset_sources") or [],
        "kernel_sources": original.get("kernel_sources") or [],
        "competition_sources": original.get("competition_sources") or ["name-that-llm"],
        "model_sources": original.get("model_sources") or [],
    }

    STAGE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_nb, STAGE / NB_NAME)
    (STAGE / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"staged {STAGE} | datasets {len(meta['dataset_sources'])} "
        f"kernels {len(meta['kernel_sources'])} models {len(meta['model_sources'])}"
    )
    return STAGE


def push(stage: Path) -> int:
    cmd = [str(KAGGLE_PY), "-m", "kaggle", "kernels", "push", "-p", str(stage)]
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="ntl-top1-repro")
    ap.add_argument("--owner", default="jokercarter")
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()

    stage = build(args.slug, args.owner)
    if args.build_only:
        return 0
    if not KAGGLE_PY.exists():
        print(f"kaggle CLI not found at {KAGGLE_PY}", file=sys.stderr)
        return 2
    return push(stage)


if __name__ == "__main__":
    raise SystemExit(main())
