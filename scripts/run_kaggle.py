#!/usr/bin/env python
"""Kaggle entry point: force the ``kaggle`` profile, then run the normal CLI.

Inside a Kaggle notebook/kernel the competition data is read-only under
``/kaggle/input`` and only ``/kaggle/working`` is writable, so artifacts land
there. The Qwen checkpoint is discovered from the attached Kaggle model; if it
is missing the run still proceeds with the remaining members.

Usage (same arguments as ``run.py``)::

    !python /kaggle/input/<code-dataset>/scripts/run_kaggle.py train
    !python /kaggle/input/<code-dataset>/scripts/run_kaggle.py predict \
        --test-file /kaggle/input/name-that-llm/test.csv

Do not run this on a laptop: it will point at ``/kaggle/input`` and require a
CUDA device.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ntl_pact18.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--profile", "kaggle", *sys.argv[1:]]
    raise SystemExit(main())
