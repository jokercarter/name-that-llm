#!/usr/bin/env python
"""Local entry point: force the ``local`` profile, then run the normal CLI.

Usage (same arguments as ``run.py``)::

    python scripts/run_local.py --config configs/real.yaml train
    python scripts/run_local.py --config configs/real.yaml predict \
        --model artifacts/real/model.joblib --test-file data/real/test.csv \
        --truth-file data/real/test_truth.csv

Only operational settings change (data/artifact paths, optional GPU, HF model
id). The members, folds, calibration and PRISM lambda are identical to the
Kaggle entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ntl_pact18.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--profile", "local", *sys.argv[1:]]
    raise SystemExit(main())
