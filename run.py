#!/usr/bin/env python
"""Entry point that works without installing the package.

    python run.py smoke
    python run.py train --config configs/default.yaml
    python run.py predict --out artifacts/submission.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntl_pact18.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
