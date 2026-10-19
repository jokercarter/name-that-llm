#!/usr/bin/env python
"""Build a training file that mixes competition rows with external passages.

Two sizing modes:

``--quota N``          fixed external rows per class (the small-data recipe)
``--target-total N``   aim for N rows in total; the per-class quota is derived
                       from how many competition rows already exist

``--oversample`` fills a short class by sampling with replacement, which is what
it takes to reach a clean 50k: the corpus caps ``qwen3.8-max`` at ~1.5k rows, so
18 classes x 2,649 external rows is only reachable with ~1.1k repeats (~2% of the
file). Duplicates keep their ``prompt_id``, so the grouped folds always place
them in the same fold and no duplicate can straddle a train/validation boundary.

Every emitted row carries an ``origin`` column (``competition`` / ``external`` /
``external_repeat``). The trainer uses it to restrict calibration, blend weights
and every reported metric to the competition distribution: the 50k file is ~96%
external text and the hidden test set is 100% competition text, so fitting the
stacking layer on all rows would optimise for the wrong domain.

    python scripts/build_external_mix.py --target-total 50000 --oversample \
        --external data/external/ntl_merged_external_large.parquet \
        --out data/real/trainval_plus_ext_50k.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["id", "prompt_id", "text", "model"]


def _take_per_class(ext: pd.DataFrame, quota: int, seed: int,
                    oversample: bool = False) -> pd.DataFrame:
    parts = []
    for _label, group in ext.groupby("model", sort=False):
        pool = group.drop_duplicates(subset=["text"])
        take = pool.sample(min(len(pool), quota), random_state=seed).copy()
        take["_repeat"] = False
        if oversample and len(take) < quota:
            extra = pool.sample(quota - len(take), replace=True,
                                random_state=seed + 1).copy()
            extra["_repeat"] = True
            take = pd.concat([take, extra], ignore_index=True)
        parts.append(take)
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/real/trainval.csv")
    ap.add_argument("--external", default="data/external/ntl_merged_external.parquet")
    ap.add_argument("--quota", type=int, default=0, help="external rows per class")
    ap.add_argument("--target-total", type=int, default=0,
                    help="derive the per-class quota from a target row count")
    ap.add_argument("--oversample", action="store_true",
                    help="fill short classes by sampling with replacement")
    ap.add_argument("--out", default="data/real/trainval_plus_ext.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    base = pd.read_csv(args.base)
    ext = pd.read_parquet(args.external)
    missing = set(base["model"]) - set(ext["model"])
    if missing:
        print(f"[warn] external data has no rows for: {sorted(missing)}")

    n_classes = int(base["model"].nunique())
    quota = args.quota
    if args.target_total > 0:
        budget = max(0, args.target_total - len(base))
        quota = max(1, int(np.ceil(budget / max(1, n_classes))))
        print(f"[mix] target {args.target_total} rows -> {quota}/class external")
    if quota <= 0:
        raise SystemExit("pass --quota or --target-total")

    picked = _take_per_class(ext, quota, args.seed, oversample=args.oversample)

    base_out = base[COLUMNS].copy()
    base_out["origin"] = "competition"
    ext_out = picked[COLUMNS].copy()
    ext_out["origin"] = np.where(picked["_repeat"].to_numpy(),
                                 "external_repeat", "external")
    # give the oversampled copies their own id, otherwise the training file has
    # duplicate ids and any later join on ``id`` double-counts the tail classes
    rep = ext_out["origin"] == "external_repeat"
    if rep.any():
        # a row can be drawn more than once, so the suffix needs a counter
        seen = ext_out.groupby("id").cumcount()
        ext_out.loc[rep, "id"] = (
            ext_out.loc[rep, "id"].astype(str) + "_r" + seen[rep].astype(str)
        )
    mixed = pd.concat([base_out, ext_out], ignore_index=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mixed.to_csv(out, index=False)

    counts = mixed["model"].value_counts()
    print(f"base rows     : {len(base_out)}")
    print(f"external rows : {len(ext_out)} ({quota}/class cap)")
    print(f"repeats       : {int((mixed['origin'] == 'external_repeat').sum())} rows")
    print(f"mixed rows    : {len(mixed)}")
    print(f"per-class min : {counts.min()}  max: {counts.max()}")
    print(f"prompt groups : {mixed['prompt_id'].nunique()}")
    print(f"median words  : {mixed['text'].str.split().str.len().median():.0f}")
    print(f"written       : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
