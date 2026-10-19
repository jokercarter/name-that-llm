#!/usr/bin/env python
"""Project the wall-clock cost of a QLoRA configuration on T4 cards.

Cost is sample-pass bound, not optimizer-step bound: a LoRA backward is cheap
next to the frozen 4-bit matmuls, so the time is almost linear in

    sample_passes x seconds_per_sample / gpus

Sample passes for one epoch are ``rows x folds``: each of the ``k`` folds trains
on ``(k-1)/k`` of the data (``rows x (k-1)`` in total) and the final refit adds
one more full copy. Forgetting the refit is the most common way an 8-hour plan
turns into 10.

``seconds_per_sample`` is derived from FLOPs unless measured:

    6 x params x padded_tokens / (peak_tflops x mfu)

T4 fp16 peak is 65 TFLOPS; 4-bit dequant, gradient checkpointing and Python
overhead put real MFU around 0.18. Length bucketing matters more than anything
else here: random batching pads every row to the longest row in its batch
(~340 tokens), bucketing brings the average down to ~150.

    python scripts/estimate_qlora_budget.py
    python scripts/estimate_qlora_budget.py --sec-per-sample 0.16 --rows 50003
"""

from __future__ import annotations

import argparse

T4_PEAK_TFLOPS = 65.0
# Calibrated on the real run (1.7B, batch 8/card, 384 max_len, bucketed):
# 623 s for 2,880 sample-passes -> 0.216 s per pass across both cards
# (0.43 s per sample on one card), i.e. MFU ~0.055. The textbook 0.18 was
# optimistic by 3x, so this default is the measured one.
DEFAULT_MFU = 0.055

# (name, params_b, padded_tokens, gpus)
PRESETS = [
    ("1.7B x 2 GPU, 3 folds",  1.7, 150, 2),
    ("1.7B x 2 GPU, 4 folds",  1.7, 150, 2),
    ("1.7B x 2 GPU, 5 folds",  1.7, 150, 2),
    ("4.0B x 2 GPU, 3 folds",  4.0, 150, 2),
    ("8.0B x 2 GPU, 3 folds",  8.0, 150, 2),
]
# folds per preset, used only for the table below
PRESET_FOLDS = [3, 4, 5, 3, 3]


def sec_per_sample(params_b: float, tokens: int, mfu: float = DEFAULT_MFU) -> float:
    flops = 6.0 * params_b * 1e9 * tokens
    effective = T4_PEAK_TFLOPS * mfu * 1e12
    return flops / effective


def hours(rows: int, folds: int, gpus: int, sec: float, epochs: float = 1.0,
          efficiency: float = 1.0) -> float:
    """``sec`` is per sample on one card; the measurement already includes both
    cards running in parallel, so no extra scaling factor is applied."""
    passes = rows * folds * epochs
    return passes * sec / max(1, gpus) / efficiency / 3600.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50003, help="rows in the training file")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--mfu", type=float, default=DEFAULT_MFU)
    ap.add_argument("--sec-per-sample", type=float, default=None,
                    help="measured value; overrides the FLOPs model")
    args = ap.parse_args()

    print(f"rows={args.rows} folds={args.folds} epochs={args.epochs} "
          f"mfu={args.mfu:.2f} peak={T4_PEAK_TFLOPS:.0f} TFLOPS")
    print(f"sample passes per epoch = rows x folds = {args.rows * args.folds:,}")
    print()
    header = (f"{'config':26s} {'params':>7s} {'folds':>5s} {'tok':>4s} {'gpu':>4s} "
              f"{'s/sample':>9s} {'hours':>6s} {'verdict':>8s}")
    print(header)
    print("-" * len(header))
    for (name, params, tokens, gpus), k in zip(PRESETS, PRESET_FOLDS):
        sec = sec_per_sample(params, tokens, args.mfu)
        h = hours(args.rows, k, gpus, sec, args.epochs)
        verdict = "fits 9h" if h <= 8.6 else ("tight" if h <= 9.2 else "over 9h")
        print(f"{name:26s} {params:6.1f}B {k:5d} {tokens:4d} {gpus:4d} "
              f"{sec:9.3f} {h:6.2f} {verdict:>8s}")

    if args.sec_per_sample is not None:
        h = hours(args.rows, args.folds, args.gpus, args.sec_per_sample, args.epochs)
        print("-" * len(header))
        print(f"{'measured':40s} {'-':>7s} {'-':>4s} {args.gpus:4d} "
              f"{args.sec_per_sample:9.3f} {h:6.2f}")
        print("\n9 h budget at that rate:")
        per_fit = args.rows * (args.folds - 1) / args.folds
        print(f"  rows per fold fit  : {int(per_fit):,}")
        print(f"  max rows in 9 h    : "
              f"{int(9 * 3600 * args.gpus * 0.92 / (args.folds * args.sec_per_sample)):,}")
        print(f"  per-fit hard cap   : "
              f"{9 * 3600 * args.gpus * 0.92 / (args.folds * args.sec_per_sample * 4):.0f} "
              f"rows  (if each of the 4 fits gets 1/4 of the budget)")

    print("\nHow many folds fill a budget (default 8 h), same rows:")
    target = 8.0
    for params, tokens, gpus in [(1.7, 150, 2), (4.0, 150, 2)]:
        sec = sec_per_sample(params, tokens, args.mfu)
        k = int(target * 3600 * gpus * 0.92 / (args.rows * sec))
        print(f"  {params:.1f}B x {gpus} GPU: {k} folds fit {target:.0f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
