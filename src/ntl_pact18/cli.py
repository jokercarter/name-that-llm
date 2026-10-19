"""Command line interface: ``python run.py <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from . import __version__
from .config import dump_config, load_config
from .data import add_group_folds, ensure_dataset, load_test, load_test_truth, load_train
from .evaluate import balanced_accuracy, per_class_report, summarize
from .labels import LABEL_TO_INDEX, LABELS
from .pipeline import PACT18Pipeline, _encode_groups
from .simulate import format_ablation, run_structure_ablation
from .utils import ensure_dir, fmt_table, save_json


PROFILE_COMMANDS = {"train", "predict", "simulate", "device"}


def _load(args) -> "object":
    # synth / smoke / split operate on local scratch files by definition, so a
    # host profile must not rewrite their paths.
    profile = getattr(args, "profile", "auto")
    if getattr(args, "command", None) not in PROFILE_COMMANDS:
        profile = "off"
    return load_config(args.config, overrides=args.set or [], profile=profile)


# --------------------------------------------------------------------- commands
def cmd_synth(args) -> int:
    cfg = _load(args)
    paths = ensure_dataset(cfg, force=True)
    for name, path in paths.items():
        n = sum(1 for _ in open(path, encoding="utf-8")) - 1
        print(f"{name:10s} {n:6d} rows -> {path}")
    return 0


def cmd_split(args) -> int:
    """Split one labelled file into prompt-grouped train / val / test parts."""
    cfg = _load(args)
    src = Path(args.source)
    if not src.exists():
        print(f"source not found: {src}", file=sys.stderr)
        return 2
    df = load_train(src)

    out_dir = Path(args.out) if args.out else cfg.path("paths.data_dir")
    existing = [p for p in ("train.csv", "test.csv", "val.csv") if (out_dir / p).exists()]
    if existing and not args.force:
        print(f"{out_dir} already contains {existing}; pass --force to overwrite", file=sys.stderr)
        return 2
    ensure_dir(out_dir)

    rng = np.random.default_rng(args.seed)
    prompts = df["prompt_id"].unique()
    prompts = prompts[rng.permutation(len(prompts))]
    n = len(prompts)
    n_test = max(1, int(round(n * args.test_frac)))
    n_val = max(1, int(round(n * args.val_frac)))
    n_train = n - n_val - n_test
    if n_train < 2:
        print("not enough prompts for the requested fractions", file=sys.stderr)
        return 2

    train_p = set(prompts[:n_train])
    val_p = set(prompts[n_train:n_train + n_val])
    test_p = set(prompts[n_train + n_val:])

    train_df = df[df["prompt_id"].isin(train_p)].reset_index(drop=True)
    val_df = df[df["prompt_id"].isin(val_p)].reset_index(drop=True)
    test_df = df[df["prompt_id"].isin(test_p)].reset_index(drop=True)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    trainval_df.to_csv(out_dir / "trainval.csv", index=False)
    val_df[["id", "text"]].to_csv(out_dir / "val.csv", index=False)
    val_df[["id", "model", "prompt_id"]].to_csv(out_dir / "val_truth.csv", index=False)
    test_df[["id", "text"]].to_csv(out_dir / "test.csv", index=False)
    test_df[["id", "model", "prompt_id"]].to_csv(out_dir / "test_truth.csv", index=False)

    meta = {
        "source": str(src),
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
        "prompts": {"train": len(train_p), "val": len(val_p), "test": len(test_p)},
        "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "overlap": {
            "train_val": len(train_p & val_p),
            "train_test": len(train_p & test_p),
            "val_test": len(val_p & test_p),
        },
    }
    save_json(meta, out_dir / "split_meta.json")
    print(f"[split] prompts train/val/test = {len(train_p)}/{len(val_p)}/{len(test_p)}")
    print(f"[split] rows    train/val/test = {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"[split] prompt overlap = {meta['overlap']} (must be all zeros)")
    print(f"[split] written -> {out_dir}")
    return 0


def cmd_train(args) -> int:
    cfg = _load(args)
    paths = ensure_dataset(cfg)
    train_path = Path(args.train_file) if args.train_file else paths["train"]
    train_df = load_train(train_path)
    print(f"[train] file={train_path}")
    print(f"[train] rows={len(train_df)} prompts={train_df['prompt_id'].nunique()} "
          f"models={train_df['model'].nunique()}")

    pipeline = PACT18Pipeline(cfg).fit(train_df, verbose=True)
    artifacts = cfg.path("paths.artifacts_dir")
    ensure_dir(artifacts)
    model_path = Path(args.model_out) if args.model_out else artifacts / "model.joblib"
    pipeline.save(model_path)
    save_json(pipeline.describe(), artifacts / "train_summary.json")
    (artifacts / "config_resolved.yaml").write_text(dump_config(cfg), encoding="utf-8")
    print(f"[train] saved model -> {model_path}")
    return 0


def cmd_predict(args) -> int:
    cfg = _load(args)
    paths = ensure_dataset(cfg)
    artifacts = cfg.path("paths.artifacts_dir")
    model_path = Path(args.model) if args.model else artifacts / "model.joblib"
    if not model_path.exists():
        print(f"model not found: {model_path} (run `train` first)", file=sys.stderr)
        return 2

    pipeline = PACT18Pipeline.load(model_path)
    # inference-time switches (prism / em / experts / shield) follow the config
    # given on the command line, not the one frozen at training time
    pipeline.cfg = cfg
    test_path = Path(args.test_file) if args.test_file else paths["test"]
    test_df = load_test(test_path)
    prediction = pipeline.predict(test_df, verbose=True)

    out_path = Path(args.out) if args.out else artifacts / "submission.csv"
    ensure_dir(out_path.parent)
    pd.DataFrame({"id": test_df["id"], "model": prediction.labels()}).to_csv(out_path, index=False)
    print(f"[predict] file={test_path} -> {out_path} ({len(test_df)} rows)")

    truth_path = Path(args.truth_file) if args.truth_file else paths["test_truth"]
    truth = load_test_truth(truth_path)
    if truth is not None and not args.no_eval:
        merged = test_df[["id"]].merge(truth, on="id", how="left")
        mask = merged["model"].notna().to_numpy()
        y = np.asarray([LABEL_TO_INDEX[v] for v in merged.loc[mask, "model"]], dtype=int)
        print("\n[stage-wise balanced accuracy]")
        rows = []
        for stage, preds in prediction.stage_preds.items():
            rows.append({"stage": stage, "balanced_accuracy": round(balanced_accuracy(y, preds[mask]), 4)})
        print(fmt_table(rows))
        pred_groups = prediction.groups[mask]
        true_groups = (
            _encode_groups(merged.loc[mask, "prompt_id"].tolist())
            if "prompt_id" in merged.columns
            else None
        )
        summary = summarize(prediction.probs[mask], y, prediction.preds[mask], pred_groups, true_groups)
        print("\n[final]")
        for key, value in summary.items():
            print(f"  {key:26s} {value:.4f}")
        print("\n[per-class recall]")
        print(per_class_report(y, prediction.preds[mask]))
    return 0


def cmd_simulate(args) -> int:
    cfg = _load(args)
    paths = ensure_dataset(cfg)
    artifacts = cfg.path("paths.artifacts_dir")
    model_path = Path(args.model) if args.model else artifacts / "model.joblib"
    if not model_path.exists():
        print(f"model not found: {model_path} (run `train` first)", file=sys.stderr)
        return 2
    pipeline = PACT18Pipeline.load(model_path)
    train_df = add_group_folds(load_train(paths["train"]), int(cfg.get("data.n_folds", 5)),
                              int(cfg.get("project.seed", 42)))
    y = np.asarray([LABEL_TO_INDEX[v] for v in train_df["model"]], dtype=int)
    prompt_codes = _encode_groups(train_df["prompt_id"].tolist())

    probs = pipeline.blended_oof()
    rows = run_structure_ablation(
        probs, y, prompt_codes,
        k_values=args.k,
        lam=float(cfg.get("prism.lam", 1.0)),
        alpha=float(cfg.get("calibration.sinkhorn_alpha", 0.25)),
        split_rate=args.split_rate,
        merge_rate=args.merge_rate,
        seed=int(cfg.get("project.seed", 42)),
    )
    print("\n[structure ablation | balanced accuracy on pseudo pools]")
    print(format_ablation(rows))
    if args.json:
        save_json(rows, args.json)
        print(f"saved -> {args.json}")
    return 0


def cmd_smoke(args) -> int:
    """Synthetic corpus -> fit -> predict -> structure ablation, all in one."""
    if not args.config:
        args.config = str(Path(__file__).resolve().parents[2] / "configs" / "smoke.yaml")
    cfg = _load(args)
    print("=" * 72)
    print(f"PACT-18 smoke test  |  version {__version__}")
    print("=" * 72)
    print(f"config: {cfg.source}")
    paths = ensure_dataset(cfg, force=True)
    train_df = load_train(paths["train"])
    test_df = load_test(paths["test"])
    truth = load_test_truth(paths["test_truth"])
    print(f"train: {len(train_df)} rows / {train_df['prompt_id'].nunique()} prompts")
    print(f"test : {len(test_df)} rows (truth available: {truth is not None})")

    print("\n" + "-" * 72 + "\n[1/4] fitting pipeline\n" + "-" * 72)
    pipeline = PACT18Pipeline(cfg).fit(train_df, verbose=True)

    print("\n" + "-" * 72 + "\n[2/4] predicting\n" + "-" * 72)
    prediction = pipeline.predict(test_df, verbose=True)

    print("\n" + "-" * 72 + "\n[3/4] evaluation\n" + "-" * 72)
    if truth is not None:
        merged = test_df[["id"]].merge(truth, on="id", how="left")
        mask = merged["model"].notna().to_numpy()
        y = np.asarray([LABEL_TO_INDEX[v] for v in merged.loc[mask, "model"]], dtype=int)
        rows = [
            {"stage": stage, "balanced_acc": round(balanced_accuracy(y, preds[mask]), 4)}
            for stage, preds in prediction.stage_preds.items()
        ]
        print(fmt_table(rows))
        pred_groups = prediction.groups[mask]
        true_groups = (
            _encode_groups(merged.loc[mask, "prompt_id"].tolist())
            if "prompt_id" in merged.columns
            else None
        )
        summary = summarize(prediction.probs[mask], y, prediction.preds[mask], pred_groups, true_groups)
        print("\nfinal metrics")
        for key, value in summary.items():
            print(f"  {key:26s} {value:.4f}")
        print("\nper-class recall")
        print(per_class_report(y, prediction.preds[mask]))
    else:
        print("(test truth missing; skipping metrics)")

    print("\n" + "-" * 72 + "\n[4/4] structure ablation on OOF pseudo pools\n" + "-" * 72)
    y_train = np.asarray([LABEL_TO_INDEX[v] for v in train_df["model"]], dtype=int)
    prompt_codes = _encode_groups(train_df["prompt_id"].tolist())
    ablation = run_structure_ablation(
        pipeline.blended_oof(), y_train, prompt_codes,
        k_values=[4, 8, 12, 18],
        lam=float(cfg.get("prism.lam", 1.0)),
        alpha=float(cfg.get("calibration.sinkhorn_alpha", 0.25)),
        seed=int(cfg.get("project.seed", 42)),
        repeats=2,
    )
    print(format_ablation(ablation))

    artifacts = cfg.path("paths.artifacts_dir")
    ensure_dir(artifacts)
    pipeline.save(artifacts / "model.joblib")
    pd.DataFrame({"id": test_df["id"], "model": prediction.labels()}).to_csv(
        artifacts / "submission.csv", index=False
    )
    save_json({"summary": pipeline.describe(), "ablation": ablation}, artifacts / "smoke_report.json")
    print(f"\nartifacts -> {artifacts}")
    print("smoke test finished OK")
    return 0


def cmd_device(args) -> int:
    cfg = _load(args)
    from .device import detect_device
    from .runtime import build_profile, describe as describe_profile

    profile = build_profile(getattr(args, "profile", "auto"))
    print("PACT-18 runtime profile")
    print("-" * 52)
    print(describe_profile(profile))
    print("-" * 52)

    info = detect_device(cfg, verbose=False)
    print("PACT-18 device report")
    print("-" * 52)
    print(info.describe())
    print("-" * 52)
    print(f"accelerator in use : {info.is_accelerator}")

    def _resolve(value, needs_cuda: bool = False):
        if isinstance(value, str):
            return value.lower() in ("auto", "true", "yes", "1") and info.is_accelerator
        return bool(value)

    qwen_on = _resolve(cfg.get("features.qwen.enabled", "auto")) and info.backends.get("torch", False)
    tl_on = _resolve(cfg.get("features.torch_linear.enabled", "auto"))
    print(f"qwen member        : {'enabled (' + str(cfg.get('features.qwen.mode', 'probe')) + ')' if qwen_on else 'disabled'}")
    print(f"torch_linear member: {'enabled' if tl_on else 'disabled'}")
    if info.is_accelerator and info.backends.get("xgboost"):
        booster = "xgboost-gpu (device=cuda)"
    elif info.is_accelerator and info.backends.get("lightgbm"):
        booster = "lightgbm-gpu"
    elif info.backends.get("lightgbm"):
        booster = "lightgbm-cpu"
    else:
        booster = "sklearn-hist (CPU)"
    print(f"stylometry backend : {booster}")
    return 0


def cmd_version(args) -> int:
    print(f"ntl-pact18 {__version__}")
    return 0


# ------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ntl-pact18",
        description="PACT-18: prompt-aware constrained attribution for 18 LLMs",
    )
    parser.add_argument("--config", default=None, help="YAML config path or name in configs/")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="config override, repeatable: --set prism.lam=1.5 --set clustering.method=tfidf")
    parser.add_argument("--profile", choices=["auto", "local", "kaggle", "off"], default="auto",
                        help="host profile: auto-detect (default), force local/kaggle, or off")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="generate the synthetic corpus")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("split", help="prompt-grouped train/val/test split of one labelled file")
    p.add_argument("--source", required=True, help="labelled CSV with id/prompt_id/text/model")
    p.add_argument("--out", default=None, help="output directory (default: paths.data_dir)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("train", help="fit the pipeline and save artifacts")
    p.add_argument("--train-file", default=None)
    p.add_argument("--model-out", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="predict a test file and write a submission")
    p.add_argument("--model", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--test-file", default=None)
    p.add_argument("--truth-file", default=None)
    p.add_argument("--no-eval", action="store_true")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("simulate", help="structure ablation on pseudo test pools")
    p.add_argument("--model", default=None)
    p.add_argument("--k", nargs="*", type=int, default=[4, 8, 12, 18])
    p.add_argument("--split-rate", type=float, default=0.0)
    p.add_argument("--merge-rate", type=float, default=0.0)
    p.add_argument("--json", default=None)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("smoke", help="end-to-end self test on synthetic data")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("device", help="show the detected compute device and backends")
    p.set_defaults(func=cmd_device)

    p = sub.add_parser("version", help="print version")
    p.set_defaults(func=cmd_version)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
