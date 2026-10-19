#!/usr/bin/env python
"""Train PACT-18 on two T4 cards with ~50k rows.

Topology
--------
Two worker processes, one card each, **data-parallel** inside every QLoRA fit:

    rank 0: 4-bit copy of the backbone  <- batch slice 0 --+
                                                           | all-reduce grads
    rank 1: 4-bit copy of the backbone  <- batch slice 1 --+

Both ranks see the same deterministic batch plan (``QwenMember._make_batches``)
and take a different slice of every global batch, so one optimizer step covers
16-32 rows instead of 8. Gradients are summed by hand
(:mod:`ntl_pact18.dist_utils`) rather than through ``DistributedDataParallel``:
the trainable tensors are LoRA adapters on bitsandbytes 4-bit modules, which is
where DDP's bucket assumptions tend to break. With ``world_size == 1`` every
collective is a no-op, so the laptop path is unchanged.

After the GPU stage one CPU process fits the classical members and the stacking
layer (temperature / blend / family experts / shield / PRISM). Those are 5-20x
cheaper than the QLoRA pass and would only contend for the 4 vCPUs, so they are
run sequentially rather than overlapped.

Stages
------
``--stage qwen``        both ranks: 3 folds + refit, writes qwen OOF + member
``--stage classical``   one process: 3 folds + refit for the CPU members
``--stage assemble``    post-processing, model + submission
``--stage all``         launcher: spawns the two ranks, then the two CPU stages

Example
-------
    python scripts/build_external_mix.py --target-total 50000 \\
        --external data/external/ntl_merged_external_large.parquet \\
        --out data/real/trainval_plus_ext_50k.csv
    python scripts/train_parallel.py --stage all \\
        --config configs/kaggle_9h_50k_2gpu.yaml \\
        --train-file data/real/trainval_plus_ext_50k.csv \\
        --model-path /kaggle/input/qwen3-1-7b/transformers/default/1 \\
        --test-file /kaggle/input/name-that-llm/test.csv \\
        --submission submission.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ntl_pact18.pipeline import PACT18Pipeline  # noqa: E402

SCRIPT = Path(__file__).resolve()


# ----------------------------------------------------------------- helpers
def _cfg(args):
    overrides = list(args.set or [])
    if args.model_path:
        overrides.append(f"features.qwen.model_id={args.model_path}")
    from ntl_pact18.config import load_config

    return load_config(args.config, overrides=overrides, profile=args.profile)


def _prepare(args, cfg):
    from ntl_pact18.data import add_group_folds, load_train

    df = load_train(args.train_file)
    df = add_group_folds(
        df, int(cfg.get("data.n_folds", 5)), int(cfg.get("project.seed", 42))
    )
    return df


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _log_mem(rank: int, tag: str) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        free, total = torch.cuda.mem_get_info()
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(
            f"[mem] rank{rank} {tag}: peak={peak:.2f} GiB / {total / 2**30:.1f} GiB "
            f"(free {free / 2**30:.1f})",
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass



def _stratified_fit_rows(df, limit: int, seed: int) -> np.ndarray:
    """Rows the CPU members actually train on.

    The Qwen member needs every row; the classical members do not. Their OOF is
    only used to fit the blend weights *on the competition rows*, so training
    TF-IDF/LinearSVC/XGBoost on 50k passages buys almost nothing and costs
    ~1 h of the 9 h budget. Keep every competition row, fill the rest per class
    from the external pool.
    """
    if limit <= 0 or limit >= len(df):
        return np.arange(len(df))
    per_class = max(1, limit // max(1, df["model"].nunique()))
    keep: list[int] = []
    for _label, group in df.groupby("model", sort=False):
        if "origin" in group.columns:
            comp = group[group["origin"] == "competition"]
            ext = group[group["origin"] != "competition"]
        else:
            comp, ext = group.iloc[:0], group
        taken = list(comp.index)
        need = max(0, per_class - len(taken))
        if need and len(ext):
            taken += list(ext.sample(min(need, len(ext)), random_state=seed).index)
        keep += taken
    return np.sort(np.asarray(keep, dtype=int))


# -------------------------------------------------------------- stage: qwen
def cmd_qwen(args) -> int:
    from ntl_pact18.data import add_group_folds
    from ntl_pact18.device import detect_device
    from ntl_pact18.dist_utils import barrier, init_distributed, shutdown
    from ntl_pact18.labels import LABELS
    from ntl_pact18.members import build_qwen_member
    from ntl_pact18.utils import encode_groups

    state = init_distributed()
    rank, world = state.rank, state.world
    cfg = _cfg(args)
    df = _prepare(args, cfg)
    texts = df["text"].tolist()
    labels = df["model"].tolist()
    codes = encode_groups(df["prompt_id"].tolist())

    # Same implication rule the member constructor uses (shared helper), so
    # the trainer can never disagree with the member about the mode.
    from ntl_pact18.members import two_stage_from_config

    two_stage = two_stage_from_config(cfg.get("features.qwen", {}) or {})
    if two_stage:
        # Stage 1 sees the external pool only; stage 2 and every OOF fold see
        # host rows only. Splitting them is the whole point: mixing the pool
        # back into stage 2 is reported to make the result worse.
        if "origin" not in df.columns:
            print("[qwen] two_stage needs an 'origin' column "
                  "(build the training file with scripts/build_external_mix.py)",
                  file=sys.stderr)
            return 2
        host_mask = (df["origin"] == "competition").to_numpy()
        # oversampled tail classes reuse a passage; stage 1 should see each
        # passage once, otherwise a handful of classes get extra weight
        pool_mask = (df["origin"] == "external").to_numpy()
        host_df = df.loc[host_mask].reset_index(drop=True)
        host_df = add_group_folds(host_df, int(df.attrs.get("n_folds", 5)),
                                  int(cfg.get("project.seed", 42)))
        host_idx = np.flatnonzero(host_mask)
        fold_ids = host_df["fold"].to_numpy()
        n_folds = int(host_df.attrs["n_folds"])
        if rank == 0:
            print(f"[qwen] two-stage: pool={int(pool_mask.sum())} rows / "
                  f"{df.loc[pool_mask, 'prompt_id'].nunique()} prompts | "
                  f"host={len(host_df)} rows / {host_df['prompt_id'].nunique()} prompts "
                  f"| folds={n_folds}", flush=True)
    else:
        host_idx = np.arange(len(df))
        fold_ids = df["fold"].to_numpy()
        n_folds = int(df.attrs["n_folds"])

    device_info = detect_device(cfg, verbose=(rank == 0))
    qwen = build_qwen_member(cfg, device_info=device_info)
    if qwen is None:
        print("[qwen] member disabled in config", file=sys.stderr)
        return 2
    qwen.verbose = rank == 0
    if rank == 0:
        print(f"[qwen] rows={len(df)} prompts={df['prompt_id'].nunique()} "
              f"classes={df['model'].nunique()} world={world} "
              f"mode={qwen.mode} truncate={qwen.truncate_layers or 'off'}", flush=True)

    cfg_budget = float(cfg.get("features.qwen.time_budget_hours", 0.0))
    hard_deadline = (time.time() + args.deadline_hours * 3600.0
                     if args.deadline_hours > 0 else None)
    oof = np.full((len(df), len(LABELS)), 1.0 / len(LABELS), dtype=np.float32)
    timings = []
    t_start = time.time()

    if two_stage:
        pool_rows = np.flatnonzero(pool_mask)
        t0 = time.time()
        qwen.fit_stage1([texts[i] for i in pool_rows],
                        [labels[i] for i in pool_rows],
                        groups=codes[pool_rows])
        timings.append(time.time() - t0)
        _log_mem(rank, "stage1")

    for fold in range(n_folds):
        tr_local = np.flatnonzero(fold_ids != fold)
        va_local = np.flatnonzero(fold_ids == fold)
        tr = host_idx[tr_local]
        va = host_idx[va_local]
        if rank == 0:
            print(f"[qwen] fold {fold + 1}/{n_folds}: train={len(tr)} val={len(va)}",
                  flush=True)
        if hard_deadline is not None:
            # leave room for the classical stage, the assemble step and the
            # container's own shutdown; truncating a fit is far better than
            # having Kaggle kill the kernel at 9 h with no submission
            left = (hard_deadline - time.time() - args.reserve_hours * 3600.0) / 3600.0
            qwen.time_budget_hours = max(0.05, min(cfg_budget, left))
            if rank == 0:
                print(f"[qwen] wall-clock guard: {left:.2f} h left, "
                      f"fit budget {qwen.time_budget_hours:.2f} h", flush=True)
        t0 = time.time()
        qwen.fit([texts[i] for i in tr], [labels[i] for i in tr], groups=codes[tr])
        probs = qwen.predict_proba([texts[i] for i in va])
        oof[va] = probs
        timings.append(time.time() - t0)
        _log_mem(rank, f"fold{fold + 1}")
        if not two_stage:
            # hand the card back before the next fold reloads; two-stage keeps
            # the trunk resident so each fold can restart from the stage-1
            # snapshot instead of paying a 30 s reload
            qwen.release()

    if rank == 0:
        print("[qwen] refit on the full host split", flush=True)
    t0 = time.time()
    qwen.fit([texts[i] for i in host_idx], [labels[i] for i in host_idx],
             groups=codes[host_idx])
    timings.append(time.time() - t0)
    _log_mem(rank, "refit")

    if rank == 0:
        Path(args.qwen_oof).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.qwen_oof, oof)
        qwen.release()   # capture the final adapter, drop the 4-bit weights
        joblib.dump(qwen, args.qwen_model)
        total = time.time() - t_start
        print(f"[qwen] total {total / 3600:.2f} h "
              f"({[round(t / 60, 1) for t in timings]} min)", flush=True)
        print(f"[qwen] oof -> {args.qwen_oof} | member -> {args.qwen_model}", flush=True)
        meta = {
            "rows": int(len(df)),
            "host_rows": int(len(host_idx)),
            "folds": int(n_folds),
            "world": int(world),
            "two_stage": bool(two_stage),
            "seconds_per_fit": [round(t, 1) for t in timings],
            "trainable_parameters": int(qwen.trainable_parameters()),
            "model_id": str(getattr(qwen, "model_id", "")),
            "max_len": int(getattr(qwen, "max_len", 0)),
            "lora_r": int(getattr(qwen, "lora_r", 0)),
            "truncate_layers": int(getattr(qwen, "truncate_layers", 0)),
            "pool_mode": str(getattr(qwen, "pool_mode", "")),
            "layers": list(getattr(qwen, "layers", [])),
        }
        (Path(args.qwen_oof).with_suffix(".json")).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    barrier()
    shutdown()
    return 0


# --------------------------------------------------------- stage: classical
def cmd_classical(args) -> int:
    from ntl_pact18.labels import LABELS
    from ntl_pact18.members import build_members, fit_members, predict_members
    from ntl_pact18.utils import encode_groups

    cfg = _cfg(args)
    df = _prepare(args, cfg)
    texts = df["text"].tolist()
    labels = df["model"].tolist()
    codes = encode_groups(df["prompt_id"].tolist())
    fold_ids = df["fold"].to_numpy()

    # device_info=None keeps XGBoost on CPU and skips the torch member: this
    # process must not touch the cards the QLoRA ranks are using
    members = [m for m in build_members(cfg, device_info=None)
               if not m.name.startswith("qwen")]
    print(f"[classical] members: {[m.name for m in members]} rows={len(df)}", flush=True)

    fit_idx = _stratified_fit_rows(
        df, int(cfg.get("classical.max_fit_rows", 0)), int(cfg.get("project.seed", 42))
    )
    fit_mask = np.zeros(len(df), dtype=bool)
    fit_mask[fit_idx] = True
    print(f"[classical] fitting on {len(fit_idx)}/{len(df)} rows "
          f"(competition rows always kept)", flush=True)

    oof = {m.name: np.full((len(df), len(LABELS)), 1.0 / len(LABELS)) for m in members}
    t0 = time.time()
    for fold in range(int(df.attrs["n_folds"])):
        tr = np.flatnonzero((fold_ids != fold) & fit_mask)
        va = np.flatnonzero(fold_ids == fold)
        print(f"[classical] fold {fold + 1}: train={len(tr)} val={len(va)}", flush=True)
        fit_members(members, [texts[i] for i in tr], [labels[i] for i in tr],
                    groups=codes[tr], verbose=False)
        probs = predict_members(members, [texts[i] for i in va], groups=None)
        for name, p in probs.items():
            oof[name][va] = p

    print("[classical] refit on the fitted rows", flush=True)
    fit_members(members, [texts[i] for i in fit_idx], [labels[i] for i in fit_idx],
                groups=codes[fit_idx], verbose=True)

    Path(args.classical_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"members": members, "oof": oof}, args.classical_model)
    print(f"[classical] done in {(time.time() - t0) / 60:.1f} min -> "
          f"{args.classical_model}", flush=True)
    return 0


# ---------------------------------------------------------- stage: assemble
def cmd_assemble(args) -> int:
    from ntl_pact18.data import load_test
    from ntl_pact18.labels import LABELS

    cfg = _cfg(args)
    df = _prepare(args, cfg)

    bundle = joblib.load(args.classical_model)
    qwen = joblib.load(args.qwen_model)
    qwen_oof = np.load(args.qwen_oof)

    members = list(bundle["members"]) + [qwen]
    oof = dict(bundle["oof"])
    oof[qwen.name] = qwen_oof

    mask = None
    if "origin" in df.columns:
        mask = (df["origin"] == "competition").to_numpy()
        print(f"[assemble] stacking layer fitted on {int(mask.sum())} competition rows "
              f"out of {len(df)}", flush=True)

    # per-member OOF on the competition rows: without this the only visible
    # number is the blend, and a degraded member is indistinguishable from a
    # degraded blend (this is what made the 0.3867 run hard to attribute)
    from ntl_pact18.evaluate import balanced_accuracy
    from ntl_pact18.labels import LABEL_TO_INDEX

    y_idx = np.asarray([LABEL_TO_INDEX[v] for v in df["model"]], dtype=int)
    rows = mask if mask is not None else np.ones(len(df), dtype=bool)
    print("[assemble] per-member OOF balanced accuracy on the eval rows:", flush=True)
    for name, arr in oof.items():
        ba = balanced_accuracy(y_idx[rows], arr[rows].argmax(axis=1))
        print(f"  {name:16s} {ba:.4f}", flush=True)

    pipe = PACT18Pipeline(cfg).fit(
        df, members=members, oof=oof, refit=False, eval_mask=mask
    )
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    pipe.save(args.model_out)
    print(f"[assemble] model -> {args.model_out}", flush=True)

    if not args.test_file:
        return 0
    test = load_test(args.test_file)
    pred = pipe.predict(test, verbose=True)
    sub = pd.DataFrame({"id": test["id"].astype(str), "model": pred.labels()})

    bad = sorted(set(sub["model"]) - set(LABELS))
    assert not bad, f"labels outside the 18 classes: {bad}"
    assert len(sub) == len(test) and sub["id"].is_unique
    Path(args.submission).parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(args.submission, index=False)
    print(f"[assemble] submission -> {args.submission} ({len(sub)} rows, "
          f"{sub['model'].nunique()} labels used)", flush=True)
    return 0


# ---------------------------------------------------------------- launchers
def _launch_ranks(args) -> int:
    try:
        import torch

        n_gpu = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
        print(f"[launch] visible GPUs: {n_gpu} {names}", flush=True)
        if n_gpu < args.nproc:
            print(f"[launch] {args.nproc} ranks requested but only {n_gpu} GPUs are "
                  f"visible - pick the 'GPU T4 x2' accelerator", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"[launch] GPU probe failed: {exc}", file=sys.stderr)
        return 1
    port = _free_port()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    procs, handles = [], []
    for r in range(args.nproc):
        env = os.environ.copy()
        env.update(
            {
                "RANK": str(r),
                # every child sees exactly one card, so its device index is 0
                "LOCAL_RANK": "0",
                "WORLD_SIZE": str(args.nproc),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "CUDA_VISIBLE_DEVICES": str(r),
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "OMP_NUM_THREADS": "2",
                "TOKENIZERS_PARALLELISM": "false",
                # container PCIe peer access is often denied; all-reduce is only
                # ~1-2% of a step, so trading P2P for a reliable bring-up is a
                # clear win on Kaggle
                "NCCL_P2P_DISABLE": "1",
                "NCCL_IB_DISABLE": "1",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            }
        )
        cmd = [
            sys.executable, "-u", str(SCRIPT),
            "--stage", "qwen",
            "--config", str(args.config),
            "--train-file", str(args.train_file),
            "--qwen-oof", str(args.qwen_oof),
            "--qwen-model", str(args.qwen_model),
            "--profile", str(args.profile),
        ]
        if args.model_path:
            cmd += ["--model-path", str(args.model_path)]
        for item in args.set or []:
            cmd += ["--set", item]
        log = open(log_dir / f"qwen_rank{r}.log", "w", encoding="utf-8")
        handles.append(log)
        print(f"[launch] rank {r} -> {log.name}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT))

    codes = [p.wait() for p in procs]
    for h in handles:
        h.close()
    if any(codes):
        print(f"[launch] ranks failed: {codes} (see {log_dir})", file=sys.stderr)
        return 1
    return 0


def cmd_all(args) -> int:
    t0 = time.time()
    if args.nproc > 1:
        if _launch_ranks(args):
            return 1
    else:
        if cmd_qwen(args):
            return 1
    if not args.skip_classical:
        if cmd_classical(args):
            return 1
    if not args.skip_assemble:
        if cmd_assemble(args):
            return 1
    print(f"[all] finished in {(time.time() - t0) / 3600:.2f} h", flush=True)
    return 0


# --------------------------------------------------------------------- cli
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "qwen", "classical", "assemble"])
    ap.add_argument("--config", default="configs/kaggle_9h_50k_2gpu.yaml")
    ap.add_argument("--profile", default="auto")
    ap.add_argument("--train-file", default="data/real/trainval_plus_ext_50k.csv")
    ap.add_argument("--test-file", default="")
    ap.add_argument("--model-path", default="", help="local dir of the Qwen checkpoint")
    ap.add_argument("--qwen-oof", default="artifacts/parallel/qwen_oof.npy")
    ap.add_argument("--qwen-model", default="artifacts/parallel/qwen_member.joblib")
    ap.add_argument("--classical-model", default="artifacts/parallel/classical.joblib")
    ap.add_argument("--model-out", default="artifacts/parallel/model.joblib")
    ap.add_argument("--submission", default="submission.parquet")
    ap.add_argument("--log-dir", default="artifacts/parallel/logs")
    ap.add_argument("--nproc", type=int, default=0, help="0 = read parallel.gpus")
    ap.add_argument("--reserve-hours", type=float, default=0.55,
                    help="time kept back for the CPU stage + assemble")
    ap.add_argument("--deadline-hours", type=float, default=0.0,
                    help="wall-clock budget for the whole GPU stage; each fit's "
                         "internal budget is clamped to what is left")
    ap.add_argument("--skip-classical", action="store_true")
    ap.add_argument("--skip-assemble", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.nproc <= 0:
        cfg = _cfg(args)
        args.nproc = int(cfg.get("parallel.gpus", 1))
    handlers = {
        "qwen": cmd_qwen,
        "classical": cmd_classical,
        "assemble": cmd_assemble,
        "all": cmd_all,
    }
    return handlers[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
