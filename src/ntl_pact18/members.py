"""Member registry: every member exposes ``fit`` / ``predict_proba`` on 18 columns."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .features import NgramLMMember, StylegramMember, StylometryMember
from .features.torch_linear import TorchLinearMember
from .labels import LABELS


class LinearBaselineMember:
    """The official starter recipe: word + char TF-IDF -> logistic regression."""

    name = "linear_baseline"

    def __init__(self, C: float = 2.0):
        self.C = float(C)
        self.pipeline: Optional[Pipeline] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, texts: Sequence[str], y: Sequence[str],
            groups: Optional[Sequence[int]] = None) -> "LinearBaselineMember":
        self.pipeline = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=25_000, sublinear_tf=True)),
                            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=35_000, sublinear_tf=True)),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(C=self.C, max_iter=1000)),
            ]
        )
        self.pipeline.fit(list(texts), list(y))
        self.classes_ = np.asarray(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict_proba(self, texts: Sequence[str],
                      groups: Optional[Sequence[int]] = None) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("LinearBaselineMember.predict_proba called before fit")
        raw = self.pipeline.predict_proba(list(texts))
        out = np.full((len(texts), len(LABELS)), 1e-6)
        for col, label in enumerate(self.classes_):
            out[:, LABELS.index(str(label))] = raw[:, col]
        return out / out.sum(axis=1, keepdims=True)


def build_members(cfg, device_info=None) -> List[object]:
    """Instantiate every enabled member from the config."""
    members: List[object] = []

    sg = cfg.section("features.stylegram")
    if sg.get("enabled", True):
        spec = {
            key: dict(
                ngram=tuple(sg.get(key, {}).get("ngram", (3, 5))),
                max_features=int(sg.get(key, {}).get("max_features", 50_000)),
                weight=float(sg.get(key, {}).get("weight", 1.0)),
            )
            for key in ("char", "word", "shape", "function")
        }
        members.append(
            StylegramMember(
                spec=spec,
                C=float(sg.get("C", 0.5)),
                center_mode=str(sg.get("center_mode", "replace")),
                min_size=int(cfg.get("clustering.min_size", 2)),
                max_size=int(cfg.get("clustering.max_size", 18)),
            )
        )

    st = cfg.section("features.stylometry")
    if st.get("enabled", True):
        members.append(
            StylometryMember(
                n_estimators=int(st.get("n_estimators", 300)),
                learning_rate=float(st.get("learning_rate", 0.05)),
                max_leaf_nodes=int(st.get("max_leaf_nodes", 15)),
                backend=str(st.get("backend", "auto")),
                use_gpu=bool(device_info is not None and device_info.is_accelerator),
            )
        )

    qwen_member = build_qwen_member(cfg, device_info=device_info)
    if qwen_member is not None:
        members.append(qwen_member)

    tl = cfg.section("features.torch_linear")
    raw_enabled = tl.get("enabled", "auto")
    if isinstance(raw_enabled, str):
        want_torch = raw_enabled.lower() in ("auto", "true", "yes", "1")
        # 'auto' means: only when an accelerator is actually available
        if raw_enabled.lower() == "auto":
            want_torch = bool(device_info is not None and device_info.is_accelerator)
    else:
        want_torch = bool(raw_enabled)
    if want_torch and device_info is not None and device_info.torch_available:
        members.append(
            TorchLinearMember(
                device_info,
                epochs=int(tl.get("epochs", 300)),
                lr=float(tl.get("lr", 0.05)),
                weight_decay=float(tl.get("weight_decay", 1e-4)),
                seed=int(cfg.get("project.seed", 42)),
                char_max_features=int(tl.get("char_max_features", 60_000)),
                word_max_features=int(tl.get("word_max_features", 40_000)),
                shape_max_features=int(tl.get("shape_max_features", 15_000)),
                function_max_features=int(tl.get("function_max_features", 12_000)),
            )
        )

    lm = cfg.section("features.ngram_lm")
    if lm.get("enabled", True):
        members.append(NgramLMMember(order=int(lm.get("order", 2)), alpha=float(lm.get("alpha", 0.1))))

    lb = cfg.section("features.linear_baseline")
    if lb.get("enabled", True):
        members.append(LinearBaselineMember(C=float(lb.get("C", 2.0))))

    # heavy GPU backbones first: they precompute once, then release VRAM
    members.sort(key=lambda m: 0 if m.name.startswith("qwen") else 1)
    return members


def two_stage_from_config(qwen_cfg) -> bool:
    """``mode: two_stage`` implies the two-stage trainer unless overridden.

    Single source of truth for the member constructor and the parallel
    trainer: if they ever disagree, the fold loop starts before
    ``fit_stage1()`` has run and every fold raises. (That exact bug shipped
    once - the trainer read the explicit key, the member read the mode.)
    """
    qwen_cfg = qwen_cfg or {}
    return bool(qwen_cfg.get("two_stage", str(qwen_cfg.get("mode", "")) == "two_stage"))


def build_qwen_member(cfg, device_info=None):
    """Instantiate the Qwen member from config, or ``None`` when disabled.

    Kept separate from :func:`build_members` because the parallel trainer builds
    this member on its own: the two GPU ranks must share exactly one definition
    of the adapter/hyper-parameters, and the classical members are built in a
    different process (CPU only) that must not import torch.
    """
    if device_info is None:
        return None
    q = cfg.section("features.qwen")
    raw_qwen = q.get("enabled", "auto")
    if isinstance(raw_qwen, str):
        want_qwen = raw_qwen.lower() in ("auto", "true", "yes", "1") and (
            device_info.kind == "cuda"
        )
    else:
        want_qwen = bool(raw_qwen)
    if not want_qwen:
        return None
    try:
        from .features.qwen_member import QwenMember

        return QwenMember(
            device_info,
            mode=str(q.get("mode", "probe")),
            # ``mode: two_stage`` implies the two-stage trainer unless the
            # explicit flag overrides it (shared with the parallel trainer)
            two_stage=two_stage_from_config(q),
            model_id=str(q.get("model_id", "Qwen/Qwen3-8B-Base")),
            max_len=int(q.get("max_len", 384)),
            batch_size=int(q.get("batch_size", 8)),
            layers=tuple(q.get("layers", [18])),
            C=float(q.get("C", 0.0025)),
            epochs=int(q.get("epochs", 1)),
            lr=float(q.get("lr", 1e-4)),
            lora_r=int(q.get("lora_r", 16)),
            lora_alpha=int(q.get("lora_alpha", 32)),
            lora_dropout=float(q.get("lora_dropout", 0.05)),
            grad_checkpoint=bool(q.get("grad_checkpoint", True)),
            load_in_4bit=bool(q.get("load_in_4bit", True)),
            supcon_weight=float(q.get("supcon_weight", 0.1)),
            label_smoothing=float(q.get("label_smoothing", 0.05)),
            grad_accum=int(q.get("grad_accum", 8)),
            warmup_ratio=float(q.get("warmup_ratio", 0.03)),
            weight_decay=float(q.get("weight_decay", 0.01)),
            time_budget_hours=float(q.get("time_budget_hours", 0.0)),
            log_every=int(q.get("log_every", 20)),
            target_modules=q.get("target_modules") or None,
            val_fraction=float(q.get("val_fraction", 0.0)),
            early_stop_patience=int(q.get("early_stop_patience", 3)),
            eval_every=int(q.get("eval_every", 100)),
            distributed=bool(q.get("distributed", True)),
            skip_lm_head=bool(q.get("skip_lm_head", True)),
            length_bucket=bool(q.get("length_bucket", True)),
            bucket_pool=int(q.get("bucket_pool", 40)),
            reset_before_fit=bool(q.get("reset_before_fit", True)),
            compute_dtype=str(q.get("compute_dtype", "auto")),
            head_proj=int(q.get("head_proj", 512)),
            head_dropout=float(q.get("head_dropout", 0.1)),
            truncate_layers=int(q.get("truncate_layers", 0)),
            pool_mode=str(q.get("pool_mode", "attention")),
            stage1_epochs=int(q.get("stage1_epochs", 1)),
            stage1_lr=float(q.get("stage1_lr", 1e-4)),
            stage2_epochs=int(q.get("stage2_epochs", 2)),
            stage2_lr=float(q.get("stage2_lr", 5e-5)),
            head_lr=float(q.get("head_lr", 1e-3)),
            stage2_head_lr=float(q.get("stage2_head_lr", 5e-4)),
            triplet_batch=int(q.get("triplet_batch", 3)),
            centred_weight=float(q.get("centred_weight", 1.0)),
            stage1_time_budget_hours=float(q.get("stage1_time_budget_hours", 0.0)),
            stage1_max_rows=int(q.get("stage1_max_rows", 0)),
            stage1_class_weighted=bool(q.get("stage1_class_weighted", False)),
            seed=int(cfg.get("project.seed", 42)),
        )
    except Exception as exc:
        print(f"[members] Qwen member disabled: {exc}")
        return None


def fit_members(members: Sequence[object], texts: Sequence[str], y: Sequence[str],
                groups: Optional[Sequence[int]] = None, verbose: bool = True) -> None:
    for member in members:
        if verbose:
            print(f"[member] fitting {member.name}")
        member.fit(texts, y, groups)
        if getattr(member, "release_after_fit", False) and hasattr(member, "release"):
            # hand the GPU back so the next member gets the whole VRAM
            member.release()


def precompute_members(members: Sequence[object], texts: Sequence[str],
                       verbose: bool = True) -> None:
    """Run any GPU backbone once over the full corpus, then free it."""
    for member in members:
        if not hasattr(member, "precompute"):
            continue
        if verbose:
            print(f"[member] precomputing {member.name} features on "
                  f"{getattr(member, 'device', 'device')}")
        member.precompute(texts)
        if getattr(member, "release_after_fit", False) and hasattr(member, "release"):
            member.release()


def predict_members(members: Sequence[object], texts: Sequence[str],
                    groups: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
    return {member.name: member.predict_proba(texts, groups) for member in members}
