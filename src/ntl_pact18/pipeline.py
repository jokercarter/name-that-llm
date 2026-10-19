"""End-to-end PACT-18 pipeline: fit, predict, persist."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from .blend import blend_probs, fit_blend_weights
from .calibration import apply_temperature, fit_temperature, sld_em
from .clustering import build_clusterer, cluster_confidence
from .data import add_group_folds
from .device import detect_device, enforce_policy
from .evaluate import summarize
from .experts import FamilyExperts
from .labels import LABELS, LABEL_TO_INDEX
from .members import build_members, fit_members, precompute_members, predict_members
from .prism import decode
from .shield import OrthographicShield
from .utils import encode_groups, ensure_dir, seed_everything


def _encode_groups(values: Sequence) -> np.ndarray:
    return encode_groups(values)


@dataclass
class Prediction:
    probs: np.ndarray
    preds: np.ndarray
    groups: np.ndarray
    stage_preds: Dict[str, np.ndarray] = field(default_factory=dict)
    stats: Dict[str, float] = field(default_factory=dict)

    def labels(self) -> List[str]:
        return [LABELS[int(i)] for i in self.preds]


class PACT18Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.seed = int(cfg.get("project.seed", 42))
        self.members: List[object] = []
        self.temperatures: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}
        self.experts: Optional[FamilyExperts] = None
        self.shield: Optional[OrthographicShield] = None
        self.clusterer = None
        self.class_prior_: Optional[np.ndarray] = None
        self.oof_probs_: Dict[str, np.ndarray] = {}
        self.train_summary_: Dict[str, float] = {}
        self.device_info = None
        self.fitted_ = False

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        train_df: pd.DataFrame,
        verbose: bool = True,
        *,
        members: Optional[Sequence[object]] = None,
        oof: Optional[Dict[str, np.ndarray]] = None,
        refit: bool = True,
        eval_mask: Optional[Sequence[bool] | np.ndarray] = None,
    ) -> "PACT18Pipeline":
        """Fit the whole stack.

        The four keyword arguments exist for the two-GPU trainer, which computes
        the expensive member pass in separate processes and then only needs the
        calibration/blend/expert stage here. Defaults reproduce the single
        process behaviour exactly.

        ``eval_mask`` restricts every *fitted* post-processing parameter
        (temperature, blend weights, family experts, shield, reported metrics) to
        a subset of rows. The 50k training file is ~96% external passages; the
        test set is 100% competition passages, so fitting the blend on all rows
        would optimise for the wrong distribution. ``None`` keeps all rows.
        """
        seed_everything(self.seed)
        t0 = time.perf_counter()
        self.device_info = detect_device(self.cfg, verbose=verbose)
        enforce_policy(self.device_info, self.cfg)
        if verbose:
            print("[device] " + self.device_info.describe().replace("\n", "\n[device] "))
        df = add_group_folds(train_df, int(self.cfg.get("data.n_folds", 5)), self.seed)
        texts = df["text"].tolist()
        y_labels = df["model"].tolist()
        y_idx = np.asarray([LABEL_TO_INDEX[v] for v in y_labels], dtype=int)
        prompt_codes = _encode_groups(df["prompt_id"].tolist())

        n_folds = int(df.attrs.get("n_folds", self.cfg.get("data.n_folds", 5)))
        self.class_prior_ = np.bincount(y_idx, minlength=len(LABELS)).astype(float)
        self.class_prior_ /= self.class_prior_.sum()

        self.members = list(members) if members is not None else build_members(
            self.cfg, device_info=self.device_info
        )
        if not self.members:
            raise RuntimeError("no members enabled in config")

        mask = (np.ones(len(df), dtype=bool) if eval_mask is None
                else np.asarray(eval_mask, dtype=bool))
        if mask.shape[0] != len(df):
            raise ValueError(f"eval_mask has {mask.shape[0]} rows, expected {len(df)}")
        if not mask.any():
            raise ValueError("eval_mask selects no rows")

        if oof is None:
            # GPU backbones (Qwen) extract once over the whole corpus, then free
            # VRAM so the remaining GPU members train without contention.
            precompute_members(self.members, texts, verbose=verbose)

            # ---------------------------------------------------------- OOF pass
            oof = {m.name: np.full((len(df), len(LABELS)), 1.0 / len(LABELS))
                   for m in self.members}
            for fold in range(n_folds):
                tr = np.flatnonzero(df["fold"].to_numpy() != fold)
                va = np.flatnonzero(df["fold"].to_numpy() == fold)
                if not len(va):
                    continue
                if verbose:
                    print(f"[oof] fold {fold + 1}/{n_folds}: train={len(tr)} val={len(va)}")
                fit_members(self.members, [texts[i] for i in tr], [y_labels[i] for i in tr],
                            groups=prompt_codes[tr], verbose=False)
                fold_probs = predict_members(self.members, [texts[i] for i in va], groups=None)
                for name, p in fold_probs.items():
                    oof[name][va] = p
        else:
            missing = [m.name for m in self.members if m.name not in oof]
            if missing:
                raise ValueError(f"oof is missing members: {missing}")
            oof = {m.name: np.asarray(oof[m.name], dtype=float) for m in self.members}
            for name, arr in oof.items():
                if arr.shape != (len(df), len(LABELS)):
                    raise ValueError(
                        f"oof[{name}] has shape {arr.shape}, expected {(len(df), len(LABELS))}"
                    )
        self.oof_probs_ = oof

        # ------------------------------------------------------ calibration
        if bool(self.cfg.get("calibration.temperature", True)):
            for name, p in oof.items():
                self.temperatures[name] = fit_temperature(p[mask], y_idx[mask])
        else:
            self.temperatures = {name: 1.0 for name in oof}
        calibrated = {n: apply_temperature(p, self.temperatures[n]) for n, p in oof.items()}

        # ------------------------------------------------------------- blend
        self.weights = fit_blend_weights(
            {n: p[mask] for n, p in calibrated.items()}, y_idx[mask],
            l2=float(self.cfg.get("blend.l2", 1.0)),
            max_iter=int(self.cfg.get("blend.max_iter", 300)),
        )
        blended_oof = blend_probs(calibrated, self.weights)

        # ----------------------------------------------------------- experts
        if bool(self.cfg.get("experts.enabled", True)):
            self.experts = FamilyExperts(
                min_rows=int(self.cfg.get("experts.min_rows", 60)),
                adaptive_rho=bool(self.cfg.get("experts.adaptive_rho", True)),
            ).fit(blended_oof[mask], y_idx[mask])

        # ------------------------------------------------------------ shield
        if bool(self.cfg.get("shield.enabled", True)):
            train_texts = [texts[i] for i in np.flatnonzero(mask)]
            self.shield = OrthographicShield(
                penalty=float(self.cfg.get("shield.penalty", 2.5)),
                max_rate=float(self.cfg.get("shield.max_rate", 0.01)),
                min_support=int(self.cfg.get("shield.min_support", 40)),
            ).fit(train_texts, y_idx[mask])

        # --------------------------------------------------------- clustering
        # unsupervised and transductive by design: clustering sees all training
        # text, because at prediction time it must cluster the test set too
        self.clusterer = build_clusterer(self.cfg)
        self.clusterer.fit(texts, prompt_codes)

        # ------------------------------------------------ refit on full train
        if refit:
            if verbose:
                print("[fit] refitting members on the full training split")
            fit_members(self.members, texts, y_labels, groups=prompt_codes, verbose=verbose)

        # ------------------------------------------------------- self-check
        self.train_summary_ = self._train_self_check(
            df, blended_oof[mask], y_idx[mask], mask=mask
        )
        self.fitted_ = True
        if verbose:
            print(
                f"[fit] done in {time.perf_counter() - t0:.1f}s | "
                f"oof blended balanced acc = {self.train_summary_['oof_blend_balanced_accuracy']:.4f}"
                f" (evaluated on {int(mask.sum())} rows)"
            )
            print("[fit] weights: " + ", ".join(f"{k}={v:.3f}" for k, v in self.weights.items()))
            print("[fit] temperatures: " + ", ".join(f"{k}={v:.2f}" for k, v in self.temperatures.items()))
            if self.shield is not None and self.shield.rules:
                print("[fit] shield rules: " + " | ".join(self.shield.describe()))
        return self

    def _train_self_check(self, df: pd.DataFrame, blended_oof: np.ndarray,
                          y_idx: np.ndarray, mask: Optional[np.ndarray] = None
                          ) -> Dict[str, float]:
        pred = blended_oof.argmax(axis=1)
        summary = summarize(blended_oof, y_idx, pred)
        summary["n_eval_rows"] = float(len(y_idx))
        summary["oof_blend_balanced_accuracy"] = summary.pop("balanced_accuracy")
        summary["accuracy"] = float((pred == y_idx).mean())
        return summary

    # -------------------------------------------------------------- predict
    def blended_oof(self) -> np.ndarray:
        """Temperature-calibrated, weight-blended out-of-fold probabilities."""
        if not self.oof_probs_:
            raise RuntimeError("blended_oof() requires fit() first")
        calibrated = {
            name: apply_temperature(p, self.temperatures.get(name, 1.0))
            for name, p in self.oof_probs_.items()
        }
        return blend_probs(calibrated, self.weights)

    def predict(self, test_df: pd.DataFrame, verbose: bool = True) -> Prediction:
        if not self.fitted_:
            raise RuntimeError("predict() called before fit()")
        texts = test_df["text"].tolist()

        # 1) prompt clustering first: the stylegram twin needs the groups
        groups = self.clusterer.predict_groups(texts)
        sim = None
        confidence = None
        if hasattr(self.clusterer, "similarity"):
            try:
                sim = self.clusterer.similarity(texts)
                confidence = cluster_confidence(sim, groups)
            except Exception:
                confidence = None

        # 2) members -> temperature -> blend
        raw = predict_members(self.members, texts, groups=groups)
        calibrated = {n: apply_temperature(p, self.temperatures.get(n, 1.0)) for n, p in raw.items()}
        probs = blend_probs(calibrated, self.weights)
        stages: Dict[str, np.ndarray] = {"blend": probs.argmax(axis=1)}

        # 3) SLD-EM prior correction
        if bool(self.cfg.get("calibration.em_prior", True)):
            probs, pi_hat = sld_em(
                probs, self.class_prior_, iters=int(self.cfg.get("calibration.em_iters", 50))
            )
        stages["em"] = probs.argmax(axis=1)

        # 4) family experts
        if self.experts is not None:
            probs = self.experts.apply(probs)
        stages["experts"] = probs.argmax(axis=1)

        # 5) orthographic shield
        if self.shield is not None:
            probs = self.shield.apply(probs, texts)
        stages["shield"] = probs.argmax(axis=1)

        # 6) PRISM joint decode (last, so the structural constraint always holds)
        stats = {}
        if bool(self.cfg.get("prism.enabled", True)):
            preds, prism_stats = decode(
                probs, groups,
                mode=str(self.cfg.get("prism.mode", "relaxed")),
                lam=float(self.cfg.get("prism.lam", 1.0)),
                min_size=int(self.cfg.get("prism.min_size", 2)),
                max_size=int(self.cfg.get("prism.max_size", 18)),
                confidence=confidence,
                confidence_threshold=float(self.cfg.get("prism.confidence_threshold", 0.0)),
            )
            stats = prism_stats.as_dict()
            stats["cluster_sizes_mean"] = (
                float(np.mean(list(prism_stats.cluster_sizes.values())))
                if prism_stats.cluster_sizes else 0.0
            )
        else:
            preds = probs.argmax(axis=1)
        stages["prism"] = preds

        if verbose:
            n_clusters = len(np.unique(groups))
            print(f"[predict] rows={len(texts)} clusters={n_clusters} "
                  f"changed_by_prism={stats.get('n_changed', 0)}")
        return Prediction(probs=probs, preds=preds, groups=groups, stage_preds=stages, stats=stats)

    # ---------------------------------------------------------------- persist
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        ensure_dir(p.parent)
        joblib.dump(self, p)
        return p

    @staticmethod
    def load(path: str | Path) -> "PACT18Pipeline":
        return joblib.load(Path(path))

    # ------------------------------------------------------------------ info
    def describe(self) -> Dict[str, object]:
        return {
            "device": self.device_info.as_dict() if self.device_info else {},
            "members": [m.name for m in self.members],
            "weights": self.weights,
            "temperatures": self.temperatures,
            "shield_rules": self.shield.describe() if self.shield else [],
            "clusterer": getattr(self.clusterer, "name", "unknown"),
            "oof_summary": self.train_summary_,
        }
