"""Structure ablation on pseudo-test pools.

The public notebooks validate with "pooled simulation": held-out prompts are
turned into pseudo test pools with k passages per prompt. We extend that by
injecting clustering noise (splits and merges), because PRISM's benefit depends
on cluster purity and the risk has to be measured, not assumed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration import sinkhorn_balance
from .evaluate import balanced_accuracy
from .prism import decode


def _build_pools(prompt_codes: np.ndarray, k: int, rng: np.random.Generator,
                 max_prompts: Optional[int] = None,
                 split_rate: float = 0.0, merge_rate: float = 0.0
                 ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Return (row indices, cluster ids, prompt ids) for a synthetic test pool."""
    prompts = np.unique(prompt_codes)
    if max_prompts is not None and len(prompts) > max_prompts:
        prompts = rng.choice(prompts, size=max_prompts, replace=False)

    rows: List[int] = []
    clusters: List[int] = []
    prompt_ids: List[int] = []
    next_cluster = 0
    for p in prompts:
        idx = np.flatnonzero(prompt_codes == p)
        if len(idx) < 2:
            continue
        take = min(k, len(idx))
        chosen = rng.choice(idx, size=take, replace=False)
        rows.extend(chosen.tolist())
        if split_rate > 0 and take >= 4 and rng.random() < split_rate:
            cut = int(rng.integers(2, take - 1))
            perm = rng.permutation(take)
            clusters.extend([next_cluster] * cut)
            clusters.extend([next_cluster + 1] * (take - cut))
            next_cluster += 2
            prompt_ids.extend([int(p)] * take)
        else:
            clusters.extend([next_cluster] * take)
            next_cluster += 1
            prompt_ids.extend([int(p)] * take)

    rows_arr = np.asarray(rows, dtype=int)
    clusters_arr = np.asarray(clusters, dtype=int)

    if merge_rate > 0 and len(np.unique(clusters_arr)) > 1:
        uniq = np.unique(clusters_arr)
        for c in uniq:
            if rng.random() < merge_rate:
                others = uniq[uniq != c]
                if len(others):
                    target = int(rng.choice(others))
                    clusters_arr[clusters_arr == c] = target
    return rows_arr, clusters_arr, prompt_ids


def run_structure_ablation(probs: np.ndarray, y_idx: Sequence[int], prompt_codes: Sequence[int],
                           k_values: Sequence[int] = (4, 8, 12, 18), lam: float = 1.0,
                           alpha: float = 0.25, split_rate: float = 0.0,
                           merge_rate: float = 0.0, max_prompts: Optional[int] = None,
                           seed: int = 42, repeats: int = 3) -> List[Dict[str, float]]:
    """Compare argmax / Sinkhorn / PRISM(hard) / PRISM(relaxed) on pseudo pools."""
    probs = np.asarray(probs, dtype=np.float64)
    y_idx = np.asarray(y_idx, dtype=int)
    prompt_codes = np.asarray(prompt_codes)
    rows_out: List[Dict[str, float]] = []

    for k in k_values:
        acc: Dict[str, List[float]] = {name: [] for name in ("argmax", "sinkhorn", "prism_hard", "prism_relaxed")}
        for r in range(int(repeats)):
            rng = np.random.default_rng(seed + 1000 * r + k)
            rows, clusters, _ = _build_pools(
                prompt_codes, k, rng, max_prompts=max_prompts,
                split_rate=split_rate, merge_rate=merge_rate,
            )
            if len(rows) < 10:
                continue
            p = probs[rows]
            y = y_idx[rows]

            acc["argmax"].append(balanced_accuracy(y, p.argmax(axis=1)))
            acc["sinkhorn"].append(balanced_accuracy(y, sinkhorn_balance(p, alpha).argmax(axis=1)))
            pred_hard, _ = decode(p, clusters, mode="hard", lam=lam)
            acc["prism_hard"].append(balanced_accuracy(y, pred_hard))
            pred_relaxed, _ = decode(p, clusters, mode="relaxed", lam=lam)
            acc["prism_relaxed"].append(balanced_accuracy(y, pred_relaxed))

        for name, values in acc.items():
            if not values:
                continue
            rows_out.append({
                "k_per_prompt": k,
                "strategy": name,
                "balanced_accuracy": float(np.mean(values)),
                "std": float(np.std(values)),
                "split_rate": split_rate,
                "merge_rate": merge_rate,
            })
    return rows_out


def format_ablation(rows: Sequence[Dict[str, float]]) -> str:
    if not rows:
        return "(no rows)"
    ks = sorted({int(r["k_per_prompt"]) for r in rows})
    strategies = ["argmax", "sinkhorn", "prism_hard", "prism_relaxed"]
    header = f"{'strategy':14s}" + "".join(f"{('k=' + str(k)):>9s}" for k in ks)
    lines = [header, "-" * len(header)]
    for strat in strategies:
        cells = []
        for k in ks:
            match = [r for r in rows if r["strategy"] == strat and int(r["k_per_prompt"]) == k]
            cells.append(f"{match[0]['balanced_accuracy']:9.4f}" if match else "      n/a")
        lines.append(f"{strat:14s}" + "".join(cells))
    return "\n".join(lines)
