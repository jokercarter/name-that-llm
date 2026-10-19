"""The 18 candidate models and their provider families."""

from __future__ import annotations

from typing import Dict, List, Tuple

LABELS: List[str] = [
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "qwen3.8-max",
    "glm-5.3",
    "glm-5.3-flash",
    "kimi-k3",
]

FAMILIES: Dict[str, List[str]] = {
    "openai": ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"],
    "google": [
        "gemini-3.1-pro-preview",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
    ],
    "anthropic": [
        "claude-fable-5",
        "claude-opus-4-6",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-5",
    ],
    "qwen": ["qwen3.8-max"],
    "zai": ["glm-5.3", "glm-5.3-flash"],
    "moonshot": ["kimi-k3"],
}

FAMILY_NAMES: List[str] = list(FAMILIES.keys())

LABEL_TO_INDEX: Dict[str, int] = {label: i for i, label in enumerate(LABELS)}
FAMILY_OF_LABEL: Dict[str, str] = {
    label: family for family, labels in FAMILIES.items() for label in labels
}
LABEL_FAMILY_INDEX = [FAMILY_NAMES.index(FAMILY_OF_LABEL[label]) for label in LABELS]

N_CLASSES = len(LABELS)
N_FAMILIES = len(FAMILY_NAMES)


def labels_to_indices(labels) -> List[int]:
    return [LABEL_TO_INDEX[label] for label in labels]


def indices_to_labels(indices) -> List[str]:
    return [LABELS[int(i)] for i in indices]


def family_masks() -> List[Tuple[str, List[int]]]:
    """Return ``[(family, [class indices])]`` with at least two members."""
    out = []
    for family in FAMILY_NAMES:
        idx = [LABEL_TO_INDEX[label] for label in FAMILIES[family]]
        if len(idx) >= 2:
            out.append((family, idx))
    return out


def family_index_array():
    import numpy as np

    return np.asarray(LABEL_FAMILY_INDEX, dtype=int)


def validate_labels(values) -> None:
    bad = sorted({v for v in values if v not in LABEL_TO_INDEX})
    if bad:
        raise ValueError(
            f"unknown model label(s): {bad[:5]} (expected one of {LABELS[:3]}... )"
        )
