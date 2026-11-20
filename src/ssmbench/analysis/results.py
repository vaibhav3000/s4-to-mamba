"""Reading the committed results for tables, figures and the demo (docs/adr/0006).

Every number a table or figure shows comes from these files:

    results/train/<config>/summary.json        (ssmbench-summarize)
    results/efficiency/sweep_<precision>.json  (ssmbench-efficiency)
    results/inference/decode_<precision>.json  (ssmbench-inference)
    results/numerics/numerics.json             (ssmbench-numerics)
    results/compile/compile.json               (ssmbench-compile)

A missing file means the job has not run; callers show it as pending, never as a value.
Labels, colors and markers are fixed per model variant, so a variant looks the same in
every figure (colors: the validated categorical order of the dataviz reference palette).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRAIN_CONFIGS = ["parity", "selective_copy", "exact_copy", "exact_copy_varlen", "imdb"]
PRECISIONS = ["fp32", "bf16"]

# Training variants, in table and legend order.
VARIANTS = [
    "transformer",
    "transformer_rope",
    "s4d",
    "mamba",
    "mamba2",
    "mamba3",
    "mamba3_no_rotation",
]
LABELS = {
    "transformer": "Transformer (sinusoidal)",
    "transformer_rope": "Transformer (RoPE)",
    "s4d": "S4D",
    "mamba": "Mamba-1",
    "mamba2": "Mamba-2",
    "mamba3": "Mamba-3",
    "mamba3_no_rotation": "Mamba-3, no rotation",
    # benchmark variants
    "transformer_flash": "Transformer, FlashAttention",
    "transformer_cudnn": "Transformer, cuDNN attention",
    "transformer_efficient": "Transformer, mem.-efficient attention",
    "transformer_math": "Transformer, math attention",
    "transformer_naive": "Transformer, naive attention (reference)",
}
# Categorical slots 1 to 7 of the reference palette (validated, adjacent order).
COLORS = {
    "transformer": "#2a78d6",
    "transformer_rope": "#eb6834",
    "s4d": "#1baf7a",
    "mamba": "#eda100",
    "mamba2": "#e87ba4",
    "mamba3": "#008300",
    "mamba3_no_rotation": "#6250d6",
}
MARKERS = {
    "transformer": "o",
    "transformer_rope": "s",
    "s4d": "D",
    "mamba": "^",
    "mamba2": "v",
    "mamba3": "P",
    "mamba3_no_rotation": "X",
}
# Attention-backend rows share the Transformer's identity but differ in line style.
BACKEND_STYLES = {
    "transformer_flash": "-",
    "transformer_cudnn": (0, (5, 2)),
    "transformer_efficient": (0, (2, 2)),
    "transformer_math": (0, (6, 2, 1, 2)),
    "transformer_naive": (0, (1, 1)),
}

# The metric each training table and figure reports, and the test sets in order.
PRIMARY = {
    "parity": ("token_accuracy", "token accuracy"),
    "selective_copy": ("token_accuracy", "answer-token accuracy"),
    "exact_copy": ("token_accuracy", "token accuracy"),
    "exact_copy_varlen": ("token_accuracy", "token accuracy"),
    "imdb": ("accuracy", "test accuracy"),
}


def style_key(label: str) -> str:
    """The training-variant key whose color and marker a benchmark label uses."""
    return "transformer" if label.startswith("transformer_") and label in BACKEND_STYLES else label


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_summary(root: Path, config: str) -> dict[str, Any] | None:
    return load_json(root / "train" / config / "summary.json")


def load_sweep(root: Path, kind: str, precision: str) -> dict[str, Any] | None:
    name = {"efficiency": "sweep", "inference": "decode"}[kind]
    return load_json(root / kind / f"{name}_{precision}.json")


def eval_order(summary: dict[str, Any]) -> list[str]:
    """Test-set names of a summary in their natural order (lengths ascending)."""
    names: list[str] = []
    for variant in summary["variants"].values():
        for name in variant["evals"]:
            if name not in names:
                names.append(name)

    def key(name: str) -> tuple[int, int | str]:
        if "=" in name:
            return (0, int(name.split("=", 1)[1]))
        return (1, {"id": "0", "ood": "1", "test": "2"}.get(name, name))

    return sorted(names, key=key)


def variants_in(summary: dict[str, Any]) -> list[str]:
    present = list(summary["variants"])
    return [v for v in VARIANTS if v in present] + [v for v in present if v not in VARIANTS]


def mean_std(stats: dict[str, Any] | None, scale: float = 100.0, digits: int = 1) -> str:
    """'97.3 ± 1.2' (scaled, sample std), or the mean alone for one seed, or '-'."""
    if not stats:
        return "-"
    mean = f"{stats['mean'] * scale:.{digits}f}"
    if stats.get("std") is None:
        return mean
    return f"{mean} ± {stats['std'] * scale:.{digits}f}"


def point_status(entry: dict[str, Any]) -> str | None:
    """Why a sweep point has no timing: 'OOM', 'skipped', 'unsupported', or None."""
    if entry.get("unsupported"):
        return "unsupported"
    if entry.get("skipped"):
        return "skipped"
    if entry.get("oom") or entry.get("oom_train"):
        return "OOM"
    return None
