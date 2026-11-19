"""Parameter-matched widths for the model variants of each training config (docs/adr/0005).

For every config, the anchor is the total parameter count (embedding, backbone and head)
of its `transformer` variant. Every other variant keeps all its settings except d_model,
which is set to the value on its architecture's constraint grid whose total count is
closest to the anchor. The script prints the chosen widths and the relative difference;
with --check it exits non-zero if a config's widths differ from the computed ones.

Usage:
    python scripts/match_params.py                 # print the matched widths
    python scripts/match_params.py --check         # verify the committed configs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ssmbench.train.cli import split_variant
from ssmbench.train.trainer import build_model
from ssmbench.utils import count_parameters, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["parity", "selective_copy", "exact_copy", "exact_copy_varlen", "imdb"]
ANCHOR = "transformer"
WIDTHS = range(8, 1025, 2)


def total_parameters(cfg: dict[str, Any], label: str, spec: dict[str, Any]) -> int:
    """Trainable parameters of the full model (embedding, backbone, head) of one variant."""
    backbone, kwargs = split_variant(label, spec)
    run_cfg = {"task": cfg["task"], "model": {"name": backbone, "kwargs": kwargs}}
    return count_parameters(build_model(run_cfg))


def _valid(backbone: str, kwargs: dict[str, Any], d_model: int) -> bool:
    if backbone in ("mamba2", "mamba3"):
        d_inner = kwargs.get("expand", 2) * d_model
        if d_inner % kwargs.get("headdim", 64):
            return False
        return (d_inner // kwargs.get("headdim", 64)) % kwargs.get("n_groups", 1) == 0
    if backbone == "transformer":
        heads = kwargs.get("n_heads", 4)
        return d_model % heads == 0 and (d_model // heads) % 2 == 0
    return True


def matched_width(cfg: dict[str, Any], label: str, spec: dict[str, Any], target: int) -> int:
    """d_model on the variant's constraint grid whose total count is closest to target."""
    backbone, kwargs = split_variant(label, spec)
    best: tuple[float, int] | None = None
    for d in WIDTHS:
        if not _valid(backbone, kwargs, d):
            continue
        n = total_parameters(cfg, label, {**spec, "d_model": d})
        gap = abs(n - target) / target
        if best is None or gap < best[0]:
            best = (gap, d)
        if n > 2 * target:
            break
    if best is None:
        raise ValueError(f"{label}: no valid width")
    return best[1]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    mismatches = 0
    for name in CONFIGS:
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        target = total_parameters(cfg, ANCHOR, cfg["models"][ANCHOR])
        print(f"{name}: anchor {ANCHOR} = {target:,} parameters")
        for label, spec in cfg["models"].items():
            if label == ANCHOR:
                continue
            width = matched_width(cfg, label, spec, target)
            n = total_parameters(cfg, label, {**spec, "d_model": width})
            flag = "" if spec.get("d_model") == width else f"  (config has {spec.get('d_model')})"
            mismatches += bool(flag)
            print(
                f"  {label:<22} d_model {width:>4}  {n:>10,}  {100 * (n - target) / target:+.1f}%{flag}"
            )
    if args.check and mismatches:
        raise SystemExit(f"{mismatches} variant(s) differ from the matched widths")


if __name__ == "__main__":
    main()
