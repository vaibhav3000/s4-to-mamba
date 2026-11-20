"""Numerical stability of every architecture under reduced precision (ssmbench-numerics).

For each model variant, sequence length and initialization seed (eval mode, batch 2,
inputs N(0, 1) of width d_model fed to the backbone):

- ``forward``: the backbone output in float32, under bf16 autocast and under fp16
  autocast, each compared with the same weights run in float64 (``*_vs_fp64``), and
  bf16 against float32. Under autocast the SSM scans and their input projections stay
  float32 (models/blocks.py), so the error comes from the other layers.
- ``decode``: prefill ``prompt_len`` positions, then ``steps`` decode steps, in float32
  and under bf16 autocast; every step's output is compared with the float64 forward
  pass over the same inputs, and the relative error is reported at steps 1, 16, 64, 256
  (those within range) and as the maximum over all steps. This shows whether error
  accumulates in the recurrent state (or KV cache) as decoding proceeds.

Each comparison gives the relative L2 error ||a - ref|| / ||ref||, the largest absolute
difference and the count of non-finite values. No pass/fail bounds are applied to the
results; they are reported as measured. Run with:

    ssmbench-numerics --config configs/numerics.yaml
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import torch

from ..models import build_backbone
from ..models.blocks import Backbone
from ..utils import dump_json, load_config, resolve_device, run_provenance, seed_everything
from .common import precision_context, resolve_variant

DRIFT_CHECKPOINTS = (1, 16, 64, 256)


def compare(a: torch.Tensor, ref: torch.Tensor) -> dict[str, float | int]:
    """Relative L2 error, max absolute difference (both in float64) and non-finite count."""
    a64, r64 = a.detach().to(torch.float64), ref.detach().to(torch.float64)
    nonfinite = int((~torch.isfinite(a64)).sum())
    finite_a = torch.nan_to_num(a64, nan=0.0, posinf=0.0, neginf=0.0)
    diff = finite_a - r64
    return {
        "rel_l2": float(diff.norm() / r64.norm()),
        "max_abs": float(diff.abs().max()),
        "nonfinite": nonfinite,
    }


def _models(
    name: str, kwargs: dict[str, Any], seed: int, device: torch.device
) -> tuple[Backbone, Backbone]:
    seed_everything(seed)
    model = build_backbone(name, kwargs).to(device).eval()
    reference = copy.deepcopy(model).double()
    return model, reference


def forward_errors(
    name: str, kwargs: dict[str, Any], seq_len: int, seed: int, device: torch.device
) -> dict[str, dict[str, float | int]]:
    """Backbone-output errors of float32, bf16 and fp16 autocast (see the module docstring)."""
    model, reference = _models(name, kwargs, seed, device)
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    x = torch.randn(2, seq_len, kwargs["d_model"], generator=gen).to(device)
    with torch.no_grad():
        y64 = reference(x.double())
        y32 = model(x)
        outputs = {}
        for precision in ("bf16", "fp16"):
            with precision_context(device, precision):
                outputs[precision] = model(x).float()
    return {
        "fp32_vs_fp64": compare(y32, y64),
        "bf16_vs_fp64": compare(outputs["bf16"], y64),
        "fp16_vs_fp64": compare(outputs["fp16"], y64),
        "bf16_vs_fp32": compare(outputs["bf16"], y32),
    }


def decode_drift(
    name: str,
    kwargs: dict[str, Any],
    prompt_len: int,
    steps: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """Per-step decode error against the float64 forward pass, float32 and bf16."""
    model, reference = _models(name, kwargs, seed, device)
    total = prompt_len + steps
    gen = torch.Generator(device="cpu").manual_seed(seed + 2)
    x = torch.randn(2, total, kwargs["d_model"], generator=gen).to(device)
    checkpoints = [c for c in DRIFT_CHECKPOINTS if c <= steps]
    if steps not in checkpoints:
        checkpoints.append(steps)
    with torch.no_grad():
        ref = reference(x.double())[:, prompt_len:]  # (B, steps, D)
    out: dict[str, dict[str, Any]] = {}
    for precision in ("fp32", "bf16"):
        with torch.no_grad(), precision_context(device, precision):
            _, state = model.prefill(x[:, :prompt_len], max_len=total)
            ys = [model.step(x[:, t : t + 1], state).float() for t in range(prompt_len, total)]
        y = torch.cat(ys, dim=1)
        per_step = [compare(y[:, i], ref[:, i])["rel_l2"] for i in range(steps)]
        out[precision] = {
            "checkpoints": checkpoints,
            "rel_l2_at": [per_step[c - 1] for c in checkpoints],
            "max_rel_l2": max(per_step),
            "nonfinite": compare(y, ref)["nonfinite"],
        }
    return out


def run_numerics(cfg: dict[str, Any]) -> dict[str, Any]:
    """Every (variant, length, seed) of cfg; writes cfg["out_path"] after each entry."""
    device = resolve_device(cfg.get("device", "auto"))
    out_path = Path(cfg["out_path"])
    results: dict[str, Any] = {
        "environment": {**run_provenance(device), "date": time.strftime("%Y-%m-%d")},
        "config": cfg,
        "entries": [],
    }
    decode_cfg = cfg.get("decode", {"prompt_len": 1024, "steps": 256})
    for label in cfg["models"]:
        name, kwargs = resolve_variant(label, cfg.get("variants"))
        for seq_len in cfg["seq_lengths"]:
            for seed in cfg.get("seeds", [0, 1, 2]):
                print(f"numerics {label} T={seq_len} seed={seed} ...", flush=True)
                entry: dict[str, Any] = {
                    "model": label,
                    "architecture": name,
                    "model_kwargs": kwargs,
                    "seq_len": seq_len,
                    "seed": seed,
                    "forward": forward_errors(name, kwargs, seq_len, seed, device),
                }
                if seq_len == cfg["seq_lengths"][0]:
                    entry["decode"] = decode_drift(
                        name, kwargs, decode_cfg["prompt_len"], decode_cfg["steps"], seed, device
                    )
                results["entries"].append(entry)
                dump_json(results, out_path)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="ssmbench-numerics",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run_numerics(load_config(args.config))


if __name__ == "__main__":
    main()
