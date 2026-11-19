"""Smoke check before any long GPU job (docs/design/step2.md section 8, job 0).

1. Training: for every variant of every training config, two optimizer steps from the
   run seed, done twice with the config's deterministic setting; the losses must be
   finite and bitwise equal across the two attempts (ADR 0008).
2. Attention: every pinned SDPA backend the benchmarks use runs one causal forward and
   backward at benchmark shapes and one single-query decode-shaped call, in its
   precision; a backend that refuses is recorded, not treated as a failure.

Writes results/logs/smoke.json and exits non-zero if any training variant failed.

Usage: .venv/Scripts/python.exe scripts/gpu_smoke.py --device cuda
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ssmbench.benchmarks.common import is_unsupported_kernel, precision_context
from ssmbench.models.attention import attention, detect_attention_backend
from ssmbench.train.cli import split_variant
from ssmbench.train.trainer import (
    build_model,
    build_param_groups,
    deterministic_algorithms,
    get_loaders,
)
from ssmbench.utils import dump_json, load_config, seed_everything

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIGS = ["parity", "selective_copy", "exact_copy", "exact_copy_varlen", "imdb"]
BENCHMARK_BACKENDS = {"fp32": ["efficient", "math"], "bf16": ["flash", "cudnn", "efficient"]}


def _two_steps(run_cfg: dict[str, Any], steps: int, device: torch.device) -> list[float]:
    seed_everything(int(run_cfg["train"].get("seed", 42)))
    model = build_model(run_cfg).to(device)
    loaders = get_loaders(run_cfg)
    opt = torch.optim.AdamW(build_param_groups(model, run_cfg["train"].get("weight_decay", 0.0)))
    task = run_cfg["task"]["type"]
    losses = []
    model.train()
    for _, batch in zip(range(steps), loaders.train, strict=False):
        tokens = batch["input_ids"].to(device)
        logits = model(tokens)
        if task == "classification":
            loss = F.cross_entropy(logits, batch["label"].to(device))
        else:
            targets = batch["targets"].to(device)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def smoke_training(cfg: dict[str, Any], device: str, steps: int = 2) -> dict[str, dict[str, Any]]:
    """Per variant: losses of `steps` steps, whether two attempts agree bitwise, finiteness."""
    dev = torch.device(device)
    report: dict[str, dict[str, Any]] = {}
    for label, spec in cfg["models"].items():
        backbone, kwargs = split_variant(label, spec)
        run_cfg = {
            "task": cfg["task"],
            "data": cfg["data"],
            "train": cfg["train"],
            "model": {"name": backbone, "kwargs": kwargs},
        }
        attempts = []
        for _ in range(2):
            with deterministic_algorithms(bool(cfg["train"].get("deterministic", False))):
                attempts.append(_two_steps(run_cfg, steps, dev))
        report[label] = {
            "losses": attempts[0],
            "deterministic": attempts[0] == attempts[1],
            "finite": all(math.isfinite(v) for v in attempts[0]),
        }
    return report


def smoke_attention(
    device: str,
    backends: list[str],
    precision: str,
    batch: int = 8,
    seq_len: int = 1024,
    heads: int = 4,
    head_dim: int = 64,
) -> dict[str, dict[str, Any]]:
    """Per backend: whether a causal forward+backward and a single-query decode call run,
    and which SDPA kernel family ran; the error text when the backend refuses."""
    dev = torch.device(device)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    report: dict[str, dict[str, Any]] = {}
    for backend in backends:
        q, k, v = (
            torch.randn(
                batch, heads, seq_len, head_dim, device=dev, dtype=dtype, requires_grad=True
            )
            for _ in range(3)
        )

        def run(
            backend: str = backend, q: torch.Tensor = q, k: torch.Tensor = k, v: torch.Tensor = v
        ) -> None:
            with precision_context(dev, "fp32"):
                attention(q, k, v, causal=True, backend=backend).float().sum().backward()
                with torch.no_grad():
                    attention(q[:, :, -1:], k, v, causal=False, backend=backend)

        try:
            ran = detect_attention_backend(run)
            report[backend] = {"ok": True, "ran": ran, "precision": precision}
        except RuntimeError as exc:
            if not is_unsupported_kernel(exc):
                raise
            report[backend] = {
                "ok": False,
                "unsupported": True,
                "error": str(exc)[:300],
                "precision": precision,
            }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()
    results: dict[str, Any] = {"training": {}, "attention": {}}
    failed = []
    for name in TRAIN_CONFIGS:
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        rows = smoke_training(cfg, args.device, args.steps)
        results["training"][name] = rows
        failed += [
            f"{name}/{label}"
            for label, r in rows.items()
            if not (r["deterministic"] and r["finite"])
        ]
        print(f"{name}: {len(rows)} variants checked", flush=True)
    for precision, backends in BENCHMARK_BACKENDS.items():
        results["attention"][precision] = smoke_attention(args.device, backends, precision)
    dump_json(results, ROOT / "results" / "logs" / "smoke.json")
    for precision, rows in results["attention"].items():
        for backend, row in rows.items():
            print(
                f"attention {precision} {backend}: {'ok ' + str(row['ran']) if row['ok'] else 'refused'}"
            )
    if failed:
        print(f"FAILED (not deterministic or not finite): {', '.join(failed)}", file=sys.stderr)
        raise SystemExit(1)
    print("smoke check passed")


if __name__ == "__main__":
    main()
