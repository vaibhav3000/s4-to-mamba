"""ssmbench-train: run every model variant of a training config, for one or more run seeds.

Usage:
    ssmbench-train --config configs/parity.yaml --out results/train --seeds 0 1 2
    ssmbench-train --config configs/parity.yaml --models mamba3 --device cuda

The config's `models` mapping is keyed by model variant label; an optional `backbone:`
entry names the architecture (default: the label), and the other entries are its
constructor arguments. For each (variant, seed) this builds a run config (task, data,
train with train.seed = the run seed, one model) and calls run_training, which writes
<out>/<name_prefix>/<variant>/seed<k>.json. Without --seeds the config's train.seed is
the only seed.

A run whose result file was written by the same source and run config is skipped; a
result file from other code or settings stops the command unless --force is given (it
is then overwritten). With --checkpoints DIR every epoch is checkpointed under
DIR/<name_prefix>/<variant>/seed<k>/ and an interrupted run resumes from there.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ..utils import load_config, source_sha256
from .trainer import run_config_sha256, run_training


def split_variant(label: str, spec: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """(architecture, constructor kwargs) of a variant entry; `backbone` defaults to label."""
    kwargs = dict(spec)
    return str(kwargs.pop("backbone", label)), kwargs


def _completed(path: Path, cfg: dict) -> bool | None:
    """True if path holds this run's result from the same source and run config, False if
    it holds a result from other code or settings, None if there is no result file."""
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    provenance = record.get("provenance") or {}
    return provenance.get("source_sha256") == source_sha256() and provenance.get(
        "config_sha256"
    ) == run_config_sha256(cfg)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="ssmbench-train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/train")
    ap.add_argument("--models", nargs="*", default=None, help="subset of model variants to run")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="run seeds")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--checkpoints", default=None, help="directory for per-epoch checkpoints")
    ap.add_argument("--force", action="store_true", help="overwrite results from other code")
    args = ap.parse_args(argv)

    # Deterministic cuBLAS needs this before CUDA initializes (docs/adr/0008).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cfg = load_config(args.config)
    prefix = cfg["name_prefix"]
    model_items = list(cfg["models"].items())
    if args.models:
        unknown = set(args.models) - {m for m, _ in model_items}
        if unknown:
            ap.error(f"models {sorted(unknown)} are not in {args.config}")
        wanted = set(args.models)
        model_items = [(m, k) for m, k in model_items if m in wanted]
    seeds = args.seeds if args.seeds else [int(cfg["train"].get("seed", 42))]

    runs = []
    for label, spec in model_items:
        backbone, kwargs = split_variant(label, spec)
        for seed in seeds:
            run_cfg: dict[str, Any] = {
                "name": f"{prefix}_{label}",
                "config": prefix,
                "label": label,
                "task": cfg["task"],
                "data": cfg["data"],
                "train": {**cfg["train"], "seed": seed},
                "model": {"name": backbone, "kwargs": kwargs},
                "device": args.device,
            }
            if args.checkpoints:
                run_cfg["checkpoint_dir"] = str(
                    Path(args.checkpoints) / prefix / label / f"seed{seed}"
                )
            out_dir = Path(args.out) / prefix / label
            runs.append((run_cfg, out_dir))

    foreign = [
        str(out / f"seed{c['train']['seed']}.json")
        for c, out in runs
        if _completed(out / f"seed{c['train']['seed']}.json", c) is False
    ]
    if foreign and not args.force:
        raise SystemExit(
            f"error: {len(foreign)} result file(s) were written by other code or settings "
            f"({', '.join(foreign[:3])}{', ...' if len(foreign) > 3 else ''}). Move them away, "
            "or pass --force to overwrite them."
        )
    for run_cfg, out_dir in runs:
        path = out_dir / f"seed{run_cfg['train']['seed']}.json"
        if _completed(path, run_cfg):
            print(f"=== skipping {run_cfg['name']} seed {run_cfg['train']['seed']} (done) ===")
            continue
        print(f"=== running {run_cfg['name']} seed {run_cfg['train']['seed']} ===", flush=True)
        run_training(run_cfg, str(out_dir))


if __name__ == "__main__":
    main()
