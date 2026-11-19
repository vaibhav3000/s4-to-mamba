"""ssmbench-train: run every model of a training config.

Usage:
    ssmbench-train --config configs/imdb.yaml --out results
    ssmbench-train --config configs/parity.yaml --models mamba3 --device cuda

For each entry in the config's `models` mapping, builds a complete run config
(task + data + train + one model) and calls run_training. Results are written
to <out>/<name_prefix>_<model>.json.
"""

from __future__ import annotations

import argparse

from ..utils import load_config
from .trainer import run_training


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="ssmbench-train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="*", default=None, help="subset of models to run")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    prefix = cfg["name_prefix"]
    model_items = list(cfg["models"].items())
    if args.models:
        unknown = set(args.models) - {m for m, _ in model_items}
        if unknown:
            ap.error(f"models {sorted(unknown)} are not in {args.config}")
        wanted = set(args.models)
        model_items = [(m, k) for m, k in model_items if m in wanted]

    for model_name, model_kwargs in model_items:
        run_cfg = {
            "name": f"{prefix}_{model_name}",
            "task": cfg["task"],
            "data": cfg["data"],
            "train": cfg["train"],
            "model": {"name": model_name, "kwargs": model_kwargs},
            "device": args.device,
        }
        print(f"=== running {run_cfg['name']} ===", flush=True)
        run_training(run_cfg, args.out)


if __name__ == "__main__":
    main()
