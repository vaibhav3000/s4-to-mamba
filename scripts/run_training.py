"""Run all training experiments defined in a config file.

Usage:
    python scripts/run_training.py --config configs/imdb.yaml --out results/

For each entry in the config's `models` mapping, builds a complete run config
(task + data + train + one model) and calls run_training. Results are written
to <out>/<name_prefix>_<model>.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssmbench.train import run_training  # noqa: E402
from ssmbench.utils import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="*", default=None, help="subset of models to run")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prefix = cfg["name_prefix"]
    model_items = list(cfg["models"].items())
    if args.models:
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
