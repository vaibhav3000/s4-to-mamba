"""Multi-seed summaries of training runs (ssmbench-summarize).

A summary covers every run of one experiment config: per model variant and test
evaluation set, the mean, sample standard deviation (ddof = 1), count, min, max and the
per-seed values of each metric, with the variant's parameter count and the task's chance
levels. Runs must come from one source tree, and the seeds of a variant must share every
setting except the seed, so a summary row never mixes code versions or configurations.

Usage:
    ssmbench-summarize results/train/parity results/train/selective_copy
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from ..utils import dump_json


class SummaryError(ValueError):
    """The runs cannot be summarized together."""


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def chance_levels(data_config: dict[str, Any]) -> dict[str, float]:
    """Score of guessing from the label distribution alone, per metric.

    parity: 0.5 per token. selective copy: 1 / (number of value tokens) per answer token.
    exact copy: 1 / alphabet size per letter (the end-of-string position is ignored).
    IMDb: 0.5 accuracy and AUC on the balanced test split.
    """
    name = data_config["name"]
    if name == "parity":
        return {"token_accuracy": 0.5}
    if name == "selective_copy":
        return {"token_accuracy": 1.0 / (data_config["train"]["vocab_size"] - 3)}
    if name == "exact_copy":
        return {"token_accuracy": 1.0 / data_config["alphabet_size"]}
    if name == "imdb":
        return {"accuracy": 0.5, "auc": 0.5}
    raise SummaryError(f"no chance level defined for task {name!r}")


def _settings(run: dict[str, Any]) -> str:
    train = {k: v for k, v in run["train_config"].items() if k != "seed"}
    return json.dumps(
        {"model": run["model"], "train": train, "data": run["data_config"]}, sort_keys=True
    )


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summary of the result records of one experiment config (see the module docstring)."""
    if not runs:
        raise SummaryError("no runs to summarize")
    sources = {r["provenance"]["source_sha256"] for r in runs}
    if len(sources) > 1:
        raise SummaryError(
            f"runs come from {len(sources)} different source trees: {sorted(sources)}"
        )
    by_label: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_label.setdefault(run["label"], []).append(run)

    variants: dict[str, Any] = {}
    for label, group in sorted(by_label.items()):
        group.sort(key=lambda r: r["seed"])
        seeds = [r["seed"] for r in group]
        if len(set(seeds)) != len(seeds):
            raise SummaryError(f"{label}: seed listed twice in {seeds}")
        if len({_settings(r) for r in group}) > 1:
            raise SummaryError(f"{label}: seeds were trained with different settings")
        evals: dict[str, dict[str, Any]] = {}
        for eval_name in group[0]["final_evals"]:
            metrics = group[0]["final_evals"][eval_name]
            evals[eval_name] = {
                metric: _stats([float(r["final_evals"][eval_name][metric]) for r in group])
                for metric, value in metrics.items()
                if value is not None
            }
        variants[label] = {
            "architecture": group[0]["model"]["name"],
            "model_kwargs": group[0]["model"]["kwargs"],
            "n_parameters": group[0]["n_parameters"],
            "attention_backend": group[0].get("attention_backend"),
            "seeds": seeds,
            "wall_time_seconds": _stats([float(r["wall_time_seconds"]) for r in group]),
            "evals": evals,
        }
    first = runs[0]
    return {
        "config": first.get("config"),
        "task": first["task"],
        "source_sha256": sources.pop(),
        "git_sha": sorted({r["provenance"].get("git_sha") or "" for r in runs}),
        "chance": chance_levels(first["data_config"]),
        "variants": variants,
    }


def summarize_dir(config_dir: Path) -> dict[str, Any]:
    """Summarize every <variant>/seed<k>.json under config_dir into config_dir/summary.json."""
    paths = sorted(config_dir.glob("*/seed*.json"))
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    summary = summarize_runs(runs)
    dump_json(summary, config_dir / "summary.json")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="ssmbench-summarize",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("config_dirs", nargs="+", type=Path, help="results/train/<config> directories")
    args = ap.parse_args(argv)
    for config_dir in args.config_dirs:
        try:
            summary = summarize_dir(config_dir)
        except SummaryError as exc:
            raise SystemExit(f"error: {config_dir}: {exc}") from exc
        print(f"{config_dir}: {len(summary['variants'])} variants summarized", flush=True)


if __name__ == "__main__":
    main()
