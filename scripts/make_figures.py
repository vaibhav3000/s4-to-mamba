"""Generate all figures from committed experiment results.

Usage:
    python scripts/make_figures.py [--results results] [--out figures]

Every plot reads the committed JSON outputs (results/*.json and
results/efficiency_gpu.json); nothing is hand-typed. Model colors and styles
are consistent across all figures. Missing/None measurements (OOM, capped
lengths) are skipped gracefully.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODELS = ["transformer", "s4d", "mamba", "mamba2", "mamba3"]
LABELS = {
    "transformer": "Transformer",
    "s4d": "S4D",
    "mamba": "Mamba-1",
    "mamba2": "Mamba-2 (SSD)",
    "mamba3": "Mamba-3",
}
COLORS = {
    "transformer": "#0072B2",
    "s4d": "#E69F00",
    "mamba": "#009E73",
    "mamba2": "#CC79A7",
    "mamba3": "#D55E00",
}


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)


def _save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / name}.png")


def _load_runs(results: Path) -> dict[str, dict]:
    runs = {}
    for path in results.glob("*.json"):
        if path.name.startswith("efficiency"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "name" in data and "history" in data:
            runs[data["name"]] = data
    return runs


def _final(run: dict, key: str):
    hist = run.get("history") or []
    for entry in reversed(hist):
        if entry.get(key) is not None:
            return entry[key]
    return None


def _model_of(run_name: str) -> str:
    for m in MODELS:
        if run_name.endswith(m):
            return m
    return run_name


def efficiency_figures(results: Path, out: Path) -> None:
    path = results / "efficiency_gpu.json"
    if not path.exists():
        print("efficiency_gpu.json not found; skipping efficiency figures")
        return
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    series: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    ooms: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        model = e.get("model")
        L = e.get("seq_len")
        for key, target in [
            ("train_step_ms", "latency"),
            ("peak_train_mb", "memory"),
            ("train_tokens_per_s", "throughput"),
            ("decode_per_token_ms", "decode"),
        ]:
            v = e.get(key)
            if v is not None:
                series[target][model].append((L, v))
            elif e.get("oom_train") or e.get("oom") or e.get("oom_infer"):
                # Measured hardware limit: plot as an explicit marker, never
                # as a line across an unmeasured value.
                ooms[target][model].append(L)

    specs = [
        ("latency", "Training step time vs sequence length", "train step time (ms)", "fig_efficiency_latency", True),
        ("memory", "Peak GPU memory vs sequence length", "peak memory (MB)", "fig_efficiency_memory", False),
        ("throughput", "Training throughput vs sequence length", "tokens / s", "fig_efficiency_throughput", True),
        ("decode", "Decode cost per token vs sequence length", "per-token cost (ms)", "fig_efficiency_decode", True),
    ]
    for key, title, ylabel, name, loglog in specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        for model in MODELS:
            pts = sorted(series[key].get(model, []))
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", ms=4, color=COLORS[model], label=LABELS[model])
            # measured hardware limits get their own marker + annotation
            for L in sorted(set(ooms[key].get(model, []))):
                ax.annotate("OOM", (L, ax.get_ylim()[1] * 0.02), rotation=90,
                            fontsize=7, color=COLORS[model], ha="right", va="bottom")
                ax.plot([L], [ax.get_ylim()[1] * 0.02], marker="x", ms=6,
                        color=COLORS[model])
            if loglog:
                ax.set_xscale("log")
                ax.set_yscale("log")
            else:
                ax.set_xscale("log")
        ax.set_xlabel("sequence length")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _style(ax)
        _save(fig, out, name)


def training_figures(results: Path, out: Path) -> None:
    runs = _load_runs(results)
    if not runs:
        print("no training results found")
        return

    # quality bars per task
    task_specs = [
        ("imdb", "IMDb sentiment (final epoch)", [("accuracy", "accuracy"), ("auc", "AUC")], "fig_imdb_quality"),
        ("selective_copy", "Selective copying (final epoch)", [("sequence_success", "sequence success"), ("token_accuracy", "token accuracy")], "fig_selective_copy"),
        ("parity", "Parity / state tracking (final epoch)", [("token_accuracy", "token accuracy")], "fig_parity"),
    ]
    for prefix, title, metrics, name in task_specs:
        task_runs = {rn: r for rn, r in runs.items() if rn.startswith(prefix)}
        if not task_runs:
            continue
        models_present = sorted({_model_of(rn) for rn in task_runs}, key=MODELS.index)
        width = 0.8 / len(metrics)
        fig, ax = plt.subplots(figsize=(6, 4))
        for mi, (key, label) in enumerate(metrics):
            xs, ys = [], []
            for pos, model in enumerate(models_present):
                run = task_runs.get(f"{prefix}_{model}")
                if run is None:
                    continue
                v = _final(run, key)
                if v is not None:
                    xs.append(pos + mi * width)
                    ys.append(v)
            ax.bar(xs, ys, width=width * 0.9, label=label,
                   color=plt.cm.viridis(mi / max(len(metrics) - 1, 1)))
        ax.set_xticks([i + width * (len(metrics) - 1) / 2 for i in range(len(models_present))])
        ax.set_xticklabels([LABELS.get(m, m) for m in models_present], fontsize=9)
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if prefix == "parity":
            ax.axhline(0.5, ls="--", lw=1, color="gray")
            ax.text(len(models_present) - 0.5, 0.505, "chance", fontsize=8, color="gray")
        _save(fig, out, name)

    # learning curves 2x2
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        ("imdb", "accuracy", "IMDb accuracy"),
        ("selective_copy", "sequence_success", "Selective copy: sequence success"),
        ("parity", "token_accuracy", "Parity: token accuracy"),
        ("imdb", "train_loss", "IMDb train loss"),
    ]
    for ax, (prefix, key, title) in zip(axes.flat, panels):
        for run_name, run in sorted(runs.items()):
            if not run_name.startswith(prefix):
                continue
            model = _model_of(run_name)
            xs = [e.get("epoch") for e in run["history"]]
            ys = [e.get(key) for e in run["history"]]
            if any(y is None for y in ys):
                continue
            ax.plot(xs, ys, marker="o", ms=3, color=COLORS[model], label=LABELS[model])
        ax.set_title(title, fontsize=10)
        _style(ax)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out, "fig_learning_curves")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    results, out = Path(args.results), Path(args.out)
    if not results.exists():
        print(f"results directory '{results}' not found; nothing to do")
        return
    training_figures(results, out)
    efficiency_figures(results, out)


if __name__ == "__main__":
    main()
