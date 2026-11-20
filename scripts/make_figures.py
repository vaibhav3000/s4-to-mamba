"""Generate every figure from the committed results (docs/adr/0006).

Usage:
    .venv/Scripts/python.exe scripts/make_figures.py [--results results] [--out figures]
                                                     [--report-out reports/figures]

Reads only the files listed in ssmbench/analysis/results.py. Writes PNG and PDF to
--out (README, demo) and the PDFs again to --report-out (the LaTeX report). A figure
whose results are missing is not drawn. Conventions in every figure:

- one fixed color and marker per model variant (validated categorical palette), and a
  line style per attention backend for Transformer benchmark rows;
- training results: mean over run seeds with error bars of one sample std; the chance
  level as a dashed gray line;
- benchmarks: medians; points that spilled to system memory are ringed; points with no
  timing (out of memory, skipped, unsupported kernel) are not drawn and are listed in
  the generated tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ssmbench.analysis.results import (
    BACKEND_STYLES,
    COLORS,
    LABELS,
    MARKERS,
    PRIMARY,
    eval_order,
    load_summary,
    load_sweep,
    point_status,
    style_key,
    variants_in,
)

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "savefig.facecolor": "#fcfcfb",
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "figure.constrained_layout.use": True,
    }
)


def _style(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig: Figure, out: Path, report_out: Path | None, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = [out / f"{name}.png", out / f"{name}.pdf"]
    fig.savefig(written[0], dpi=200, bbox_inches="tight")
    # No CreationDate, so re-running on unchanged JSON gives identical PDFs.
    fig.savefig(written[1], bbox_inches="tight", metadata={"CreationDate": None})
    if report_out is not None:
        report_out.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            report_out / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None}
        )
        written.append(report_out / f"{name}.pdf")
    plt.close(fig)
    return written


def _chance(ax: Axes, value: float | None, scale: float = 100.0) -> None:
    """Dashed chance line, labelled just outside the right edge of the axes."""
    if value is not None:
        ax.axhline(value * scale, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0, zorder=0)
        ax.annotate(
            "chance",
            (1.0, value * scale),
            xycoords=("axes fraction", "data"),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=MUTED,
            fontsize=7,
            annotation_clip=False,
        )


def _lengths_figure(summaries: list[tuple[str, dict[str, Any]]], ylabel: str) -> Figure:
    """Accuracy versus evaluation length, one panel per summary (shared y axis)."""
    fig, axes = plt.subplots(
        1, len(summaries), figsize=(3.4 * len(summaries), 2.8), sharey=True, squeeze=False
    )
    for ax, (title, summary) in zip(axes[0], summaries, strict=True):
        metric = "token_accuracy"
        sets = eval_order(summary)
        xs = [int(s.split("=")[1]) for s in sets]
        for label in variants_in(summary):
            evals = summary["variants"][label]["evals"]
            means = [evals[s][metric]["mean"] * 100 for s in sets]
            stds = [(evals[s][metric]["std"] or 0.0) * 100 for s in sets]
            ax.errorbar(
                xs,
                means,
                yerr=stds,
                color=COLORS.get(label),
                marker=MARKERS.get(label),
                label=LABELS.get(label, label),
                capsize=2,
                elinewidth=1.0,
            )
        _chance(ax, summary.get("chance", {}).get(metric))
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.set_title(title)
        ax.set_xlabel(title.split(":")[0] if ":" in title else "evaluation length")
        _style(ax)
    axes[0][0].set_ylabel(ylabel)
    axes[0][-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig


def _bars(ax: Axes, summary: dict[str, Any], eval_name: str, metric: str) -> None:
    labels = variants_in(summary)
    for i, label in enumerate(labels):
        stats = summary["variants"][label]["evals"][eval_name][metric]
        ax.bar(
            i,
            stats["mean"] * 100,
            width=0.72,
            color=COLORS.get(label),
            yerr=(stats["std"] or 0.0) * 100,
            capsize=2,
            error_kw={"elinewidth": 1.0, "ecolor": INK},
            label=LABELS.get(label, label),
        )
    ax.set_xticks(range(len(labels)), [LABELS.get(v, v) for v in labels], rotation=30, ha="right")
    _chance(ax, summary.get("chance", {}).get(metric))
    _style(ax)


def training_figures(root: Path, out: Path, report_out: Path | None) -> list[Path]:
    written: list[Path] = []
    parity = load_summary(root, "parity")
    if parity is not None:
        fig = _lengths_figure(
            [("Parity: sequence length (trained at 128)", parity)], "token accuracy (%)"
        )
        written += _save(fig, out, report_out, "fig_parity")
    sel = load_summary(root, "selective_copy")
    if sel is not None:
        fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True)
        for ax, name, title in zip(
            axes, ("id", "ood"), ("in distribution", "out of distribution"), strict=True
        ):
            _bars(ax, sel, name, "token_accuracy")
            ax.set_title(f"Selective copy, {title}")
        axes[0].set_ylabel("answer-token accuracy (%)")
        written += _save(fig, out, report_out, "fig_selective_copy")
    fixed, varlen = load_summary(root, "exact_copy"), load_summary(root, "exact_copy_varlen")
    panels = [
        (t, s)
        for t, s in (
            ("String length: trained at 32", fixed),
            ("String length: trained on 1 to 32", varlen),
        )
        if s
    ]
    if panels:
        fig = _lengths_figure(panels, "token accuracy (%)")
        written += _save(fig, out, report_out, "fig_exact_copy")
    imdb = load_summary(root, "imdb")
    if imdb is not None:
        fig, ax = plt.subplots(figsize=(3.4, 2.8))
        _bars(ax, imdb, "test", PRIMARY["imdb"][0])
        ax.set_ylabel("test accuracy (%)")
        ax.set_title("IMDb sentiment")
        written += _save(fig, out, report_out, "fig_imdb")
    return written


def _series(entries: list[dict[str, Any]], key: str) -> dict[str, list[tuple[int, float, bool]]]:
    out: dict[str, list[tuple[int, float, bool]]] = {}
    for e in entries:
        if point_status(e) is not None or key not in e:
            continue
        spilled = bool(e.get("spilled_train") or e.get("spilled"))
        out.setdefault(e["model"], []).append((int(e["seq_len"]), float(e[key]), spilled))
    return {m: sorted(v) for m, v in out.items()}


def _plot_series(ax: Axes, series: dict[str, list[tuple[int, float, bool]]]) -> None:
    for model, points in series.items():
        key = style_key(model)
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        ax.plot(
            xs,
            ys,
            color=COLORS.get(key, INK),
            marker=MARKERS.get(key, "o"),
            linestyle=BACKEND_STYLES.get(model, "-"),
            label=LABELS.get(model, model),
        )
        ringed = [(x, y) for x, y, s in points if s]
        if ringed:
            ax.scatter(
                *zip(*ringed, strict=True),
                s=110,
                facecolors="none",
                edgecolors=INK,
                linewidths=1.0,
                zorder=5,
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _style(ax)


def benchmark_figures(root: Path, out: Path, report_out: Path | None) -> list[Path]:
    written: list[Path] = []
    sweeps = {p: load_sweep(root, "efficiency", p) for p in ("fp32", "bf16")}
    present = [(p, s) for p, s in sweeps.items() if s]
    if present:
        fig, axes = plt.subplots(len(present), 2, figsize=(7.0, 2.7 * len(present)), squeeze=False)
        for row, (precision, sweep) in enumerate(present):
            for col, (key, ylabel) in enumerate(
                (
                    ("train_step_ms_median", "training step (ms)"),
                    ("peak_train_mb", "peak allocated (MiB)"),
                )
            ):
                ax = axes[row][col]
                _plot_series(ax, _series(sweep["entries"], key))
                ax.set_ylabel(ylabel)
                ax.set_title(f"{precision}, batch 8")
                ax.set_xlabel("sequence length T")
            axes[row][1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        written += _save(fig, out, report_out, "fig_efficiency")
    decode = load_sweep(root, "inference", "bf16") or load_sweep(root, "inference", "fp32")
    if decode is not None:
        precision = "bf16" if load_sweep(root, "inference", "bf16") else "fp32"
        fig, axes = plt.subplots(1, 3, figsize=(10.2, 2.8))
        entries = decode["entries"]
        state = [
            dict(e, decode_state_mib=e["decode_state_bytes"] / 1024**2)
            for e in entries
            if "decode_state_bytes" in e
        ]
        panels = (
            (_series(entries, "ttft_ms_median"), "time to first token (ms)"),
            (_series(entries, "decode_ms_per_token_median"), "decode latency per token (ms)"),
            (_series(state, "decode_state_mib"), "decode state (MiB)"),
        )
        for ax, (series, ylabel) in zip(axes, panels, strict=True):
            _plot_series(ax, series)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("prompt length")
            ax.set_title(f"{precision}, batch 1")
        axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        written += _save(fig, out, report_out, "fig_inference")
    return written


def make_all(root: Path, out: Path, report_out: Path | None = None) -> list[Path]:
    return training_figures(root, out, report_out) + benchmark_figures(root, out, report_out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "figures"))
    ap.add_argument("--report-out", default=str(ROOT / "reports" / "figures"))
    args = ap.parse_args()
    written = make_all(Path(args.results), Path(args.out), Path(args.report_out))
    if not written:
        print("no results found; no figures written")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
