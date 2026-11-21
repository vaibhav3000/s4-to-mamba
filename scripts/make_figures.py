"""Generate every figure from the committed results (docs/adr/0006).

Usage:
    .venv/Scripts/python.exe scripts/make_figures.py [--results results] [--out figures]

Reads only the files listed in ssmbench/analysis/results.py and writes each figure once,
as a PDF and a PNG of the same drawing, to --out. The LaTeX report includes the PDFs from
figures/ and the README shows the PNGs, so both use one source. A figure whose results are
missing is not drawn. Conventions in every figure:

- sized to the report's column (3.49 in) or text width (7.14 in) at 8 pt type, so the
  report includes it at its natural size; TrueType fonts (pdf.fonttype 42) and an opaque
  white background, which keeps the PNGs legible in GitHub's dark mode;
- one fixed color and marker per model variant (a validated colorblind-safe categorical
  palette), and a line style per attention backend or precision for Transformer rows;
- training results: mean over run seeds with error bars of one sample standard deviation,
  and every individual seed as a small point; the chance level as a dashed gray line;
- benchmarks: medians; points that spilled to system memory are ringed; points with no
  timing (out of memory, skipped, unsupported kernel) are not drawn and are listed in the
  generated tables. Benchmark legends give each model's backbone parameter count, because
  the benchmark models have equal width, not equal size.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

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
INK, MUTED, GRID, PAPER = "#1a1a1a", "#4a4a47", "#dedcd6", "#ffffff"
# IEEEtran conference: \columnwidth = 252 pt, \textwidth = 516 pt (1 in = 72.27 pt).
COLUMN_IN, TEXT_IN = 252 / 72.27, 516 / 72.27

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "savefig.facecolor": PAPER,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _style(ax: Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def _save(fig: Figure, out: Path, name: str) -> list[Path]:
    """Write one drawing as PDF and PNG at exactly the figure size (no tight bbox)."""
    out.mkdir(parents=True, exist_ok=True)
    written = [out / f"{name}.png", out / f"{name}.pdf"]
    fig.savefig(written[0], dpi=300)
    # No CreationDate, so re-running on unchanged JSON gives identical PDFs.
    fig.savefig(written[1], metadata={"CreationDate": None})
    plt.close(fig)
    return written


def _chance(ax: Axes, value: float | None, scale: float = 100.0, label: bool = True) -> None:
    """Dashed chance line; labelled at the right end inside the axes, or (label=False) named
    in the legend instead, where the label would sit on top of the data."""
    if value is not None:
        ax.axhline(
            value * scale,
            color=MUTED,
            linestyle=(0, (4, 3)),
            linewidth=0.8,
            zorder=0,
            label=None if label else f"chance ({value * scale:.1f}%)",
        )
    if value is not None and label:
        ax.annotate(
            "chance",
            (1.0, value * scale),
            xycoords=("axes fraction", "data"),
            xytext=(-2, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color=MUTED,
            fontsize=6.5,
        )


def _legend_below(fig: Figure, ax: Axes, ncol: int) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, handlelength=2.2)


def _lengths_figure(
    summaries: list[tuple[str, dict[str, Any]]], width: float, height: float, ncol: int
) -> Figure:
    """Accuracy versus evaluation length, one panel per summary (shared y axis): mean with
    one sample std, and each seed as a small point, offset slightly per variant."""
    fig, axes = plt.subplots(1, len(summaries), figsize=(width, height), sharey=True, squeeze=False)
    for ax, (title, summary) in zip(axes[0], summaries, strict=True):
        metric = "token_accuracy"
        sets = eval_order(summary)
        xs = [int(s.split("=")[1]) for s in sets]
        labels = variants_in(summary)
        for i, label in enumerate(labels):
            shift = 2 ** ((i - (len(labels) - 1) / 2) * 0.045)
            px = [x * shift for x in xs]
            evals = summary["variants"][label]["evals"]
            means = [evals[s][metric]["mean"] * 100 for s in sets]
            stds = [(evals[s][metric]["std"] or 0.0) * 100 for s in sets]
            color = COLORS.get(label)
            for x, s in zip(px, sets, strict=True):
                values = [v * 100 for v in evals[s][metric].get("values", [])]
                ax.scatter(
                    [x] * len(values), values, s=5, color=color, alpha=0.45, linewidths=0, zorder=2
                )
            ax.errorbar(
                px,
                means,
                yerr=stds,
                color=color,
                marker=MARKERS.get(label),
                label=LABELS.get(label, label),
                capsize=1.5,
                elinewidth=0.8,
                capthick=0.8,
                zorder=3,
            )
        _chance(ax, summary.get("chance", {}).get(metric), label=False)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel("test length")
        _style(ax)
    axes[0][0].set_ylabel("token accuracy (%)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    seeds = Line2D([], [], linestyle="none", marker="o", markersize=3, color=MUTED, alpha=0.6)
    fig.legend(
        [*handles, seeds],
        [*labels, "one run seed"],
        loc="outside lower center",
        ncol=ncol,
        handlelength=2.2,
    )
    return fig


def _bars(ax: Axes, summary: dict[str, Any], eval_name: str, metric: str) -> None:
    labels = variants_in(summary)
    for i, label in enumerate(labels):
        stats = summary["variants"][label]["evals"][eval_name][metric]
        ax.bar(
            i,
            stats["mean"] * 100,
            width=0.7,
            color=COLORS.get(label),
            yerr=(stats["std"] or 0.0) * 100,
            capsize=1.5,
            error_kw={"elinewidth": 0.8, "ecolor": INK, "capthick": 0.8},
            label=LABELS.get(label, label),
            zorder=2,
        )
        values = [v * 100 for v in stats.get("values", [])]
        ax.scatter([i] * len(values), values, s=6, color=INK, alpha=0.6, linewidths=0, zorder=3)
    ax.set_xticks(range(len(labels)), [LABELS.get(v, v) for v in labels], rotation=30, ha="right")
    _chance(ax, summary.get("chance", {}).get(metric), label=False)
    _style(ax)


def _bar_legend(fig: Figure, ax: Axes) -> None:
    """The chance line and the seed points, below the panels (bars are named on the axis)."""
    pairs = [
        (h, lab)
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True)
        if lab.startswith("chance")
    ]
    handles, labels = [h for h, _ in pairs], [lab for _, lab in pairs]
    seeds = Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=3,
        color=INK,
        alpha=0.6,
        label="one run seed",
    )
    fig.legend([*handles, seeds], [*labels, "one run seed"], loc="outside lower center", ncol=2)


def training_figures(root: Path, out: Path) -> list[Path]:
    written: list[Path] = []
    parity = load_summary(root, "parity")
    if parity is not None:
        fig = _lengths_figure([("Parity, trained at length 128", parity)], COLUMN_IN, 3.0, 2)
        written += _save(fig, out, "fig_parity")
    sel = load_summary(root, "selective_copy")
    if sel is not None:
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.8), sharey=True)
        for ax, name, title in zip(
            axes, ("id", "ood"), ("in distribution", "out of distribution"), strict=True
        ):
            _bars(ax, sel, name, "token_accuracy")
            ax.set_title(f"Selective copy, {title}")
        axes[0].set_ylabel("answer-token accuracy (%)")
        _bar_legend(fig, axes[0])
        written += _save(fig, out, "fig_selective_copy")
    fixed, varlen = load_summary(root, "exact_copy"), load_summary(root, "exact_copy_varlen")
    panels = [
        (t, s)
        for t, s in (
            ("Exact copy, trained at length 32", fixed),
            ("Exact copy, trained on lengths 1 to 32", varlen),
        )
        if s
    ]
    if panels:
        fig = _lengths_figure(panels, TEXT_IN, 2.8, 4)
        written += _save(fig, out, "fig_exact_copy")
    imdb = load_summary(root, "imdb")
    if imdb is not None:
        fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.7))
        _bars(ax, imdb, "test", PRIMARY["imdb"][0])
        ax.set_ylabel("test accuracy (%)")
        ax.set_title("IMDb sentiment")
        _bar_legend(fig, ax)
        written += _save(fig, out, "fig_imdb")
    return written


def _size(n: int) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1_000_000 else f"{n / 1e3:.0f}K"


def _params(entries: list[dict[str, Any]]) -> dict[str, int]:
    params: dict[str, int] = {}
    for e in entries:
        if "n_parameters" in e:
            params.setdefault(e["model"], int(e["n_parameters"]))
    return params


def _series(entries: list[dict[str, Any]], key: str) -> dict[str, list[tuple[int, float, bool]]]:
    out: dict[str, list[tuple[int, float, bool]]] = {}
    for e in entries:
        if point_status(e) is not None or key not in e:
            continue
        spilled = bool(e.get("spilled_train") or e.get("spilled"))
        out.setdefault(e["model"], []).append((int(e["seq_len"]), float(e[key]), spilled))
    return {m: sorted(v) for m, v in out.items()}


def _plot_series(
    ax: Axes,
    series: dict[str, list[tuple[int, float, bool]]],
    params: dict[str, int],
    linestyle: Any = None,
    suffix: str = "",
    names: dict[str, str] | None = None,
) -> None:
    lengths: set[int] = set()
    for model, points in series.items():
        key = style_key(model)
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        lengths.update(xs)
        label = (names or {}).get(model, LABELS.get(model, model)) + suffix
        if model in params:
            label += f" ({_size(params[model])})"
        ax.plot(
            xs,
            ys,
            color=COLORS.get(key, INK),
            marker=MARKERS.get(key, "o"),
            linestyle=linestyle or BACKEND_STYLES.get(model, "-"),
            label=label,
        )
        ringed = [(x, y) for x, y, s in points if s]
        if ringed:
            ax.scatter(
                *zip(*ringed, strict=True),
                s=60,
                facecolors="none",
                edgecolors=INK,
                linewidths=0.8,
                zorder=5,
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    if lengths:
        ticks = sorted(lengths | _known_ticks(ax))
        ax.set_xticks(ticks, [f"{x // 1024}K" if x >= 1024 else str(x) for x in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    _style(ax)


def _known_ticks(ax: Axes) -> set[int]:
    """Lengths already plotted on this axes by earlier series."""
    found: set[int] = set()
    for line in ax.get_lines():
        found.update(int(x) for x in np.asarray(line.get_xdata(), dtype=float) if x >= 1)
    return found


def _merged_legend(fig: Figure, axes: list[Axes], ncol: int, extra: list[Line2D]) -> None:
    seen: dict[str, Any] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(label, handle)
    for handle in extra:
        seen.setdefault(str(handle.get_label()), handle)
    fig.legend(list(seen.values()), list(seen.keys()), loc="outside lower center", ncol=ncol)


def _spill_handle() -> Line2D:
    return Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=8,
        markerfacecolor="none",
        markeredgecolor=INK,
        label="spilled to system memory",
    )


def benchmark_figures(root: Path, out: Path) -> list[Path]:
    written: list[Path] = []
    sweeps = {p: load_sweep(root, "efficiency", p) for p in ("fp32", "bf16")}
    present = [(p, s) for p, s in sweeps.items() if s]
    if present:
        fig, axes = plt.subplots(
            len(present), 2, figsize=(TEXT_IN, 2.1 * len(present) + 0.9), squeeze=False
        )
        for row, (precision, sweep) in enumerate(present):
            params = _params(sweep["entries"])
            for col, (key, ylabel) in enumerate(
                (
                    ("train_step_ms_median", "training step (ms)"),
                    ("peak_train_mb", "peak allocated (MiB)"),
                )
            ):
                ax = axes[row][col]
                _plot_series(ax, _series(sweep["entries"], key), params)
                ax.set_ylabel(ylabel)
                ax.set_title(f"{precision}, batch 8")
                ax.set_xlabel("sequence length T")
        _merged_legend(fig, [a for row in axes for a in row], 3, [_spill_handle()])
        written += _save(fig, out, "fig_efficiency")
    bf16, fp32 = load_sweep(root, "inference", "bf16"), load_sweep(root, "inference", "fp32")
    decode = bf16 or fp32
    if decode is not None:
        precision = "bf16" if bf16 else "fp32"
        entries = decode["entries"]
        params = _params(entries)
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_IN, 2.9))
        panels = (
            (_series(entries, "ttft_ms_median"), "time to first token (ms)"),
            (_series(entries, "decode_ms_per_token_median"), "decode latency per token (ms)"),
        )
        for ax, (series, ylabel) in zip(axes, panels, strict=True):
            _plot_series(ax, series, params)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("prompt length")
            ax.set_title(f"{precision}, batch 1, 64 new tokens")
        _merged_legend(fig, list(axes), 3, [])
        written += _save(fig, out, "fig_inference")
    sweeps_inf = [(p, s) for p, s in (("fp32", fp32), ("bf16", bf16)) if s]
    if sweeps_inf:
        fig, ax = plt.subplots(figsize=(COLUMN_IN, 3.1))
        for precision, sweep in sweeps_inf:
            state = [
                dict(e, decode_state_mib=e["decode_state_bytes"] / 1024**2)
                for e in sweep["entries"]
                if "decode_state_bytes" in e
            ]
            series = _series(state, "decode_state_mib")
            # The KV cache depends on the precision; every SSM state is kept in float32, so
            # the SSM rows come from one sweep only (the bf16 one when present).
            kv = {m: v for m, v in series.items() if m == "transformer_efficient"}
            ssm = {m: v for m, v in series.items() if not m.startswith("transformer")}
            _plot_series(
                ax,
                {"transformer": kv["transformer_efficient"]} if kv else {},
                {},
                linestyle="-" if precision == "fp32" else (0, (4, 2)),
                suffix=f" KV cache, {precision}",
                names={"transformer": "Transformer"},
            )
            if precision == sweeps_inf[-1][0]:
                _plot_series(ax, ssm, {}, suffix=" state")
        ax.set_ylabel("decode state (MiB)")
        ax.set_xlabel("prompt length")
        ax.set_title("Decode memory, batch 1")
        _legend_below(fig, ax, 2)
        written += _save(fig, out, "fig_decode_memory")
    return written


def make_all(root: Path, out: Path) -> list[Path]:
    return training_figures(root, out) + benchmark_figures(root, out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "figures"))
    args = ap.parse_args()
    written = make_all(Path(args.results), Path(args.out))
    if not written:
        print("no results found; no figures written")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
