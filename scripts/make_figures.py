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
from matplotlib.patches import FancyBboxPatch
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


# The overview diagram: each stage's header color (the model's palette color, darkened where
# white text sits on it), its subtitle and what it changes, as the code implements it.
STAGES = [
    (
        "S4D",
        "linear time-invariant, ZOH",
        "#15805a",
        "white",
        [
            "complex diagonal A",
            "same kernel, every input",
            "train: FFT convolution",
            "decode: complex state",
        ],
    ),
    (
        "Mamba-1",
        "selective scan",
        "#a86f00",
        "white",
        [
            "Δ, B, C from the input",
            "keeps or forgets tokens",
            "train: sequential scan",
            "decode: state + conv",
        ],
    ),
    (
        "Mamba-2",
        "SSD, chunked",
        "#e87ba4",
        INK,
        [
            "one scalar A per head",
            "train: chunked matmuls",
            "state carried per chunk",
            "decode: (N, P) per head",
        ],
    ),
    (
        "Mamba-3",
        "as implemented here",
        "#006b00",
        "white",
        [
            "complex A per head, slot",
            "exp. trapezoidal rule",
            "static θ, real B and C",
            "no MIMO",
            "train: chunked SSD",
            "decode: complex state",
        ],
    ),
]
HARNESS = [
    "training step, peak memory",
    "time to first token",
    "decode latency, tokens/s",
    "bf16, fp16 error vs float64",
    "torch.compile graph breaks",
    "1K to 32K tokens, CUDA events",
]


def _panel(ax: Axes, x: float, y: float, w: float, h: float, fill: str = PAPER) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.01",
            transform=ax.transAxes,
            facecolor=fill,
            edgecolor=MUTED,
            linewidth=0.8,
        )
    )


def _arrow(ax: Axes, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.annotate(
        "",
        (x1, y1),
        xytext=(x0, y0),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.0, "mutation_scale": 9},
    )


def _longest(sweep: dict[str, Any] | None) -> int:
    rows = [e for e in (sweep or {}).get("entries", []) if "decode_state_bytes" in e]
    return max((int(e["seq_len"]) for e in rows), default=0)


def _state_mib(sweep: dict[str, Any] | None, models: set[str] | None) -> list[float]:
    """Decode state in MiB at the sweep's longest prompt, for the given models (None: SSMs)."""
    if sweep is None:
        return []
    rows = [e for e in sweep["entries"] if "decode_state_bytes" in e]
    if not rows:
        return []
    longest = max(int(e["seq_len"]) for e in rows)
    return [
        e["decode_state_bytes"] / 1024**2
        for e in rows
        if int(e["seq_len"]) == longest
        and (e["model"] in models if models else not e["model"].startswith("transformer"))
    ]


def overview_figure(root: Path, out: Path) -> list[Path]:
    """The README hero: the S4D to Mamba-3 evolution above the shared decode interface, the
    decode-state comparison (numbers from the inference sweeps) and what the harness measures."""
    fp32, bf16 = load_sweep(root, "inference", "fp32"), load_sweep(root, "inference", "bf16")
    kv32 = _state_mib(fp32, {"transformer_efficient"})
    kv16 = _state_mib(bf16, {"transformer_efficient"})
    ssm = _state_mib(bf16 or fp32, None)
    if not (kv32 or kv16) or not ssm:
        return []
    fig = plt.figure(figsize=(TEXT_IN, 4.9))
    fig.set_layout_engine("none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    margin, gap = 0.012, 0.028
    width = (1 - 2 * margin - 3 * gap) / 4
    top, bottom, head = 0.925, 0.5, 0.09
    ax.text(margin, 0.965, "Architecture evolution", fontsize=9, fontweight="bold", color=INK)
    for i, (name, subtitle, color, text_color, lines) in enumerate(STAGES):
        x = margin + i * (width + gap)
        _panel(ax, x, bottom, width, top - bottom)
        ax.add_patch(
            FancyBboxPatch(
                (x, top - head),
                width,
                head,
                boxstyle="round,pad=0,rounding_size=0.01",
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
            )
        )
        ax.text(x + 0.01, top - 0.035, name, fontsize=9, fontweight="bold", color=text_color)
        ax.text(x + 0.01, top - 0.075, subtitle, fontsize=7.5, color=text_color)
        for k, line in enumerate(lines):
            ax.text(x + 0.01, top - head - 0.05 - k * 0.052, line, fontsize=8, color=INK)
        if i < len(STAGES) - 1:
            mid = (top - head + bottom) / 2
            _arrow(ax, x + width + 0.003, mid, x + width + gap - 0.003, mid)

    lower_top, lower_bottom = 0.39, 0.025
    ax.text(
        margin,
        0.428,
        "Shared decode interface and benchmark harness",
        fontsize=9,
        fontweight="bold",
        color=INK,
    )
    widths = (0.40, 0.27, 0.27)
    lower_gap = (1 - 2 * margin - sum(widths)) / 2
    xs = [margin, margin + widths[0] + lower_gap, margin + widths[0] + widths[1] + 2 * lower_gap]
    longest = _longest(bf16 or fp32)
    titles = ("Every backbone", f"State at a {longest // 1024}K prompt", "The harness measures")
    for x, w, title in zip(xs, widths, titles, strict=True):
        _panel(ax, x, lower_bottom, w, lower_top - lower_bottom, fill="#f7f7f5")
        ax.text(x + 0.012, lower_top - 0.05, title, fontsize=8.5, fontweight="bold", color=INK)

    # Panel 1: the decode interface as three calls.
    chip_h, chip_y, chip_gap = 0.075, lower_top - 0.175, 0.022
    cx = xs[0] + 0.012
    for j, (call, chip_w) in enumerate(
        (("init_state()", 0.10), ("prefill(prompt)", 0.125), ("step(token)", 0.105))
    ):
        ax.add_patch(
            FancyBboxPatch(
                (cx, chip_y),
                chip_w,
                chip_h,
                boxstyle="round,pad=0,rounding_size=0.008",
                transform=ax.transAxes,
                facecolor=PAPER,
                edgecolor=MUTED,
                linewidth=0.8,
            )
        )
        ax.text(
            cx + chip_w / 2,
            chip_y + chip_h / 2,
            call,
            fontsize=6.8,
            family="DejaVu Sans Mono",
            ha="center",
            va="center",
            color=INK,
        )
        if j < 2:
            _arrow(
                ax,
                cx + chip_w + 0.002,
                chip_y + chip_h / 2,
                cx + chip_w + chip_gap - 0.002,
                chip_y + chip_h / 2,
            )
        cx += chip_w + chip_gap
    for k, line in enumerate(
        (
            "prefill runs the parallel form and returns the",
            "state the recurrence would hold; step adds one",
            "token. prefill + steps = forward (float64 tests)",
        )
    ):
        ax.text(xs[0] + 0.012, chip_y - 0.06 - k * 0.05, line, fontsize=7.5, color=INK)

    # Panel 2: KV cache against the recurrent state, from the committed inference sweeps.
    kv = " / ".join(
        f"{v:.0f} MiB {p}"
        for v, p in ((max(kv32, default=0), "fp32"), (max(kv16, default=0), "bf16"))
        if v
    )
    rows = (
        ("Transformer KV cache", "grows with the prompt", kv),
        (
            "SSM recurrent state",
            "fixed size, O(1) per token",
            f"{min(ssm):.2f} to {max(ssm):.2f} MiB",
        ),
    )
    for k, (label, note, value) in enumerate(rows):
        y = lower_top - 0.11 - k * 0.138
        ax.text(xs[1] + 0.012, y, label, fontsize=8, fontweight="bold", color=INK)
        ax.text(xs[1] + 0.012, y - 0.045, note, fontsize=7.5, color=MUTED)
        ax.text(xs[1] + 0.012, y - 0.09, value, fontsize=8, color=INK)

    # Panel 3: what the benchmark harness records.
    for k, line in enumerate(HARNESS):
        ax.text(xs[2] + 0.012, lower_top - 0.105 - k * 0.046, line, fontsize=7.5, color=INK)
    return _save(fig, out, "fig_overview")


def make_all(root: Path, out: Path) -> list[Path]:
    return training_figures(root, out) + benchmark_figures(root, out) + overview_figure(root, out)


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
