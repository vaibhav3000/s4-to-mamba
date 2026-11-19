"""Generate all figures from committed experiment results.

Usage:
    python scripts/make_figures.py [--results results] [--out figures]
                                   [--report-out reports/figures]

Every plot reads the committed JSON outputs (results/*.json and
results/efficiency_gpu.json); nothing is hand-typed. figures/ holds the
README figures; reports/figures/ holds the four PDFs that
reports/Efficient_Sequence_Modeling_S4_to_Mamba3.tex includes. Model colors
and styles are consistent across all figures. Missing measurements are never
drawn as values: out-of-memory points get an x on the bottom edge of the
axes, and points whose training step needed more memory than the GPU has are
ringed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, NullFormatter

if TYPE_CHECKING:
    from matplotlib.typing import RcKeyType

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
# Distinct marker shapes keep coinciding series apart (exact copy: Transformer and
# S4D overlap at every length).
MARKERS = {"transformer": "o", "s4d": "s", "mamba": "^", "mamba2": "D", "mamba3": "v"}
METRIC_COLORS = ("#0a7d33", "#0a5aa0")  # report figures: first and second metric

# Chance levels follow from how src/ssmbench/data/synthetic.py draws targets.
# Parity: targets are cumsum(bits) % 2 of uniform random bits, so each target bit
# is uniform on {0, 1}.
PARITY_CHANCE = 0.5
# Selective copy: tokens 0, 1, 2 are pad, marker and separator; the remembered
# values are randint(3, vocab_size), so there are vocab_size - 3 value tokens.
SELCOPY_FIRST_VALUE_TOKEN = 3
# Exact copy: string letters are randint(0, alphabet_size) + 4, so a uniform guess
# over the alphabet is right 1 / alphabet_size of the time (the eos target, one per
# sequence, is ignored here).

# Memory of the GPU that produced the legacy efficiency entries: the CUDA allocator
# message in experiments.log reports "total: 6438780928" bytes (6140.5 MiB). It is
# used only for entries recorded before the benchmark wrote its own fields below.
LEGACY_DEVICE_TOTAL_BYTES = 6_438_780_928
MIB = 1024**2
# Field names written by src/ssmbench/benchmarks/efficiency.py. Keep these in sync
# with the benchmark.
SPILL_FLAG_KEY = "spilled_train"
DEVICE_TOTAL_KEY = "device_total_mb"
RESERVED_TRAIN_KEY = "peak_train_reserved_mb"
# Results written by the fixed code carry provenance.source_sha256 (P5).
PROVENANCE_KEY = "source_sha256"
STALE_NOTE = "Results from pre-fix code; see results/PROVENANCE.md"

# Every decode_mode string the benchmark has written, and what its number means.
# "stepwise": block.step() per token with carried state (a real decode step).
# "full_forward": one full forward over the prefix per new token (no KV cache).
# "legacy_forward_over_T": the b327a35 Transformer value, forward time / T.
# "prefill_over_T": the parallel forward divided by T (no step() path).
DECODE_KINDS = {
    "stepwise (O(1) state)": "stepwise",
    "full-prefix recompute per new token (no KV cache)": "full_forward",
    "full-prefix recompute (no KV cache)": "legacy_forward_over_T",
    "sequential scan over full sequence (per-token = time/T)": "prefill_over_T",
    "not measured: no step() path; prefill_per_token_ms is one full forward / T": (
        "prefill_over_T"
    ),
}


def _style(ax: Axes, legend_outside: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)
    if legend_outside:
        ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    else:
        ax.legend(frameon=False, fontsize=9)


def _save(fig: Figure, out: Path, name: str, png: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if png:
        fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    # No CreationDate, so re-running on unchanged JSON gives identical PDFs.
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"wrote {out / name}.{'png' if png else 'pdf'}")


def _stamp(fig: Figure, stale: bool) -> None:
    if stale:
        # below the figure edge; bbox_inches="tight" grows the canvas to include it
        fig.text(0.995, -0.01, STALE_NOTE, ha="right", va="top", fontsize=6, color="gray")


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


def _has_provenance(record: dict) -> bool:
    return bool((record.get("provenance") or {}).get(PROVENANCE_KEY))


def _shared(runs: list[dict], *keys: str):
    """A data_config value that every run of one task must share."""
    values = set()
    for run in runs:
        v = run["data_config"]
        for k in keys:
            v = v[k]
        values.add(v)
    if len(values) != 1:
        raise ValueError(f"runs disagree on data_config.{'.'.join(keys)}: {sorted(values)}")
    return values.pop()


def _selcopy_value_tokens(runs: list[dict]) -> int:
    vocab = _shared(runs, "test", "vocab_size")
    return vocab - SELCOPY_FIRST_VALUE_TOKEN


# ------------------------------------------------------------------ efficiency


def _decode_kind(entry: dict) -> str | None:
    mode = entry.get("decode_mode")
    if mode is None:
        return None
    try:
        return DECODE_KINDS[mode]
    except KeyError:
        raise ValueError(
            f"unknown decode_mode {mode!r} for {entry.get('model')} @ T={entry.get('seq_len')}; "
            "add it to DECODE_KINDS in scripts/make_figures.py"
        ) from None


def _device_total_mib(entry: dict) -> float:
    total = entry.get(DEVICE_TOTAL_KEY)
    return float(total) if total is not None else LEGACY_DEVICE_TOTAL_BYTES / MIB


def _spilled(entry: dict) -> bool:
    """True when the training step needed more memory than the GPU has.

    On Windows the WDDM driver then pages to system memory, so the timing no
    longer measures the GPU alone. Uses the benchmark's own flag when present,
    else peak reserved memory against device memory. Legacy entries recorded
    only allocated memory, a lower bound on reserved memory: above device memory
    proves a spill, below it proves nothing.
    """
    flag = entry.get(SPILL_FLAG_KEY)
    if flag is not None:
        return bool(flag)
    peak = entry.get(RESERVED_TRAIN_KEY)
    if peak is None:
        peak = entry.get("peak_train_mb")
    return peak is not None and peak > _device_total_mib(entry)


def _train_oom(entry: dict) -> bool:
    # "oom" is written when the whole benchmark_single call ran out of memory.
    return bool(entry.get("oom_train") or entry.get("oom"))


def _mark_oom(ax: Axes, entries: list[dict], is_oom) -> list:
    """x on the bottom edge at each out-of-memory length, in the model's colour.

    y is in axes units, so the mark never sits on the value scale. Returns the
    legend handle (an empty list when nothing was marked).
    """
    trans = ax.get_xaxis_transform()
    drawn = False
    for model in MODELS:
        for L in sorted({e["seq_len"] for e in entries if e.get("model") == model and is_oom(e)}):
            ax.plot(
                [L],
                [0.03],
                marker="x",
                ms=6,
                mew=1.5,
                ls="none",
                color=COLORS[model],
                transform=trans,
                clip_on=False,
                zorder=5,
            )
            drawn = True
    if not drawn:
        return []
    return ax.plot(
        [],
        [],
        marker="x",
        ms=6,
        mew=1.5,
        ls="none",
        color="black",
        label="out of memory (no value)",
    )


def _ring_spilled(ax: Axes, pts: list[tuple[float, float]], ms: float) -> list:
    if not pts:
        return []
    xs, ys = zip(*pts, strict=True)
    return ax.plot(
        xs,
        ys,
        ls="none",
        marker="o",
        ms=ms,
        mfc="none",
        mec="black",
        mew=1.0,
        zorder=4,
        label="training step exceeded GPU memory",
    )


def _log_length_axis(ax: Axes, lengths: list[int]) -> None:
    ax.set_xscale("log")
    ticks = sorted(set(lengths))
    ax.set_xticks(ticks, labels=[f"{L // 1024}K" if L % 1024 == 0 else str(L) for L in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())


def efficiency_figures(results: Path, out: Path) -> None:
    path = results / "efficiency_gpu.json"
    if not path.exists():
        print("efficiency_gpu.json not found; skipping efficiency figures")
        return
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    stale = not all(_has_provenance(e) for e in entries)
    lengths = [e["seq_len"] for e in entries]
    specs = [
        (
            "train_step_ms",
            "Training step time vs sequence length",
            "train step time (ms)",
            "fig_efficiency_latency",
            True,
        ),
        (
            "peak_train_mb",
            "Peak GPU memory vs sequence length",
            "peak allocated memory, training step (MiB)",
            "fig_efficiency_memory",
            False,
        ),
        (
            "train_tokens_per_s",
            "Training throughput vs sequence length",
            "tokens / s",
            "fig_efficiency_throughput",
            True,
        ),
    ]
    for key, title, ylabel, name, logy in specs:
        fig, ax = plt.subplots(figsize=(6, 4))
        spilled_pts: list[tuple[float, float]] = []
        for model in MODELS:
            rows = sorted(
                (e for e in entries if e.get("model") == model and e.get(key) is not None),
                key=lambda e: e["seq_len"],
            )
            if rows:
                ax.plot(
                    [e["seq_len"] for e in rows],
                    [e[key] for e in rows],
                    marker=MARKERS[model],
                    ms=4,
                    color=COLORS[model],
                    label=LABELS[model],
                )
                spilled_pts += [(e["seq_len"], e[key]) for e in rows if _spilled(e)]
        _ring_spilled(ax, spilled_pts, ms=9)
        if key == "peak_train_mb":
            for total in sorted({_device_total_mib(e) for e in entries if e.get(key) is not None}):
                ax.axhline(total, ls=":", lw=1, color="gray", label=f"GPU memory ({total:.1f} MiB)")
        _mark_oom(ax, entries, _train_oom)
        _log_length_axis(ax, lengths)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("sequence length T (batch 8)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _style(ax, legend_outside=True)
        _stamp(fig, stale)
        _save(fig, out, name)
    _decode_figure(entries, out, stale, lengths)


def _decode_figure(entries: list[dict], out: Path, stale: bool, lengths: list[int]) -> None:
    """Left: real per-token generation cost. Right: parallel forward time / T for every model."""
    step: dict[str, list] = defaultdict(list)
    full: dict[str, list] = defaultdict(list)
    prefill: dict[str, list] = defaultdict(list)
    left_out: set[str] = set()
    for e in entries:
        model, L = str(e["model"]), int(e["seq_len"])
        kind = _decode_kind(e)
        v = e.get("decode_per_token_ms")
        if v is not None and kind == "stepwise":
            step[model].append((L, v))
        elif v is not None and kind == "full_forward":
            full[model].append((L, v))
        elif kind in ("legacy_forward_over_T", "prefill_over_T"):
            left_out.add(model)
        if e.get("infer_ms") is not None:
            prefill[model].append((L, e["infer_ms"] / L))
    fig, (ax_dec, ax_pre) = plt.subplots(1, 2, figsize=(11, 4))
    for model in MODELS:
        for pts, suffix, ls in (
            (step[model], "step(), carried state", "-"),
            (full[model], "no KV cache, full forward per token", "--"),
        ):
            if pts:
                xs, ys = zip(*sorted(pts), strict=True)
                ax_dec.plot(
                    xs,
                    ys,
                    ls=ls,
                    marker=MARKERS[model],
                    ms=4,
                    color=COLORS[model],
                    label=f"{LABELS[model]} ({suffix})",
                )
        if prefill[model]:
            xs, ys = zip(*sorted(prefill[model]), strict=True)
            ax_pre.plot(
                xs, ys, marker=MARKERS[model], ms=4, color=COLORS[model], label=LABELS[model]
            )
    _mark_oom(ax_dec, entries, lambda e: e.get("oom_decode"))
    _mark_oom(ax_pre, entries, lambda e: e.get("oom_infer"))
    note = ""
    if left_out:
        names = ", ".join(LABELS[m] for m in MODELS if m in left_out)
        note = (
            f"Left panel omits {names}: their decode_per_token_ms is a forward time / T, "
            "not a decode step."
        )
    ax_dec.set_title("Generation cost per new token (batch 1)", fontsize=10)
    ax_dec.set_ylabel("ms per generated token")
    ax_pre.set_title("Parallel forward time / T (infer_ms / T, batch 1)", fontsize=10)
    ax_pre.set_ylabel("ms per token")
    for ax in (ax_dec, ax_pre):
        _log_length_axis(ax, lengths)
        lo, hi = ax.dataLim.intervaly
        if lo > 0 and hi / lo > 20:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel("sequence length T")
        _style(ax)
    fig.tight_layout(rect=(0, 0.06 if note else 0, 1, 1))
    fig.text(0.01, 0.005, note, fontsize=7, color="gray", va="bottom")
    _stamp(fig, stale)
    _save(fig, out, "fig_efficiency_decode")


# ------------------------------------------------------------------ training tasks


def training_figures(results: Path, out: Path) -> None:
    runs = _load_runs(results)
    if not runs:
        print("no training results found")
        return
    stale = not all(_has_provenance(r) for r in runs.values())

    def task_title(prefix: str, task_runs: list[dict]) -> str:
        if prefix == "selective_copy":
            tr = lambda k: _shared(task_runs, "train", k)  # noqa: E731
            te = lambda k: _shared(task_runs, "test", k)  # noqa: E731
            return (
                f"Selective copying, test L={te('seq_len')} with {te('min_pairs')}-{te('max_pairs')} pairs\n"
                f"(trained on L={tr('seq_len')} with {tr('min_pairs')}-{tr('max_pairs')} pairs; final epoch)"
            )
        if prefix == "parity":
            return (
                f"Parity (cumulative XOR), train and test L={_shared(task_runs, 'test', 'seq_len')}"
                " (final epoch)"
            )
        return "IMDb sentiment (final epoch, 25k test split)"

    task_specs = [
        ("imdb", [("accuracy", "accuracy"), ("auc", "AUC")], "fig_imdb_quality"),
        (
            "selective_copy",
            [("sequence_success", "sequence success"), ("token_accuracy", "token accuracy")],
            "fig_selective_copy",
        ),
        ("parity", [("token_accuracy", "token accuracy")], "fig_parity"),
    ]
    for prefix, metrics, name in task_specs:
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
            bars = ax.bar(
                xs,
                ys,
                width=width * 0.9,
                label=label,
                color=plt.cm.viridis(mi / max(len(metrics) - 1, 1)),
            )
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.set_xticks([i + width * (len(metrics) - 1) / 2 for i in range(len(models_present))])
        ax.set_xticklabels([LABELS.get(m, m) for m in models_present], fontsize=9)
        ax.set_title(task_title(prefix, list(task_runs.values())), fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        chance = None
        if prefix == "parity":
            chance = (PARITY_CHANCE, "chance 0.5")
        elif prefix == "selective_copy":
            n_values = _selcopy_value_tokens(list(task_runs.values()))
            chance = (1 / n_values, f"token-accuracy chance 1/{n_values}")
        if chance is not None:
            ax.axhline(chance[0], ls="--", lw=1, color="gray", label=chance[1])
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        _stamp(fig, stale)
        _save(fig, out, name)

    # learning curves 2x2: per-epoch evaluation, which the trainer runs on the test split
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        ("imdb", "accuracy", "IMDb accuracy (test split)"),
        (
            "selective_copy",
            "sequence_success",
            "Selective copy: sequence success (test L=128, 5-10 pairs)",
        ),
        ("parity", "token_accuracy", "Parity: token accuracy (test split)"),
        ("imdb", "train_loss", "IMDb train loss"),
    ]
    for ax, (prefix, key, title) in zip(axes.flat, panels, strict=True):
        for run_name, run in sorted(runs.items()):
            if not run_name.startswith(prefix):
                continue
            model = _model_of(run_name)
            xs = [e.get("epoch") for e in run["history"]]
            ys = [e.get(key) for e in run["history"]]
            if any(y is None for y in ys):
                continue
            ax.plot(xs, ys, marker=MARKERS[model], ms=3, color=COLORS[model], label=LABELS[model])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _style(ax)
    by_label: dict[str, Artist] = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            by_label.setdefault(lab, h)
    order = [LABELS[m] for m in MODELS if LABELS[m] in by_label]
    fig.legend(
        [by_label[lab] for lab in order],
        order,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _stamp(fig, stale)
    _save(fig, out, "fig_learning_curves")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--report-out", default="reports/figures")
    args = ap.parse_args()
    results, out = Path(args.results), Path(args.out)
    if not results.exists():
        print(f"results directory '{results}' not found; nothing to do")
        return
    training_figures(results, out)
    efficiency_figures(results, out)
    exact_copy_figures(results, out)
    task_formulation_figure(results, out)
    report_figures(results, Path(args.report_out))


def exact_copy_figures(results: Path, out: Path) -> None:
    """Repeat-After-Me reproduction figures: metrics vs string length per model."""
    runs = {}
    for model in MODELS:
        p = results / f"exact_copy_{model}.json"
        if p.exists():
            runs[model] = json.loads(p.read_text(encoding="utf-8"))
    if not runs:
        print("no exact_copy results; skipping")
        return
    run_list = list(runs.values())
    alphabet = _shared(run_list, "alphabet_size")
    train_len = _shared(run_list, "train", "max_len")
    stale = not all(_has_provenance(r) for r in run_list)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title, ylabel in [
        (
            axes[0],
            "token_accuracy",
            "Exact copy: token accuracy vs string length",
            "token accuracy",
        ),
        (
            axes[1],
            "sequence_success",
            "Exact copy: full-sequence success vs length",
            "sequence success",
        ),
    ]:
        lengths: set[int] = set()
        for model, run in runs.items():
            fb = run.get("final_by_length") or {}
            pts = sorted((int(L), v[key]) for L, v in fb.items() if v.get(key) is not None)
            if pts:
                xs, ys = zip(*pts, strict=True)
                lengths.update(xs)
                hollow = model == "transformer"
                ax.plot(
                    xs,
                    ys,
                    ls="--" if hollow else "-",
                    marker=MARKERS[model],
                    ms=6,
                    mfc="none" if hollow else COLORS[model],
                    color=COLORS[model],
                    label=LABELS[model],
                    zorder=3 if hollow else 2,
                )
        if key == "token_accuracy":
            # A whole-sequence guess succeeds with probability about alphabet**-L, so the
            # success panel gets no chance line.
            ax.axhline(1 / alphabet, ls="--", lw=1, color="gray", label=f"chance 1/{alphabet}")
        ax.set_xscale("log", base=2)
        ticks = sorted(lengths)
        ax.set_xticks(ticks, labels=[f"{L}\n(train)" if train_len == L else str(L) for L in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(f"string length (every training string has length {train_len})")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(title, fontsize=10)
        _style(ax)
    fig.tight_layout()
    _stamp(fig, stale)
    _save(fig, out, "fig_exact_copy")


def task_formulation_figure(results: Path, out: Path) -> None:
    """Full-sequence success on the two copy tasks, with their test conditions in the legend."""
    sel, exact = {}, {}
    sel_runs, exact_runs = [], []
    for model in MODELS:
        p1 = results / f"selective_copy_{model}.json"
        if p1.exists():
            r = json.loads(p1.read_text(encoding="utf-8"))
            v = _final(r, "sequence_success")
            if v is not None:
                sel[model] = v
                sel_runs.append(r)
        p3 = results / f"exact_copy_{model}.json"
        if p3.exists():
            r = json.loads(p3.read_text(encoding="utf-8"))
            train_len = r["data_config"]["train"]["max_len"]
            v = (r.get("final_by_length") or {}).get(str(train_len), {}).get("sequence_success")
            if v is not None:
                exact[model] = v
                exact_runs.append(r)
    if not sel or not exact:
        print("missing task results; skipping formulation figure")
        return
    te = lambda k: _shared(sel_runs, "test", k)  # noqa: E731
    tr = lambda k: _shared(sel_runs, "train", k)  # noqa: E731
    ex_len = _shared(exact_runs, "train", "max_len")
    sel_label = (
        f"selective copy: test L={te('seq_len')}, {te('min_pairs')}-{te('max_pairs')} pairs "
        f"(trained on L={tr('seq_len')}, {tr('min_pairs')}-{tr('max_pairs')} pairs)"
    )
    ex_label = f"exact copy: L={ex_len}, the training length"
    models_present = [m for m in MODELS if m in sel or m in exact]
    x = np.arange(len(models_present))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for values, offset, label, color in (
        (sel, -w / 2, sel_label, "#0a7d33"),
        (exact, w / 2, ex_label, "#0a5aa0"),
    ):
        ax.bar(x + offset, [values.get(m, 0) for m in models_present], w, label=label, color=color)
        for xi, m in zip(x, models_present, strict=True):
            # a zero-height bar is invisible: say "0.0" so it is not read as "not run"
            text = "not run" if m not in values else ("0.0" if values[m] == 0 else None)
            if text:
                ax.text(
                    xi + offset,
                    0.02,
                    text,
                    rotation=90,
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    color=color,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in models_present], fontsize=9)
    ax.set_ylabel("full-sequence success")
    ax.set_title(
        "Copy tasks under different test conditions (not a like-for-like comparison)", fontsize=10
    )
    ax.set_ylim(0, 1.08)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _stamp(fig, not all(_has_provenance(r) for r in sel_runs + exact_runs))
    _save(fig, out, "fig_task_formulation")


# ------------------------------------------------------------------ report figures

REPORT_RC: dict[RcKeyType, Any] = {
    "font.size": 8,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
REPORT_LABELS = dict(LABELS, mamba2="Mamba-2")


def _report_axes(page_pt: tuple[float, float], boxes_pt: list[tuple[float, float, float, float]]):
    """Figure sized like the committed PDF page, axes at the committed positions (points)."""
    w, h = page_pt
    fig = plt.figure(figsize=(w / 72, h / 72))
    axes = [
        fig.add_axes((x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)) for x0, y0, x1, y1 in boxes_pt
    ]
    return fig, axes


def report_figures(results: Path, out: Path) -> None:
    """The four PDFs included by reports/Efficient_Sequence_Modeling_S4_to_Mamba3.tex.

    Same data, page size and axes placement as the committed 2026-09-18 PDFs, with
    the chance line, OOM and memory-limit markers of the figures/ versions.
    """
    runs = _load_runs(results)
    eff_path = results / "efficiency_gpu.json"
    if not runs or not eff_path.exists():
        print("missing results; skipping report figures")
        return
    with plt.rc_context(REPORT_RC):
        _report_bars(runs, "parity", [("token_accuracy", None)], "token accuracy", out, "parity")
        _report_bars(
            runs,
            "selective_copy",
            [("token_accuracy", "token accuracy"), ("sequence_success", "sequence success")],
            "accuracy",
            out,
            "selcopy",
        )
        _report_bars(runs, "imdb", [("accuracy", "accuracy"), ("auc", "AUC")], "score", out, "imdb")
        _report_efficiency(json.loads(eff_path.read_text(encoding="utf-8"))["entries"], out)


def _report_bars(runs: dict, prefix: str, metrics: list, ylabel: str, out: Path, name: str) -> None:
    task_runs = {_model_of(rn): r for rn, r in runs.items() if rn.startswith(prefix)}
    models = [m for m in MODELS if m in task_runs]
    box = (
        (44.747, 21.95, 233.007, 159.31) if prefix == "imdb" else (39.653, 37.937, 232.953, 169.312)
    )
    page = (240.207, 169.549) if prefix == "imdb" else (240.153, 176.512)
    fig, (ax,) = _report_axes(page, [box])
    x = np.arange(len(models))
    if len(metrics) == 1:  # one bar per model, coloured by model
        key = metrics[0][0]
        ax.bar(
            x, [_final(task_runs[m], key) for m in models], 0.8, color=[COLORS[m] for m in models]
        )
    else:
        w = 0.38
        for (key, label), offset, color in zip(
            metrics, (-w / 2, w / 2), METRIC_COLORS, strict=True
        ):
            ax.bar(
                x + offset, [_final(task_runs[m], key) for m in models], w, label=label, color=color
            )
    ax.set_xticks(x, labels=[REPORT_LABELS[m] for m in models])
    if prefix != "imdb":
        ax.tick_params(axis="x", labelrotation=20)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)  # bars start at zero (the committed imdb.pdf started at 0.70)
    if prefix == "parity":
        ax.axhline(PARITY_CHANCE, ls="--", lw=1, color="gray")
        ax.text(
            0.98,
            PARITY_CHANCE + 0.015,
            "chance 0.5",
            transform=ax.get_yaxis_transform(),
            ha="right",
            fontsize=7,
            color="gray",
        )
    if prefix == "selective_copy":
        n_values = _selcopy_value_tokens(list(task_runs.values()))
        ax.axhline(
            1 / n_values, ls="--", lw=1, color="gray", label=f"token-acc. chance 1/{n_values}"
        )
    if len(metrics) > 1:
        ax.legend(frameon=False, loc="upper left" if prefix == "selective_copy" else "upper right")
    _save(fig, out, name, png=False)


def _report_efficiency(entries: list[dict], out: Path) -> None:
    fig, axes = _report_axes(
        (499.417, 175.949),
        [(41.919, 34.669, 244.807, 168.749), (289.329, 34.669, 492.217, 168.749)],
    )
    lengths = [e["seq_len"] for e in entries]
    for ax, key, ylabel in (
        (axes[0], "train_step_ms", "train step time (ms)"),
        (axes[1], "peak_train_mb", "peak allocated memory (MiB)"),
    ):
        spilled: list[tuple[float, float]] = []
        for model in MODELS:
            rows = sorted(
                (e for e in entries if e.get("model") == model and e.get(key) is not None),
                key=lambda e: e["seq_len"],
            )
            if rows:
                ax.plot(
                    [e["seq_len"] for e in rows],
                    [e[key] for e in rows],
                    marker=MARKERS[model],
                    ms=3,
                    lw=1.2,
                    color=COLORS[model],
                    label=REPORT_LABELS[model],
                )
                spilled += [(e["seq_len"], e[key]) for e in rows if _spilled(e)]
        marks = _ring_spilled(ax, spilled, ms=7)
        if key == "peak_train_mb":
            for total in sorted({_device_total_mib(e) for e in entries if e.get(key) is not None}):
                marks += [
                    ax.axhline(
                        total, ls=":", lw=1, color="gray", label=f"GPU memory ({total:.1f} MiB)"
                    )
                ]
        marks += _mark_oom(ax, entries, _train_oom)
        _log_length_axis(ax, lengths)
        ax.set_yscale("log")
        ax.set_xlabel("sequence length T")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if key == "peak_train_mb":
            ax.legend(handles=marks, frameon=False, loc="lower right", fontsize=7)
        else:
            ax.legend(
                handles=[h for h in ax.get_lines() if h.get_label() in REPORT_LABELS.values()],
                frameon=False,
                loc="lower right",
            )
    _save(fig, out, "efficiency", png=False)


if __name__ == "__main__":
    main()
