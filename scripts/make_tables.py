"""Generate every results table from the committed results (docs/adr/0006).

Writes:
    reports/tables/<name>.tex          booktabs tables the LaTeX report inputs
    results/SUMMARY.md                 every table in Markdown
    README.md                          the blocks between <!-- generated:NAME --> and
                                       <!-- /generated:NAME --> markers

A table whose results are missing says "Pending: not yet run on the GPU" instead of
showing numbers. With --check nothing is written; the command fails if any generated
file differs from what is committed (tests/test_report_assets.py runs the same check).

Usage:
    .venv/Scripts/python.exe scripts/make_tables.py [--check]
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ssmbench.analysis.results import (
    LABELS,
    PRIMARY,
    eval_order,
    load_json,
    load_summary,
    load_sweep,
    mean_std,
    point_status,
    variants_in,
)

ROOT = Path(__file__).resolve().parents[1]
PENDING = "Pending: not yet run on the GPU."

# A table is a header row, body rows and a note; both renderers read the same Table.
Table = tuple[list[str], list[list[str]], str]


def _training(config: str) -> Callable[[Path], Table | None]:
    def build(root: Path) -> Table | None:
        summary = load_summary(root, config)
        if summary is None:
            return None
        metric, metric_name = PRIMARY[config]
        sets = eval_order(summary)
        header = ["Model", "Parameters", *sets]
        rows = []
        seeds = set()
        for label in variants_in(summary):
            variant = summary["variants"][label]
            seeds.add(len(variant["seeds"]))
            cells = [mean_std(variant["evals"].get(s, {}).get(metric)) for s in sets]
            rows.append([LABELS.get(label, label), f"{variant['n_parameters']:,}", *cells])
        chance = summary.get("chance", {}).get(metric)
        n = "/".join(str(k) for k in sorted(seeds))
        note = f"{metric_name} in percent, mean ± sample std over n = {n} run seeds"
        if chance is not None:
            note += f"; chance {chance * 100:.1f}"
        if config == "imdb":
            note += "; the single test set is the 25k IMDb test split"
        return header, rows, note + "."

    return build


def _sweep_table(kind: str, precision: str) -> Callable[[Path], Table | None]:
    def build(root: Path) -> Table | None:
        sweep = load_sweep(root, kind, precision)
        if sweep is None:
            return None
        entries = sweep["entries"]
        lengths = sorted({int(e["seq_len"]) for e in entries})
        labels: list[str] = []
        for e in entries:
            if e["model"] not in labels:
                labels.append(e["model"])
        cells: dict[tuple[str, int], str] = {}
        for e in entries:
            status = point_status(e)
            if status is not None:
                text = status
            elif kind == "efficiency":
                text = f"{e['train_step_ms_median']:.1f} / {e['peak_train_mb']:.0f}"
                if e.get("spilled_train"):
                    text += " (spilled)"
            else:
                text = f"{e['ttft_ms_median']:.1f} / {e['decode_ms_per_token_median']:.2f}"
            cells[(e["model"], int(e["seq_len"]))] = text
        header = ["Model", *(f"T={L}" for L in lengths)]
        rows = [[LABELS.get(m, m), *(cells.get((m, L), "-") for L in lengths)] for m in labels]
        if kind == "efficiency":
            note = (
                f"{precision}, batch 8: median training-step time in ms / peak allocated MiB. "
                "spilled: the step ran past free GPU memory (WDDM paging, not a VRAM number); "
                "skipped: not run after a shorter length spilled; unsupported: the pinned "
                "attention kernel refused the inputs; -: not in the sweep."
            )
        else:
            note = (
                f"{precision}, batch 1: median time to first token in ms / median decode "
                "latency per token in ms, prompt length T, 64 new tokens. Decode state at the "
                "longest measured prompt: "
                + "; ".join(f"{LABELS.get(m, m)} {_state_mib(entries, m)}" for m in labels)
                + "."
            )
        return header, rows, note

    return build


def _state_mib(entries: list[dict[str, Any]], model: str) -> str:
    measured = [e for e in entries if e["model"] == model and "decode_state_bytes" in e]
    if not measured:
        return "-"
    last = max(measured, key=lambda e: int(e["seq_len"]))
    return f"{last['decode_state_bytes'] / 1024**2:.2f} MiB at T={last['seq_len']}"


def _sci(values: list[float]) -> str:
    return f"{statistics.fmean(values):.1e}" if values else "-"


def _numerics(root: Path) -> Table | None:
    data = load_json(root / "numerics" / "numerics.json")
    if data is None:
        return None
    entries = data["entries"]
    longest = max(int(e["seq_len"]) for e in entries)
    labels: list[str] = []
    for e in entries:
        if e["model"] not in labels:
            labels.append(e["model"])
    header = ["Model", "fp32 vs fp64", "bf16 vs fp64", "fp16 vs fp64", "bf16 decode, max"]
    rows = []
    for m in labels:
        at = [e for e in entries if e["model"] == m and int(e["seq_len"]) == longest]
        fwd = [e["forward"] for e in at]
        drift = [
            e["decode"]["bf16"]["max_rel_l2"] for e in entries if e["model"] == m and "decode" in e
        ]
        rows.append(
            [
                LABELS.get(m, m),
                *(
                    _sci([f[k]["rel_l2"] for f in fwd])
                    for k in ("fp32_vs_fp64", "bf16_vs_fp64", "fp16_vs_fp64")
                ),
                _sci(drift),
            ]
        )
    seeds = len({e["seed"] for e in entries})
    note = (
        f"Relative L2 error of the backbone output against float64 at T={longest}, mean over "
        f"{seeds} initialization seeds; the last column is the largest per-step error of a bf16 "
        "decode against the float64 forward pass."
    )
    return header, rows, note


def _compile(root: Path) -> Table | None:
    data = load_json(root / "compile" / "compile.json")
    if data is None:
        return None
    header = [
        "Model",
        "Forward graphs / breaks",
        "Step graphs / breaks",
        "aot_eager max diff",
        "Inductor",
    ]
    rows = []
    for e in data["entries"]:
        f, s = e["forward"], e["step"]
        rows.append(
            [
                LABELS.get(e["model"], e["model"]),
                f"{f['graph_count']} / {f['graph_break_count']}",
                f"{s['graph_count']} / {s['graph_break_count']}",
                f"{max(e['aot_eager'].values()):.1e}",
                e["inductor"]["status"],
            ]
        )
    note = "torch.compile readiness: Dynamo graphs and graph breaks, and the largest output or gradient difference of aot_eager against eager."
    return header, rows, note


TABLES: dict[str, Callable[[Path], Table | None]] = {
    "parity": _training("parity"),
    "selective_copy": _training("selective_copy"),
    "exact_copy": _training("exact_copy"),
    "exact_copy_varlen": _training("exact_copy_varlen"),
    "imdb": _training("imdb"),
    "efficiency_fp32": _sweep_table("efficiency", "fp32"),
    "efficiency_bf16": _sweep_table("efficiency", "bf16"),
    "inference_fp32": _sweep_table("inference", "fp32"),
    "inference_bf16": _sweep_table("inference", "bf16"),
    "numerics": _numerics,
    "compile": _compile,
}


def _md(table: Table | None) -> str:
    if table is None:
        return f"_{PENDING}_\n"
    header, rows, note = table
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
        "",
        note,
    ]
    return "\n".join(lines) + "\n"


def _tex_escape(text: str) -> str:
    text = text.replace("\\", "\\textbackslash{}")
    for char in "&%$#_{}":
        text = text.replace(char, "\\" + char)
    return text.replace("±", "$\\pm$").replace("~", "\\textasciitilde{}")


def _tex(table: Table | None) -> str:
    if table is None:
        return f"\\emph{{{PENDING}}}\n"
    header, rows, note = table
    cols = "l" + "r" * (len(header) - 1)
    out = [
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(_tex_escape(h) for h in header) + " \\\\",
        "\\midrule",
        *(" & ".join(_tex_escape(c) for c in r) + " \\\\" for r in rows),
        "\\bottomrule",
        "\\end{tabular}",
        f"\\par\\smallskip{{\\footnotesize {_tex_escape(note)}}}",
    ]
    return "\n".join(out) + "\n"


def render_markdown(root: Path) -> dict[str, str]:
    return {name: _md(build(root)) for name, build in TABLES.items()}


def render_latex(root: Path) -> dict[str, str]:
    return {name: _tex(build(root)) for name, build in TABLES.items()}


_BLOCK = re.compile(r"(<!-- generated:(\w+) -->\n)(.*?)(<!-- /generated:\2 -->)", re.DOTALL)


def replace_blocks(text: str, blocks: dict[str, str]) -> str:
    """Replace the body of every <!-- generated:NAME --> block whose NAME is in blocks."""

    def sub(m: re.Match[str]) -> str:
        name = m.group(2)
        body = blocks.get(name, m.group(3))
        return m.group(1) + body + m.group(4)

    return _BLOCK.sub(sub, text)


def _summary_md(md: dict[str, str]) -> str:
    parts = [
        "# Results summary",
        "",
        "Generated by scripts/make_tables.py from the files under results/; do not edit.",
        "",
    ]
    for name, body in md.items():
        parts += [f"## {name}", "", body]
    return "\n".join(parts)


def _outputs(root: Path, repo: Path) -> dict[Path, str]:
    md = render_markdown(root)
    tex = render_latex(root)
    outputs = {repo / "reports" / "tables" / f"{name}.tex": body for name, body in tex.items()}
    outputs[repo / "results" / "SUMMARY.md"] = _summary_md(md)
    readme = repo / "README.md"
    if readme.exists():
        outputs[readme] = replace_blocks(readme.read_text(encoding="utf-8"), md)
    return outputs


def write_all(root: Path, repo: Path) -> None:
    for path, text in _outputs(root, repo).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(repo)}")


def check_all(root: Path, repo: Path) -> list[str]:
    """Generated files that differ from what is on disk (relative paths)."""
    stale = []
    for path, text in _outputs(root, repo).items():
        current = path.read_text(encoding="utf-8").replace("\r\n", "\n") if path.exists() else None
        if current != text:
            stale.append(path.relative_to(repo).as_posix())
    return stale


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        stale = check_all(ROOT / "results", ROOT)
        if stale:
            raise SystemExit("stale generated files: " + ", ".join(stale))
        print("generated tables are up to date")
        return
    write_all(ROOT / "results", ROOT)


if __name__ == "__main__":
    main()
