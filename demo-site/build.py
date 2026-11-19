#!/usr/bin/env python
"""Build demo-site/data.js from the committed results/ JSONs.

    python demo-site/build.py        # from the repo root

Everything the explorer renders is a committed measurement; nothing is
hand-typed. Re-run after regenerating results. Decode labels, spill flags and
pre-fix markers are derived from the committed entries; no number is changed.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MODELS = ["transformer", "s4d", "mamba", "mamba2", "mamba3"]
LABELS = {
    "transformer": "Transformer",
    "s4d": "S4D (LTI)",
    "mamba": "Mamba-1",
    "mamba2": "Mamba-2 (SSD)",
    "mamba3": "Mamba-3 (complex)",
}


# GPU memory reported by the CUDA allocator during the committed sweep
# (experiments.log:209, "total: 6438780928" bytes = 6140.5 MiB). peak_train_mb
# is torch.cuda.max_memory_allocated() / 1024**2, so it is in MiB as well.
DEVICE_TOTAL_MIB = 6438780928 / 1024**2

# Keys written into each entry by the fixed benchmark. Entries without the
# provenance key were produced by the pre-fix code.
PROVENANCE_KEY = "provenance"
SPILL_KEY = "spilled_train"

# What the decode_mode strings written by the pre-fix benchmark actually measure.
# Both are one batch-1 forward over the whole sequence divided by T.
LEGACY_DECODE_LABELS = {
    # Transformer entries come from b327a35, which stored forward_ms / T, not the
    # full forward per new token that the label suggests.
    "full-prefix recompute (no KV cache)": "prefill / T (no KV cache)",
    # mamba2/mamba3 have no step(); mamba2 ran its chunked form, not the sequential scan.
    "sequential scan over full sequence (per-token = time/T)": "prefill / T (no step() path)",
    # Written by the fixed benchmark, which stores the value as prefill_per_token_ms.
    "not measured: no step() path; prefill_per_token_ms is one full forward / T": (
        "prefill / T (no step() path)"
    ),
}


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def final(run: dict, key: str):
    for entry in reversed(run.get("history", [])):
        if entry.get(key) is not None:
            return entry[key]
    return None


def task_results(prefix: str, keys: list[str]) -> dict:
    out = {}
    for model in MODELS:
        path = RESULTS / f"{prefix}_{model}.json"
        if path.exists():
            run = load(path.name)
            out[model] = {k: final(run, k) for k in keys}
    return out


def exact_copy_in_dist() -> dict:
    out = {}
    for model in MODELS:
        path = RESULTS / f"exact_copy_{model}.json"
        if path.exists():
            run = load(path.name)
            by_len = run.get("final_by_length") or {}
            in_dist = by_len.get("32", {})
            out[model] = {
                "sequence_success": in_dist.get("sequence_success"),
                "token_accuracy": in_dist.get("token_accuracy"),
            }
    return out


def decode_label(entry: dict) -> str | None:
    """Describe what decode_per_token_ms measures for this entry."""
    mode = entry.get("decode_mode")
    if mode is None:
        return None
    return LEGACY_DECODE_LABELS.get(mode, mode)


def spilled(entry: dict) -> bool | None:
    """True when the training step needed more memory than the GPU has.

    Prefers the flag recorded by the fixed benchmark. For older entries, compares
    the allocated training peak with the device total from the sweep log.
    """
    if entry.get(SPILL_KEY) is not None:
        return bool(entry[SPILL_KEY])
    peak = entry.get("peak_train_mb")
    if peak is None:
        return None
    return peak > DEVICE_TOTAL_MIB


eff = load("efficiency_gpu.json")
# Results written by the fixed code carry provenance; anything without it is pre-fix.
training_runs = [load(p.name) for p in RESULTS.glob("*.json") if p.name != "efficiency_gpu.json"]
RESULTS_PRE_FIX = not all(PROVENANCE_KEY in r for r in [*eff.get("entries", []), *training_runs])
data = {
    "results_status": {"pre_fix": RESULTS_PRE_FIX, "details": "results/PROVENANCE.md"},
    "environment": eff.get("environment", {}),
    "models": [{"id": m, "label": LABELS[m]} for m in MODELS],
    "parity": task_results("parity", ["token_accuracy", "sequence_success"]),
    "selective_copy": task_results("selective_copy", ["sequence_success", "token_accuracy"]),
    "exact_copy_in_dist": exact_copy_in_dist(),
    "imdb": task_results("imdb", ["accuracy", "auc"]),
    "efficiency": [
        {
            "model": e.get("model"),
            "seq_len": e.get("seq_len"),
            "train_step_ms": e.get("train_step_ms"),
            "peak_train_mb": e.get("peak_train_mb"),
            "train_tokens_per_s": e.get("train_tokens_per_s"),
            "decode_per_token_ms": e.get("decode_per_token_ms"),
            "prefill_per_token_ms": e.get("prefill_per_token_ms"),
            "decode_mode": e.get("decode_mode"),
            "decode_label": decode_label(e),
            "oom": bool(e.get("oom") or e.get("oom_train") or e.get("oom_infer")),
            "spill": spilled(e),
            "pre_fix": PROVENANCE_KEY not in e,
        }
        for e in eff.get("entries", [])
    ],
}

out_path = Path(__file__).resolve().parent / "data.js"
out_path.write_text(
    "// Generated by demo-site/build.py from committed results/ - do not hand-edit.\n"
    "window.DATA = " + json.dumps(data, indent=1) + ";\n",
    encoding="utf-8",
    newline="\n",
)
print(f"wrote {out_path}")
