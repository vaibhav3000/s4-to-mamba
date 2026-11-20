"""Synthetic results files for testing the table and figure generators.

The numbers are made up to exercise formatting and edge cases; they never reach a
committed table, figure or document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _stats(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5 if n > 1 else None
    return {
        "mean": mean,
        "std": std,
        "n": n,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_fake_results(root: Path) -> None:
    """Fill root (a stand-in for results/) with one file of every kind."""
    evals = {
        "parity": ["T=128", "T=256"],
        "selective_copy": ["id", "ood"],
        "exact_copy": ["L=16", "L=32"],
        "exact_copy_varlen": ["L=16", "L=32"],
        "imdb": ["test"],
    }
    for config, names in evals.items():
        metric = "accuracy" if config == "imdb" else "token_accuracy"
        variants = {}
        for i, label in enumerate(["transformer", "s4d", "mamba3"]):
            variants[label] = {
                "architecture": label,
                "n_parameters": 1000 + i,
                "seeds": [0, 1, 2],
                "wall_time_seconds": _stats([10.0, 11.0, 12.0]),
                "evals": {
                    name: {metric: _stats([0.5 + 0.1 * i, 0.52 + 0.1 * i, 0.54 + 0.1 * i])}
                    for name in names
                },
            }
        _write(
            root / "train" / config / "summary.json",
            {"config": config, "chance": {metric: 0.5}, "variants": variants},
        )
    for precision in ("fp32", "bf16"):
        entries = []
        for label in ("transformer_efficient", "s4d"):
            for T in (1024, 2048):
                entries.append(
                    {
                        "model": label,
                        "seq_len": T,
                        "n_parameters": {"transformer_efficient": 2048, "s4d": 512}[label],
                        "train_step_ms_median": 10.0 * T / 1024,
                        "infer_ms_median": 1.0,
                        "peak_train_mb": 100.0 * T / 1024,
                        "spilled_train": T == 2048 and label == "s4d",
                    }
                )
        entries.append({"model": "s4d", "seq_len": 4096, "skipped": "spilled at T=2048"})
        entries.append({"model": "transformer_efficient", "seq_len": 4096, "oom_train": True})
        entries.append(
            {"model": "transformer_cudnn", "seq_len": 1024, "unsupported": "No available kernel"}
        )
        _write(root / "efficiency" / f"sweep_{precision}.json", {"entries": entries})
        inf = []
        for label in ("transformer_efficient", "mamba3"):
            for L in (1024, 2048):
                inf.append(
                    {
                        "model": label,
                        "seq_len": L,
                        "n_parameters": {"transformer_efficient": 2048, "mamba3": 768}[label],
                        "ttft_ms_median": 5.0 * L / 1024,
                        "decode_ms_per_token_median": 0.5,
                        "tokens_per_s": 2000.0,
                        "decode_state_bytes": 4096 * (L if label.startswith("transformer") else 1),
                        "peak_allocated_mb": 50.0,
                    }
                )
        _write(root / "inference" / f"decode_{precision}.json", {"entries": inf})
    num = []
    for label in ("transformer", "mamba3"):
        for seed in (0, 1):
            err = {"rel_l2": 1e-7, "max_abs": 1e-6, "nonfinite": 0}
            num.append(
                {
                    "model": label,
                    "seq_len": 1024,
                    "seed": seed,
                    "forward": {
                        "fp32_vs_fp64": err,
                        "bf16_vs_fp64": {**err, "rel_l2": 3e-3},
                        "fp16_vs_fp64": {**err, "rel_l2": 5e-4},
                        "bf16_vs_fp32": {**err, "rel_l2": 3e-3},
                    },
                    "decode": {
                        "fp32": {
                            "checkpoints": [1, 16],
                            "rel_l2_at": [1e-7, 2e-7],
                            "max_rel_l2": 2e-7,
                            "nonfinite": 0,
                        },
                        "bf16": {
                            "checkpoints": [1, 16],
                            "rel_l2_at": [3e-3, 4e-3],
                            "max_rel_l2": 4e-3,
                            "nonfinite": 0,
                        },
                    },
                }
            )
    _write(root / "numerics" / "numerics.json", {"entries": num})
    comp = [
        {
            "model": label,
            "forward": {"graph_count": 1, "graph_break_count": 0, "break_reasons": []},
            "step": {"graph_count": 1, "graph_break_count": 0, "break_reasons": []},
            "aot_eager": {"max_abs_output_diff": 0.0, "max_abs_grad_diff": 1e-8},
            "inductor": {"status": "not run: Triton is not installed in this environment"},
        }
        for label in ("transformer", "mamba3")
    ]
    _write(root / "compile" / "compile.json", {"entries": comp})
