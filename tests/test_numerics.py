"""Numerical-stability benchmark on CPU (tiny models; bounds are sanity checks, not results)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from ssmbench.benchmarks import numerics

TINY: dict[str, dict] = {
    "transformer": {"d_model": 16, "n_layers": 2, "n_heads": 2},
    "s4d": {"d_model": 16, "n_layers": 2, "d_state": 4},
    "mamba": {"d_model": 16, "n_layers": 2, "d_state": 4},
    "mamba2": {"d_model": 16, "n_layers": 2, "d_state": 4, "headdim": 8, "chunk_size": 8},
    "mamba3": {"d_model": 16, "n_layers": 2, "d_state": 4, "headdim": 8, "chunk_size": 8},
}


def test_compare_reports_relative_and_absolute_error_and_nonfinite_values() -> None:
    ref = torch.tensor([3.0, 4.0], dtype=torch.float64)
    same = numerics.compare(ref.clone(), ref)
    assert same == {"rel_l2": 0.0, "max_abs": 0.0, "nonfinite": 0}
    off = numerics.compare(torch.tensor([3.0, 4.5]), ref)
    assert math.isclose(off["rel_l2"], 0.1, rel_tol=1e-12) and math.isclose(off["max_abs"], 0.5)
    assert numerics.compare(torch.tensor([float("nan"), 4.0]), ref)["nonfinite"] == 1


@pytest.mark.parametrize("name", list(TINY))
def test_forward_errors_rank_precisions(name: str) -> None:
    """float32 tracks the float64 reference closely; bf16 autocast is far coarser but finite."""
    errors = numerics.forward_errors(
        name, TINY[name], seq_len=24, seed=0, device=torch.device("cpu")
    )
    assert errors["fp32_vs_fp64"]["rel_l2"] < 1e-5
    assert errors["bf16_vs_fp64"]["nonfinite"] == 0
    assert errors["fp32_vs_fp64"]["rel_l2"] < errors["bf16_vs_fp64"]["rel_l2"] < 0.1
    assert set(errors) == {"fp32_vs_fp64", "bf16_vs_fp64", "fp16_vs_fp64", "bf16_vs_fp32"}


@pytest.mark.parametrize("name", ["transformer", "mamba3"])
def test_decode_drift_is_measured_per_step_against_float64(name: str) -> None:
    drift = numerics.decode_drift(
        name, TINY[name], prompt_len=12, steps=20, seed=0, device=torch.device("cpu")
    )
    fp32 = drift["fp32"]
    assert fp32["checkpoints"] == [1, 16, 20]
    assert max(fp32["rel_l2_at"]) < 1e-5 and fp32["nonfinite"] == 0
    assert len(drift["bf16"]["rel_l2_at"]) == 3


def test_run_numerics_writes_entries_with_provenance(tmp_path: Path) -> None:
    cfg = {
        "device": "cpu",
        "out_path": str(tmp_path / "numerics.json"),
        "seq_lengths": [16],
        "seeds": [0, 1],
        "decode": {"prompt_len": 8, "steps": 4},
        "models": ["mamba2"],
        "variants": {"mamba2": TINY["mamba2"]},
    }
    results = numerics.run_numerics(cfg)
    assert [(e["model"], e["seq_len"], e["seed"]) for e in results["entries"]] == [
        ("mamba2", 16, 0),
        ("mamba2", 16, 1),
    ]
    assert "decode" in results["entries"][0]
    assert "source_sha256" in results["environment"]
