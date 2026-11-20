"""torch.compile readiness: graph breaks and compiled-vs-eager agreement (CPU)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ssmbench.benchmarks import compile_check
from ssmbench.models import build_backbone

SMALL: dict[str, tuple[str, dict]] = {
    "transformer": ("transformer", {"d_model": 16, "n_layers": 2, "n_heads": 2}),
    "transformer_rope": (
        "transformer",
        {"d_model": 16, "n_layers": 2, "n_heads": 2, "pos_encoding": "rope"},
    ),
    "s4d": ("s4d", {"d_model": 16, "n_layers": 2, "d_state": 4}),
    "mamba": ("mamba", {"d_model": 16, "n_layers": 2, "d_state": 4}),
    "mamba2": (
        "mamba2",
        {"d_model": 16, "n_layers": 2, "d_state": 4, "headdim": 8, "chunk_size": 8},
    ),
    "mamba3": (
        "mamba3",
        {"d_model": 16, "n_layers": 2, "d_state": 4, "headdim": 8, "chunk_size": 8},
    ),
}


@pytest.mark.parametrize("label", list(SMALL))
def test_training_forward_compiles_into_one_graph(label: str) -> None:
    """No graph breaks in forward (training mode, T = 16). Before the fix, autocast
    entered through an ExitStack broke every Mamba scan into extra graphs."""
    name, kwargs = SMALL[label]
    torch.manual_seed(0)
    backbone = build_backbone(name, kwargs)
    report = compile_check.graph_report(backbone, torch.randn(2, 16, 16))
    assert report["graph_break_count"] == 0, report["break_reasons"]
    assert report["graph_count"] == 1


@pytest.mark.parametrize("label", list(SMALL))
def test_aot_eager_compiled_forward_matches_eager(label: str) -> None:
    """Dynamo plus AOTAutograd (no inductor) reproduces eager outputs and input gradients
    within float32 rounding: rtol = atol = 1e-5."""
    name, kwargs = SMALL[label]
    torch.manual_seed(0)
    backbone = build_backbone(name, kwargs)
    diff = compile_check.compiled_difference(backbone, torch.randn(2, 16, 16), backend="aot_eager")
    assert diff["max_abs_output_diff"] <= 1e-5
    assert diff["max_abs_grad_diff"] <= 1e-5


def test_report_covers_forward_and_step(tmp_path: Path) -> None:
    cfg = {
        "device": "cpu",
        "out_path": str(tmp_path / "compile.json"),
        "seq_len": 16,
        "models": ["s4d", "mamba3"],
        "variants": {"s4d": SMALL["s4d"][1], "mamba3": SMALL["mamba3"][1]},
    }
    results = compile_check.run_compile(cfg)
    by_model = {e["model"]: e for e in results["entries"]}
    for label in ("s4d", "mamba3"):
        entry = by_model[label]
        assert entry["forward"]["graph_break_count"] == 0
        assert "graph_break_count" in entry["step"]
        assert entry["aot_eager"]["max_abs_output_diff"] <= 1e-5
        assert entry["inductor"]["status"].startswith(("measured", "not run"))
    assert "source_sha256" in results["environment"]
