"""demo-site/build.py: efficiency rows pair each variant with its own decode measurement."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

BUILD = Path(__file__).resolve().parents[1] / "demo-site" / "build.py"


def _load_build() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo_build", BUILD)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_decode_latency_follows_the_variant_not_the_architecture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    variants = ("transformer_flash", "transformer_efficient", "transformer_cudnn")
    sweep = [
        {
            "model": "transformer_flash",
            "architecture": "transformer",
            "seq_len": 1024,
            "unsupported": "No available kernel",
        },
        {
            "model": "transformer_efficient",
            "architecture": "transformer",
            "seq_len": 1024,
            "train_step_ms_median": 10.0,
            "peak_train_mb": 100.0,
        },
        {
            "model": "transformer_cudnn",
            "architecture": "transformer",
            "seq_len": 1024,
            "train_step_ms_median": 9.0,
            "peak_train_mb": 100.0,
        },
    ]
    decode = [
        {
            "model": "transformer_flash",
            "architecture": "transformer",
            "seq_len": 1024,
            "unsupported": "No available kernel",
        },
        {
            "model": "transformer_efficient",
            "architecture": "transformer",
            "seq_len": 1024,
            "decode_ms_per_token_median": 1.25,
        },
        {
            "model": "transformer_cudnn",
            "architecture": "transformer",
            "seq_len": 1024,
            "decode_ms_per_token_median": 0.75,
        },
    ]
    _write(tmp_path / "efficiency" / "sweep_bf16.json", {"entries": sweep})
    _write(tmp_path / "inference" / "decode_bf16.json", {"entries": decode})
    build = _load_build()
    monkeypatch.setattr(build, "RESULTS", tmp_path)

    precision, rows = build.efficiency()

    assert precision == "bf16"
    by_model = {row["model"]: row["decode_per_token_ms"] for row in rows}
    assert set(by_model) == set(variants)
    assert by_model["transformer_efficient"] == pytest.approx(1.25, abs=0.0)
    assert by_model["transformer_cudnn"] == pytest.approx(0.75, abs=0.0)
    assert by_model["transformer_flash"] is None
