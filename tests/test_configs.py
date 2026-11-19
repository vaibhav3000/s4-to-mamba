"""The committed training configs: parameter matching and protocol settings (ADR 0001, 0005)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssmbench.train.cli import split_variant
from ssmbench.train.trainer import build_model
from ssmbench.utils import count_parameters, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["parity", "selective_copy", "exact_copy", "exact_copy_varlen", "imdb"]


def _total(cfg: dict, label: str) -> int:
    backbone, kwargs = split_variant(label, cfg["models"][label])
    return count_parameters(
        build_model({"task": cfg["task"], "model": {"name": backbone, "kwargs": kwargs}})
    )


@pytest.mark.parametrize("name", CONFIGS)
def test_every_variant_is_within_ten_percent_of_the_transformer(name: str) -> None:
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    anchor = _total(cfg, "transformer")
    for label in cfg["models"]:
        n = _total(cfg, label)
        assert abs(n - anchor) <= 0.10 * anchor, f"{label}: {n} vs {anchor}"


@pytest.mark.parametrize("name", CONFIGS)
def test_protocol_settings(name: str) -> None:
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg["train"]["deterministic"] is True
    data = cfg["data"]
    assert "seed" in data
    assert data.get("val_samples") or data.get("val_size")
    labels = set(cfg["models"])
    assert {"transformer", "transformer_rope", "s4d", "mamba2", "mamba3"} <= labels
    for label, spec in cfg["models"].items():
        backbone, _ = split_variant(label, spec)
        if backbone == "transformer":
            assert spec.get("attn_backend") == "efficient", label


def test_parity_has_the_rotation_ablation_and_extrapolation_lengths() -> None:
    cfg = load_config(ROOT / "configs" / "parity.yaml")
    ablation = cfg["models"]["mamba3_no_rotation"]
    assert ablation["backbone"] == "mamba3" and ablation["rotation"] is False
    assert {k: v for k, v in ablation.items() if k not in ("backbone", "rotation")} == cfg[
        "models"
    ]["mamba3"]
    assert cfg["data"]["eval_lengths"][0] == cfg["data"]["train"]["seq_len"]
    assert max(cfg["data"]["eval_lengths"]) >= 8 * cfg["data"]["train"]["seq_len"]


def test_exact_copy_configs_differ_only_in_training_lengths() -> None:
    fixed = load_config(ROOT / "configs" / "exact_copy.yaml")
    varlen = load_config(ROOT / "configs" / "exact_copy_varlen.yaml")
    assert varlen["data"]["train"] == {"min_len": 1, "max_len": 32}
    assert fixed["data"]["train"] == {"max_len": 32}
    for key in ("task", "train", "models"):
        assert fixed[key] == varlen[key]
    rest = {k: v for k, v in fixed["data"].items() if k != "train"}
    assert rest == {k: v for k, v in varlen["data"].items() if k != "train"}
