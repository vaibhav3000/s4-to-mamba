"""Console entry points (ssmbench-train, ssmbench-efficiency)."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

from ssmbench.train import cli

ROOT = Path(__file__).resolve().parents[1]


def test_console_scripts_resolve_to_callables() -> None:
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    assert set(scripts) == {
        "ssmbench-train",
        "ssmbench-efficiency",
        "ssmbench-summarize",
        "ssmbench-inference",
    }
    for target in scripts.values():
        module_name, func_name = target.split(":")
        assert callable(getattr(importlib.import_module(module_name), func_name))


def test_train_cli_builds_one_run_config_per_requested_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs: list[tuple[dict, str]] = []
    monkeypatch.setattr(cli, "run_training", lambda cfg, out: runs.append((cfg, out)))
    cli.main(
        [
            "--config",
            str(ROOT / "configs" / "parity.yaml"),
            "--models",
            "mamba3",
            "--out",
            str(tmp_path),
            "--device",
            "cpu",
        ]
    )
    assert len(runs) == 1
    cfg, out = runs[0]
    # One directory per variant under <out>/<name_prefix>/; the run writes seed<k>.json.
    assert out == str(tmp_path / "parity" / "mamba3")
    assert cfg["name"] == "parity_mamba3" and cfg["device"] == "cpu"
    assert cfg["train"]["seed"] == 42  # no --seeds: the config's train.seed
    assert cfg["model"] == {
        "name": "mamba3",
        "kwargs": {"d_model": 80, "n_layers": 2, "d_state": 32, "headdim": 32},
    }


def test_train_cli_rejects_models_missing_from_the_config() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--config", str(ROOT / "configs" / "imdb.yaml"), "--models", "mamba"])
    assert excinfo.value.code == 2
