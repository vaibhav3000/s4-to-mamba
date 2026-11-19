"""Runs: result layout, evaluation sets, determinism, checkpoint resume, skip and refuse.

CPU only, tiny parity runs (seconds each).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ssmbench.train import cli, trainer


def _cfg(model: str = "mamba2", epochs: int = 3, **model_kwargs: object) -> dict:
    kwargs = {
        "mamba2": {"d_model": 16, "n_layers": 1, "d_state": 4, "headdim": 8},
        "transformer": {"d_model": 16, "n_layers": 1, "n_heads": 2, "max_len": 64},
    }[model]
    return {
        "name": f"parity_{model}",
        "config": "parity",
        "label": model,
        "task": {"type": "token_classification", "vocab_size": 2, "n_classes": 2},
        "data": {
            "name": "parity",
            "batch_size": 8,
            "seed": 0,
            "train_samples": 32,
            "val_samples": 8,
            "test_samples": 8,
            "train": {"seq_len": 8},
            "eval_lengths": [8, 16],
        },
        "train": {"epochs": epochs, "lr": 3e-3, "seed": 5, "deterministic": True},
        "model": {"name": model, "kwargs": {**kwargs, **model_kwargs}},
        "device": "cpu",
    }


def test_result_has_validation_history_and_final_test_sets(tmp_path: Path) -> None:
    results = trainer.run_training(_cfg(epochs=2), tmp_path)
    out = tmp_path / "seed5.json"
    assert json.loads(out.read_text(encoding="utf-8"))["final_evals"] == results["final_evals"]
    assert set(results["final_evals"]) == {"T=8", "T=16"}
    assert len(results["history"]) == 2 and "val_token_accuracy" in results["history"][0]
    assert results["seed"] == 5 and results["label"] == "mamba2"
    assert results["provenance"]["config_sha256"] == trainer.run_config_sha256(_cfg(epochs=2))
    assert results["attention_backend"] is None


def test_transformer_runs_record_the_attention_backend_that_ran(tmp_path: Path) -> None:
    results = trainer.run_training(_cfg("transformer", epochs=1, attn_backend="math"), tmp_path)
    assert results["attention_backend"] == ["math"]


def test_same_seed_reproduces_bitwise(tmp_path: Path) -> None:
    a = trainer.run_training(_cfg(epochs=2), tmp_path / "a")
    b = trainer.run_training(_cfg(epochs=2), tmp_path / "b")
    strip = [{k: v for k, v in h.items() if k != "epoch_seconds"} for h in a["history"]]
    assert strip == [{k: v for k, v in h.items() if k != "epoch_seconds"} for h in b["history"]]
    assert a["final_evals"] == b["final_evals"]


class _Crash(Exception):
    pass


def test_interrupted_run_resumes_to_the_same_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crash during epoch 3 (after epoch 2's checkpoint), rerun: the history and final
    metrics equal an uninterrupted run exactly, and the checkpoint is removed at the end."""
    straight = trainer.run_training(_cfg(epochs=3), tmp_path / "straight")

    cfg = _cfg(epochs=3)
    cfg["checkpoint_dir"] = str(tmp_path / "ckpt")
    real_evaluate = trainer.evaluate
    calls = {"n": 0}

    def crashing(*args: object, **kwargs: object) -> dict:
        calls["n"] += 1
        if calls["n"] == 3:  # validation of epoch 3
            raise _Crash
        return real_evaluate(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(trainer, "evaluate", crashing)
    with pytest.raises(_Crash):
        trainer.run_training(cfg, tmp_path / "resumed")
    assert (tmp_path / "ckpt" / "last.pt").exists()
    monkeypatch.setattr(trainer, "evaluate", real_evaluate)
    resumed = trainer.run_training(cfg, tmp_path / "resumed")

    def strip(history: list[dict]) -> list[dict]:
        return [{k: v for k, v in h.items() if k != "epoch_seconds"} for h in history]

    assert strip(resumed["history"]) == strip(straight["history"])
    assert resumed["final_evals"] == straight["final_evals"]
    assert resumed["resumed_from_epoch"] == 2
    assert not (tmp_path / "ckpt" / "last.pt").exists()


def test_checkpoint_from_other_settings_is_not_resumed(tmp_path: Path) -> None:
    cfg = _cfg(epochs=1)
    cfg["checkpoint_dir"] = str(tmp_path / "ckpt")
    (tmp_path / "ckpt").mkdir()
    torch.save({"config_sha256": "something else", "epoch": 1}, tmp_path / "ckpt" / "last.pt")
    results = trainer.run_training(cfg, tmp_path)
    assert results["resumed_from_epoch"] is None


def test_deterministic_mode_is_restored_after_a_run(tmp_path: Path) -> None:
    before = torch.are_deterministic_algorithms_enabled()
    trainer.run_training(_cfg(epochs=1), tmp_path)
    assert torch.are_deterministic_algorithms_enabled() == before


ROOT = Path(__file__).resolve().parents[1]


def _write_config(tmp_path: Path) -> Path:
    cfg = _cfg(epochs=1)
    text = json.dumps(
        {
            "name_prefix": "parity",
            "task": cfg["task"],
            "data": cfg["data"],
            "train": cfg["train"],
            "models": {
                "mamba2": cfg["model"]["kwargs"],
                "mamba2_wide": {"backbone": "mamba2", **cfg["model"]["kwargs"], "d_model": 32},
            },
        }
    )
    path = tmp_path / "tiny.yaml"
    path.write_text(text, encoding="utf-8")  # JSON is valid YAML
    return path


def test_cli_runs_every_variant_and_seed_into_its_own_file(tmp_path: Path) -> None:
    cfg_path = _write_config(tmp_path)
    out = tmp_path / "results"
    cli.main(["--config", str(cfg_path), "--out", str(out), "--seeds", "0", "1", "--device", "cpu"])
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.json"))
    assert files == [
        "parity/mamba2/seed0.json",
        "parity/mamba2/seed1.json",
        "parity/mamba2_wide/seed0.json",
        "parity/mamba2_wide/seed1.json",
    ]
    wide = json.loads((out / "parity/mamba2_wide/seed1.json").read_text(encoding="utf-8"))
    assert wide["model"]["name"] == "mamba2" and wide["model"]["kwargs"]["d_model"] == 32
    assert wide["seed"] == 1 and wide["label"] == "mamba2_wide"


def test_cli_skips_completed_runs_and_refuses_foreign_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _write_config(tmp_path)
    out = tmp_path / "results"
    args = ["--config", str(cfg_path), "--out", str(out), "--seeds", "0", "--device", "cpu"]
    cli.main([*args, "--models", "mamba2"])
    ran: list[str] = []
    monkeypatch.setattr(cli, "run_training", lambda cfg, out_dir: ran.append(cfg["name"]))
    cli.main([*args, "--models", "mamba2"])
    assert ran == []  # same source and settings: skipped

    path = out / "parity/mamba2/seed0.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["provenance"]["source_sha256"] = "older code"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit, match="other code"):
        cli.main([*args, "--models", "mamba2"])
    cli.main([*args, "--models", "mamba2", "--force"])
    assert ran == ["parity_mamba2"]
