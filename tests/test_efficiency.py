"""CPU tests for the efficiency benchmark's bookkeeping (no timing claims)."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import pytest
import torch
from torch import nn

from ssmbench.benchmarks import efficiency
from ssmbench.utils import source_sha256

TINY_KWARGS: dict[str, dict[str, int]] = {
    "transformer": {"d_model": 32, "n_layers": 1, "n_heads": 4},
    "mamba2": {"d_model": 32, "n_layers": 1, "d_state": 8, "headdim": 16},
}


@pytest.mark.parametrize("name", list(TINY_KWARGS))
def test_inference_and_decode_timing_run_in_eval_mode(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The train step runs in training mode; every later forward (inference and
    full-forward decode) runs in eval mode."""
    modes: list[bool] = []
    real_build = efficiency.build_backbone

    def recording_build(model_name: str, kwargs: dict) -> nn.Module:
        backbone = real_build(model_name, kwargs)
        backbone.register_forward_pre_hook(lambda module, args: modes.append(module.training))
        return backbone

    monkeypatch.setattr(efficiency, "build_backbone", recording_build)
    entry = efficiency.benchmark_single(
        name,
        seq_len=16,
        batch_size=2,
        repeats=2,
        device=torch.device("cpu"),
        model_kwargs=TINY_KWARGS[name],
    )
    assert "train_step_ms" in entry and "infer_ms" in entry
    assert "decode_per_token_ms" in entry or "prefill_per_token_ms" in entry
    # _time_gpu: 2 warmup + 2 timed train steps, then inference and decode calls.
    assert modes[:4] == [True] * 4
    assert len(modes) > 4 and not any(modes[4:])


def test_train_step_uses_an_lm_head_even_when_d_model_is_below_vocab() -> None:
    """The timed step maps features to VOCAB logits with a Linear head. Without it,
    d_model = 16 < VOCAB = 32 made the d_model-way 'logits' reject targets >= 16."""
    d_model = 16
    assert d_model < efficiency.VOCAB
    entry = efficiency.benchmark_single(
        "s4d",
        seq_len=16,
        batch_size=2,
        repeats=2,
        device=torch.device("cpu"),
        model_kwargs={"d_model": d_model, "n_layers": 1, "d_state": 4},
    )
    assert entry["train_step_ms"] > 0 and entry["train_step_ms_std"] is not None
    vocab = efficiency.VOCAB
    embed_and_head = vocab * d_model + (d_model * vocab + vocab)  # Embedding + Linear(+bias)
    assert entry["n_parameters_with_embed_and_head"] == entry["n_parameters"] + embed_and_head


DECODE_CASES = {
    "s4d": ({"d_model": 32, "n_layers": 1, "d_state": 4}, "decode_per_token_ms", "stepwise"),
    "transformer": (TINY_KWARGS["transformer"], "decode_per_token_ms", "no KV cache"),
    "mamba2": (TINY_KWARGS["mamba2"], "prefill_per_token_ms", "no step() path"),
    "mamba3": (
        {"d_model": 32, "n_layers": 1, "d_state": 8, "headdim": 16},
        "prefill_per_token_ms",
        "no step() path",
    ),
}


@pytest.mark.parametrize("name", list(DECODE_CASES))
def test_decode_numbers_are_labeled_by_what_was_timed(name: str) -> None:
    """Only models with a step() path (or the Transformer's documented full-prefix
    recompute) report decode_per_token_ms; Mamba-2/3 report prefill time / T under
    its own key, never as a decode cost."""
    kwargs, key, label = DECODE_CASES[name]
    entry = efficiency.benchmark_single(
        name, seq_len=16, batch_size=2, repeats=2, device=torch.device("cpu"), model_kwargs=kwargs
    )
    assert entry[key] > 0 and label in entry["decode_mode"]
    other = {"decode_per_token_ms", "prefill_per_token_ms"} - {key}
    assert not other & set(entry)


def test_single_repeat_reports_no_std() -> None:
    mean_ms, std_ms = efficiency._time_gpu(lambda: None, repeats=1, warmup=0, device="cpu")
    assert mean_ms >= 0 and std_ms is None


@pytest.mark.parametrize(
    ("exc", "is_oom"),
    [
        (torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"), True),
        (RuntimeError("CUDA error: out of memory"), True),
        (RuntimeError("CUDA error: an illegal memory access was encountered"), False),
        (IndexError("Target 29 is out of bounds."), False),
    ],
)
def test_only_out_of_memory_errors_count_as_oom(exc: BaseException, is_oom: bool) -> None:
    assert efficiency._is_oom(exc) is is_oom


def _sweep_cfg(tmp_path: Path, models: list[str], lengths: list[int]) -> dict:
    return {
        "device": "cpu",
        "out_path": str(tmp_path / "efficiency.json"),
        "batch_size": 2,
        "repeats": 2,
        "seq_lengths": lengths,
        "models": models,
    }


def test_oom_is_recorded_and_the_sweep_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_benchmark(name: str, seq_len: int, *args: object, **kwargs: object) -> dict:
        if seq_len == 16:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return {"model": name, "seq_len": seq_len, "train_step_ms": 1.0}

    monkeypatch.setattr(efficiency, "benchmark_single", fake_benchmark)
    results = efficiency.run_efficiency(_sweep_cfg(tmp_path, ["s4d"], [16, 32]))
    assert [(e["seq_len"], bool(e.get("oom"))) for e in results["entries"]] == [
        (16, True),
        (32, False),
    ]


def test_non_oom_error_is_recorded_and_aborts_the_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A shape bug or a broken CUDA context must stop the sweep, not be plotted as a
    hardware limit or skipped over."""
    attempted: list[int] = []

    def failing_benchmark(name: str, seq_len: int, *args: object, **kwargs: object) -> dict:
        attempted.append(seq_len)
        raise IndexError("Target 29 is out of bounds.")

    monkeypatch.setattr(efficiency, "benchmark_single", failing_benchmark)
    cfg = _sweep_cfg(tmp_path, ["s4d"], [16, 32])
    with pytest.raises(RuntimeError, match="sweep aborted"):
        efficiency.run_efficiency(cfg)
    assert attempted == [16]
    saved = json.loads(Path(cfg["out_path"]).read_text(encoding="utf-8"))
    assert saved["entries"][-1]["error"].startswith("IndexError")
    assert not any(e.get("oom") for e in saved["entries"])


def _fake_benchmark(name: str, seq_len: int, *args: object, **kwargs: object) -> dict:
    return {"model": name, "seq_len": seq_len, "train_step_ms": 1.0}


PROVENANCE_KEYS = {
    "git_sha",
    "git_dirty",
    "source_sha256",
    "torch",
    "cuda",
    "python",
    "platform",
    "device",
    "gpu",
    "timestamp_utc",
}


def test_every_entry_carries_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(efficiency, "benchmark_single", _fake_benchmark)
    results = efficiency.run_efficiency(_sweep_cfg(tmp_path, ["s4d", "mamba2"], [16]))
    assert len(results["entries"]) == 2
    for entry in results["entries"]:
        assert set(entry["provenance"]) == PROVENANCE_KEYS
        assert entry["provenance"]["source_sha256"] == source_sha256()
        assert entry["provenance"]["python"] == platform.python_version()


def test_resume_skips_points_measured_with_the_same_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    measured: list[tuple[str, int]] = []

    def counting_benchmark(name: str, seq_len: int, *args: object, **kwargs: object) -> dict:
        measured.append((name, seq_len))
        return _fake_benchmark(name, seq_len)

    monkeypatch.setattr(efficiency, "benchmark_single", counting_benchmark)
    efficiency.run_efficiency(_sweep_cfg(tmp_path, ["s4d"], [16]))
    results = efficiency.run_efficiency(_sweep_cfg(tmp_path, ["s4d"], [16, 32]))
    assert measured == [("s4d", 16), ("s4d", 32)]
    assert [e["seq_len"] for e in results["entries"]] == [16, 32]


@pytest.mark.parametrize(
    "prior_provenance",
    [None, {"source_sha256": "0" * 64}],
    ids=["no-provenance", "other-source"],
)
def test_resume_refuses_entries_from_other_code_unless_forced(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, prior_provenance: dict | None
) -> None:
    """A results file written by other code (or before provenance existed, like the
    committed one) is not silently extended; --force merges explicitly."""
    monkeypatch.setattr(efficiency, "benchmark_single", _fake_benchmark)
    cfg = _sweep_cfg(tmp_path, ["s4d"], [16, 32])
    prior_entry: dict = {"model": "s4d", "seq_len": 16, "train_step_ms": 9.0}
    if prior_provenance is not None:
        prior_entry["provenance"] = prior_provenance
    Path(cfg["out_path"]).write_text(json.dumps({"entries": [prior_entry]}), encoding="utf-8")

    with pytest.raises(efficiency.ProvenanceMismatchError, match="--force"):
        efficiency.run_efficiency(cfg)
    assert json.loads(Path(cfg["out_path"]).read_text(encoding="utf-8")) == {
        "entries": [prior_entry]
    }

    results = efficiency.run_efficiency(cfg, force=True)
    assert [(e["seq_len"], e["train_step_ms"]) for e in results["entries"]] == [
        (16, 9.0),
        (32, 1.0),
    ]


def test_resume_remeasures_points_that_ended_in_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(efficiency, "benchmark_single", _fake_benchmark)
    cfg = _sweep_cfg(tmp_path, ["s4d"], [16])
    error_entry = {
        "model": "s4d",
        "seq_len": 16,
        "error": "IndexError: boom",
        "provenance": {"source_sha256": source_sha256()},
    }
    Path(cfg["out_path"]).write_text(json.dumps({"entries": [error_entry]}), encoding="utf-8")
    results = efficiency.run_efficiency(cfg)
    assert len(results["entries"]) == 1 and "error" not in results["entries"][0]
