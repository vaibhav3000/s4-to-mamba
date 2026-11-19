"""CPU tests for the efficiency benchmark's bookkeeping (no timing claims)."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from ssmbench import utils
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


# Every backbone now decodes through prefill/step (docs/adr/0003): the SSMs with an
# O(1) recurrent state, the Transformer with a KV cache. No full-prefix recompute and no
# prefill-time-per-token stand-in remain.
DECODE_CASES = {
    "s4d": ({"d_model": 32, "n_layers": 1, "d_state": 4}, "decode_per_token_ms", "stepwise"),
    "transformer": (TINY_KWARGS["transformer"], "decode_per_token_ms", "KV cache"),
    "mamba2": (TINY_KWARGS["mamba2"], "decode_per_token_ms", "stepwise"),
    "mamba3": (
        {"d_model": 32, "n_layers": 1, "d_state": 8, "headdim": 16},
        "decode_per_token_ms",
        "stepwise",
    ),
}


@pytest.mark.parametrize("name", list(DECODE_CASES))
def test_decode_numbers_are_labeled_by_what_was_timed(name: str) -> None:
    """decode_per_token_ms is always a real decode (step() with a recurrent state or a
    KV cache), labeled by decode_mode; no other per-token key stands in for it."""
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
    "nvidia_driver",
    "timestamp_utc",
    "config_sha256",
    "gpu_state_at_sweep_start",
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


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"batch_size": 4}, "batch_size differs"),
        ({"repeats": 3}, "repeats differs"),
        ({"provenance_torch": "0.0.0"}, "torch differs"),
        ({"warmup_note": "another measurement setting"}, "config_sha256 differs"),
    ],
)
def test_resume_refuses_entries_from_other_settings_or_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, change: dict, reason: str
) -> None:
    """Same code is not enough: a point measured with another batch size, repeat
    count or torch build is not the same measurement."""
    monkeypatch.setattr(efficiency, "benchmark_single", _fake_benchmark)
    cfg = _sweep_cfg(tmp_path, ["s4d"], [16, 32])
    efficiency.run_efficiency(dict(cfg, seq_lengths=[16]))
    if "provenance_torch" in change:
        path = Path(cfg["out_path"])
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["entries"][0]["provenance"]["torch"] = change["provenance_torch"]
        path.write_text(json.dumps(saved), encoding="utf-8")
        new_cfg = cfg
    else:
        new_cfg = dict(cfg, **change)
    with pytest.raises(efficiency.ProvenanceMismatchError, match=reason):
        efficiency.run_efficiency(new_cfg)


def test_stepwise_decode_runs_in_eval_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """S4D and Mamba-1 decode through block.step(), which forward hooks do not see;
    check the modes when the decode session starts."""
    seen: list[list[bool]] = []
    real_decode = efficiency._decode_stepwise

    def recording_decode(
        backbone: nn.Module, embed: nn.Module, head: nn.Module, tokens: torch.Tensor
    ) -> None:
        seen.append([m.training for m in (embed, backbone, head, *backbone.modules())])
        real_decode(backbone, embed, head, tokens)

    monkeypatch.setattr(efficiency, "_decode_stepwise", recording_decode)
    efficiency.benchmark_single(
        "s4d",
        seq_len=8,
        batch_size=2,
        repeats=2,
        device=torch.device("cpu"),
        model_kwargs={"d_model": 32, "n_layers": 1, "d_state": 4},
    )
    assert seen and not any(any(flags) for flags in seen)


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


def test_config_hash_ignores_point_selection_but_not_measurement_settings() -> None:
    """Adding models or lengths extends a sweep; changing how points are measured does
    not, so only the latter changes the hash that resume compares."""
    base = {
        "device": "cuda",
        "batch_size": 8,
        "repeats": 5,
        "out_path": "a.json",
        "models": ["s4d"],
        "seq_lengths": [1024],
        "model_length_limits": {"mamba": 4096},
    }
    h = efficiency.config_sha256(base)
    assert h == efficiency.config_sha256(
        dict(base, out_path="b.json", models=["s4d", "mamba"], seq_lengths=[1024, 2048])
    )
    assert h == efficiency.config_sha256(dict(base, model_length_limits={}))
    assert h != efficiency.config_sha256(dict(base, batch_size=4))
    assert h != efficiency.config_sha256(dict(base, device="cpu"))


def _fake_nvidia_smi(monkeypatch: pytest.MonkeyPatch, stdout: str | None) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if stdout is None:
            raise FileNotFoundError("nvidia-smi")
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(utils.subprocess, "run", fake_run)
    return calls


def test_nvidia_smi_query_parses_one_gpu_and_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _fake_nvidia_smi(monkeypatch, "2100, 8001, P0\n")
    assert utils.nvidia_smi_query(["clocks.sm", "clocks.mem", "pstate"]) == {
        "clocks.sm": "2100",
        "clocks.mem": "8001",
        "pstate": "P0",
    }
    assert calls[0][0] == "nvidia-smi"
    _fake_nvidia_smi(monkeypatch, None)  # not installed
    assert utils.nvidia_smi_query(["driver_version"]) is None
    _fake_nvidia_smi(monkeypatch, "only-one-field\n")  # unexpected output
    assert utils.nvidia_smi_query(["clocks.sm", "clocks.mem"]) is None


def test_cuda_provenance_records_the_driver_version(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_nvidia_smi(monkeypatch, "551.86\n")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "Fake GPU")
    provenance = utils.run_provenance(torch.device("cuda"))
    assert provenance["nvidia_driver"] == "551.86" and provenance["gpu"] == "Fake GPU"
    assert provenance["python"] == platform.python_version()


def test_gpu_state_at_sweep_start_is_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_nvidia_smi(monkeypatch, "1500, 8001, 2370, 8001, 61, 35.2, P2\n")
    state = efficiency.gpu_state(torch.device("cuda"))
    assert state is not None and state["clocks.sm"] == "1500" and state["pstate"] == "P2"
    _fake_nvidia_smi(monkeypatch, None)
    assert efficiency.gpu_state(torch.device("cuda")) is None
    assert efficiency.gpu_state(torch.device("cpu")) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_cuda_timing_and_memory_path_smoke() -> None:
    """On a CUDA machine, one tiny point runs through CUDA-event timing and the real
    peak-memory, availability and spill bookkeeping (skipped without CUDA)."""
    entry = efficiency.benchmark_single(
        "s4d",
        seq_len=64,
        batch_size=2,
        repeats=2,
        device=torch.device("cuda"),
        model_kwargs={"d_model": 32, "n_layers": 1, "d_state": 4},
    )
    for key in (
        "train_step_ms",
        "peak_train_mb",
        "peak_train_reserved_mb",
        "device_total_mb",
        "available_at_train_start_mb",
        "free_at_train_end_mb",
        "spilled_train",
        "infer_ms",
        "decode_per_token_ms",
    ):
        assert key in entry, key
    assert entry["peak_train_reserved_mb"] >= entry["peak_train_mb"] > 0
    assert entry["train_step_ms"] > 0


def test_bf16_points_run_under_autocast_and_say_so() -> None:
    entry = efficiency.benchmark_single(
        "mamba2",
        seq_len=16,
        batch_size=2,
        repeats=2,
        device=torch.device("cpu"),
        model_kwargs=TINY_KWARGS["mamba2"],
        precision="bf16",
    )
    assert entry["precision"] == "bf16" and entry["train_step_ms"] > 0
    with pytest.raises(ValueError, match="precision"):
        efficiency.benchmark_single(
            "s4d", 16, 2, 2, torch.device("cpu"), {"d_model": 8, "n_layers": 1}, precision="fp8"
        )


def test_timings_report_median_and_p90() -> None:
    # Ten repeats: below that the nearest-rank p90 is the maximum and is not reported.
    entry = efficiency.benchmark_single(
        "s4d", 16, 2, 10, torch.device("cpu"), {"d_model": 16, "n_layers": 1, "d_state": 4}
    )
    for key in ("train_step_ms_median", "train_step_ms_p90", "infer_ms_median", "infer_ms_p90"):
        assert entry[key] > 0, key
    assert entry["train_step_ms_p90"] >= entry["train_step_ms_median"]


def test_variants_are_labelled_and_record_the_attention_backend(tmp_path: Path) -> None:
    cfg = _sweep_cfg(tmp_path, ["transformer_math", "s4d"], [16])
    cfg["variants"] = {
        "transformer_math": {
            "backbone": "transformer",
            "attn_backend": "math",
            **TINY_KWARGS["transformer"],
        },
        "s4d": {"d_model": 16, "n_layers": 1, "d_state": 4},
    }
    results = efficiency.run_efficiency(cfg)
    by_model = {e["model"]: e for e in results["entries"]}
    tr = by_model["transformer_math"]
    assert tr["architecture"] == "transformer" and tr["attn_backend_used"] == ["math"]
    assert tr["model_kwargs"]["attn_backend"] == "math"
    assert by_model["s4d"]["attn_backend_used"] is None


def test_longer_lengths_are_skipped_after_a_spill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempted: list[int] = []

    def spilling(name: str, seq_len: int, *args: object, **kwargs: object) -> dict:
        attempted.append(seq_len)
        return {"model": name, "seq_len": seq_len, "train_step_ms": 1.0, "spilled_train": True}

    monkeypatch.setattr(efficiency, "benchmark_single", spilling)
    cfg = dict(_sweep_cfg(tmp_path, ["s4d"], [16, 32, 64]), stop_after_spill=True)
    results = efficiency.run_efficiency(cfg)
    assert attempted == [16]
    skipped = [e for e in results["entries"] if e.get("skipped")]
    assert [e["seq_len"] for e in skipped] == [32, 64]
    assert "spilled at T=16" in skipped[0]["skipped"]
    # Resume treats skipped points as done.
    efficiency.run_efficiency(cfg)
    assert attempted == [16]


def test_decode_can_be_switched_off() -> None:
    entry = efficiency.benchmark_single(
        "s4d",
        16,
        2,
        2,
        torch.device("cpu"),
        {"d_model": 16, "n_layers": 1, "d_state": 4},
        measure_decode=False,
    )
    assert "decode_per_token_ms" not in entry and entry["decode_mode"] == "not measured"


def test_p90_is_reported_only_from_ten_or_more_samples() -> None:
    from ssmbench.benchmarks.common import summarize_ms

    few = summarize_ms([1.0, 2.0, 3.0, 4.0, 5.0], "t")
    assert few["t_p90"] is None and few["t_median"] == 3.0 and few["t_n"] == 5
    many = summarize_ms([float(i) for i in range(1, 11)], "t")
    assert many["t_p90"] == 9.0


def test_fp16_training_steps_are_refused_without_a_grad_scaler() -> None:
    with pytest.raises(ValueError, match="GradScaler"):
        efficiency.benchmark_single(
            "s4d", 16, 2, 2, torch.device("cpu"), {"d_model": 8, "n_layers": 1}, precision="fp16"
        )


def test_skipped_points_are_measured_again_once_the_spill_is_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spill = {"on": True}
    attempted: list[int] = []

    def maybe_spilling(name: str, seq_len: int, *args: object, **kwargs: Any) -> dict:
        attempted.append(seq_len)
        return {
            "model": name,
            "seq_len": seq_len,
            "train_step_ms": 1.0,
            "spilled_train": spill["on"],
        }

    monkeypatch.setattr(efficiency, "benchmark_single", maybe_spilling)
    cfg: dict[str, Any] = {**_sweep_cfg(tmp_path, ["s4d"], [16, 32, 64]), "stop_after_spill": True}
    efficiency.run_efficiency(cfg)
    path = Path(cfg["out_path"])
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["entries"] = [e for e in saved["entries"] if e["seq_len"] != 16]  # drop the spill
    path.write_text(json.dumps(saved), encoding="utf-8")
    spill["on"] = False
    results = efficiency.run_efficiency(cfg)
    assert attempted == [16, 16, 32, 64]
    assert not any(e.get("skipped") for e in results["entries"])


def test_a_pinned_kernel_that_cannot_run_is_recorded_and_the_sweep_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempted: list[str] = []

    def picky(name: str, seq_len: int, *args: object, **kwargs: Any) -> dict:
        attempted.append(f"{kwargs.get('model_kwargs', {}).get('attn_backend')}@{seq_len}")
        if kwargs.get("model_kwargs", {}).get("attn_backend") == "cudnn":
            raise RuntimeError("No available kernel. Aborting execution.")
        return {"model": name, "seq_len": seq_len, "train_step_ms": 1.0}

    monkeypatch.setattr(efficiency, "benchmark_single", picky)
    cfg = _sweep_cfg(tmp_path, ["transformer_cudnn", "s4d"], [16, 32])
    cfg["variants"] = {"transformer_cudnn": {"backbone": "transformer", "attn_backend": "cudnn"}}
    results = efficiency.run_efficiency(cfg)
    unsupported = [e for e in results["entries"] if e.get("unsupported")]
    assert [e["seq_len"] for e in unsupported] == [16, 32]
    assert "No available kernel" in unsupported[0]["unsupported"]
    assert [e["seq_len"] for e in results["entries"] if e["model"] == "s4d"] == [16, 32]
    efficiency.run_efficiency(cfg)  # resume: unsupported points are not retried
    assert attempted.count("cudnn@16") == 1


def test_transformer_position_table_is_sized_to_the_point() -> None:
    entry = efficiency.benchmark_single(
        "transformer", 16, 2, 2, torch.device("cpu"), TINY_KWARGS["transformer"]
    )
    assert entry["model_kwargs"]["max_len"] == 16
