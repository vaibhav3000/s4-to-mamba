"""Peak-memory bookkeeping, tested on CPU with a monkeypatched torch.cuda."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ssmbench.benchmarks import efficiency
from ssmbench.train import trainer
from ssmbench.utils import peak_memory_stats, reset_peak_memory

GIB = 1024**3
DEVICE_TOTAL = 6438780928  # RTX 4050 Laptop total memory as logged in experiments.log
CUDA = torch.device("cuda")


def _fake_cuda(
    monkeypatch: pytest.MonkeyPatch, allocated: int, reserved: int, total: int
) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device=None: calls.append("reset")
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: reserved)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(total_memory=total)
    )
    return calls


def test_reset_empties_the_cache_then_resets_peaks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_cuda(monkeypatch, allocated=0, reserved=0, total=DEVICE_TOTAL)
    reset_peak_memory(CUDA)
    assert calls == ["empty_cache", "reset"]


@pytest.mark.parametrize(
    ("reserved", "spilled"),
    [(5 * GIB, False), (DEVICE_TOTAL, False), (DEVICE_TOTAL + 1, True), (9 * GIB, True)],
)
def test_stats_report_mib_and_flag_reserved_above_device_total(
    monkeypatch: pytest.MonkeyPatch, reserved: int, spilled: bool
) -> None:
    _fake_cuda(monkeypatch, allocated=3 * GIB, reserved=reserved, total=DEVICE_TOTAL)
    stats = peak_memory_stats(CUDA)
    assert stats == {
        "peak_allocated_mb": 3072.0,
        "peak_reserved_mb": round(reserved / 1024**2, 1),
        "device_total_mb": 6140.5,
        "spilled": spilled,
    }


def test_helpers_do_nothing_off_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_cuda(monkeypatch, allocated=1, reserved=1, total=1)
    reset_peak_memory(torch.device("cpu"))
    assert peak_memory_stats(torch.device("cpu")) is None
    assert calls == []


FAKE_STATS = {
    "peak_allocated_mb": 100.0,
    "peak_reserved_mb": 7000.0,
    "device_total_mb": 6140.5,
    "spilled": True,
}


def test_trainer_resets_peaks_per_run_and_records_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each run_training call opens its own peak window, so models trained one after
    another in one process no longer inherit earlier peaks."""
    resets: list[torch.device] = []
    monkeypatch.setattr(trainer, "reset_peak_memory", resets.append)
    monkeypatch.setattr(trainer, "peak_memory_stats", lambda device: dict(FAKE_STATS))
    cfg = {
        "name": "parity_tiny",
        "task": {"type": "token_classification", "vocab_size": 2, "n_classes": 2},
        "data": {
            "name": "parity",
            "batch_size": 8,
            "seed": 0,
            "train_samples": 16,
            "test_samples": 8,
            "train": {"seq_len": 8},
            "test": {"seq_len": 8},
        },
        "train": {"epochs": 1, "lr": 1e-3, "seed": 0},
        "model": {
            "name": "transformer",
            "kwargs": {"d_model": 16, "n_layers": 1, "n_heads": 2, "max_len": 16},
        },
        "device": "cpu",
    }
    for _ in range(2):
        results = trainer.run_training(cfg, tmp_path)
        assert results["peak_gpu_mb"] == 100.0
        assert results["peak_gpu_reserved_mb"] == 7000.0
        assert results["device_total_mb"] == 6140.5
        assert results["gpu_memory_spilled"] is True
    assert resets == [torch.device("cpu"), torch.device("cpu")]


def test_benchmark_records_memory_and_spill_flags_per_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resets: list[torch.device] = []
    phase_stats = iter([dict(FAKE_STATS), dict(FAKE_STATS, peak_reserved_mb=200.0, spilled=False)])
    monkeypatch.setattr(efficiency, "reset_peak_memory", resets.append)
    monkeypatch.setattr(efficiency, "peak_memory_stats", lambda device: next(phase_stats))
    entry = efficiency.benchmark_single(
        "s4d",
        seq_len=16,
        batch_size=2,
        repeats=2,
        device=torch.device("cpu"),
        model_kwargs={"d_model": 32, "n_layers": 1, "d_state": 4},
    )
    assert len(resets) == 2  # one window for training, one for inference
    assert entry["peak_train_mb"] == 100.0 and entry["peak_train_reserved_mb"] == 7000.0
    assert entry["spilled_train"] is True
    assert entry["peak_infer_reserved_mb"] == 200.0 and entry["spilled_infer"] is False
    assert entry["device_total_mb"] == 6140.5
