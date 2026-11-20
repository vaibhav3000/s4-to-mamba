"""Peak-memory bookkeeping, tested on CPU with a monkeypatched torch.cuda."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ssmbench.benchmarks import efficiency
from ssmbench.train import trainer
from ssmbench.utils import (
    LOW_FREE_FRACTION,
    SPILL_MARGIN_FRACTION,
    peak_memory_stats,
    reset_peak_memory,
)

GIB = 1024**3
DEVICE_TOTAL = 6438780928  # RTX 4050 Laptop total memory as logged in experiments.log
CUDA = torch.device("cuda")


def _fake_cuda(
    monkeypatch: pytest.MonkeyPatch,
    allocated: int,
    reserved: int,
    total: int,
    free: int = 0,
    reserved_at_reset: int = 0,
) -> list[str]:
    """Fake torch.cuda; `free` is what mem_get_info reports at every call."""
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
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (free, total))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: reserved_at_reset)
    return calls


def test_reset_empties_the_cache_then_resets_peaks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_cuda(
        monkeypatch,
        allocated=0,
        reserved=0,
        total=DEVICE_TOTAL,
        free=4 * GIB,
        reserved_at_reset=GIB,
    )
    available = reset_peak_memory(CUDA)
    assert calls == ["empty_cache", "reset"]
    assert available == 5 * GIB  # driver-free memory plus what this process holds


MIB = 1024**2


def test_spill_constants() -> None:
    """Reserved memory leaves out the CUDA context and lazily loaded cuBLAS/cuFFT, so a
    reserved peak within 7% of the limit already counts as a spill; less than 2% of the
    card free at the end of a phase counts too."""
    assert SPILL_MARGIN_FRACTION == 0.07 and LOW_FREE_FRACTION == 0.02


@pytest.mark.parametrize(
    ("reserved_mib", "spilled"),
    [(5120, False), (5700, False), (5750, True), (6141, True), (9216, True)],
)
def test_stats_flag_a_reserved_peak_within_the_margin_of_the_device_total(
    monkeypatch: pytest.MonkeyPatch, reserved_mib: int, spilled: bool
) -> None:
    """Limit = 6140.5 MiB minus 7% = 5710.7 MiB; plenty of memory is free at the end."""
    reserved = reserved_mib * MIB
    _fake_cuda(
        monkeypatch, allocated=3 * GIB, reserved=reserved, total=DEVICE_TOTAL, free=DEVICE_TOTAL
    )
    stats = peak_memory_stats(CUDA)
    assert stats == {
        "peak_allocated_mb": 3072.0,
        "peak_reserved_mb": float(reserved_mib),
        "device_total_mb": 6140.5,
        "available_at_start_mb": None,
        "free_at_end_mb": 6140.5,
        "spilled": spilled,
    }


@pytest.mark.parametrize(("reserved_mib", "spilled"), [(3600, False), (3700, True)])
def test_spill_threshold_is_what_was_available_when_the_window_opened(
    monkeypatch: pytest.MonkeyPatch, reserved_mib: int, spilled: bool
) -> None:
    """With about 2 GiB of the 6 GB card held by other processes, 4096 MiB were
    available at the start, so the limit is 4096 - 0.07 * 6140.5 = 3666.2 MiB, far below
    the device total."""
    reserved = reserved_mib * MIB
    _fake_cuda(monkeypatch, allocated=GIB, reserved=reserved, total=DEVICE_TOTAL, free=2 * GIB)
    stats = peak_memory_stats(CUDA, available_at_start=4 * GIB)
    assert stats is not None
    assert reserved < DEVICE_TOTAL
    assert stats["available_at_start_mb"] == 4096.0
    assert stats["spilled"] is spilled


@pytest.mark.parametrize(("free_mib", "spilled"), [(100, True), (200, False)])
def test_near_zero_free_memory_at_the_end_of_a_phase_counts_as_a_spill(
    monkeypatch: pytest.MonkeyPatch, free_mib: int, spilled: bool
) -> None:
    """A small reserved peak, but the driver reports less than 2% of the card (122.8 MiB)
    free when the phase ends: other memory use pushed the card to its limit."""
    _fake_cuda(monkeypatch, allocated=GIB, reserved=GIB, total=DEVICE_TOTAL, free=free_mib * MIB)
    stats = peak_memory_stats(CUDA, available_at_start=5 * GIB)
    assert stats is not None
    assert stats["free_at_end_mb"] == float(free_mib)
    assert stats["spilled"] is spilled


def test_helpers_do_nothing_off_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_cuda(monkeypatch, allocated=1, reserved=1, total=1)
    reset_peak_memory(torch.device("cpu"))
    assert peak_memory_stats(torch.device("cpu")) is None
    assert calls == []


FAKE_STATS = {
    "peak_allocated_mb": 100.0,
    "peak_reserved_mb": 7000.0,
    "device_total_mb": 6140.5,
    "available_at_start_mb": 5000.0,
    "free_at_end_mb": 50.0,
    "spilled": True,
}


def test_trainer_resets_peaks_per_run_and_records_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each run_training call opens its own peak window, so models trained one after
    another in one process no longer inherit earlier peaks."""
    resets: list[torch.device] = []
    monkeypatch.setattr(trainer, "reset_peak_memory", resets.append)
    monkeypatch.setattr(
        trainer, "peak_memory_stats", lambda device, available=None: dict(FAKE_STATS)
    )
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
        assert results["gpu_available_at_start_mb"] == 5000.0
        assert results["gpu_free_at_end_mb"] == 50.0
    assert resets == [torch.device("cpu"), torch.device("cpu")]


def test_benchmark_records_memory_and_spill_flags_per_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resets: list[torch.device] = []
    phase_stats = iter([dict(FAKE_STATS), dict(FAKE_STATS, peak_reserved_mb=200.0, spilled=False)])
    monkeypatch.setattr(efficiency, "reset_peak_memory", resets.append)
    monkeypatch.setattr(
        efficiency, "peak_memory_stats", lambda device, available=None: next(phase_stats)
    )
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
    assert entry["spilled_train"] is True and entry["available_at_train_start_mb"] == 5000.0
    assert entry["free_at_train_end_mb"] == 50.0
    assert entry["peak_infer_reserved_mb"] == 200.0 and entry["spilled_infer"] is False
    assert entry["device_total_mb"] == 6140.5
