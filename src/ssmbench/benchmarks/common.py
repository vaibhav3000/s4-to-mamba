"""Pieces shared by the benchmarks: timing, precision, variants, decode-state size."""

from __future__ import annotations

import contextlib
import statistics
import time
from collections.abc import Callable, Iterator
from typing import Any

import torch

from ..models.blocks import DecodeState

PRECISIONS = ("fp32", "bf16", "fp16")
_AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def check_precision(precision: str) -> None:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")


@contextlib.contextmanager
def precision_context(device: torch.device, precision: str) -> Iterator[None]:
    """fp32: no autocast. bf16/fp16: autocast to that dtype on the device (the SSM scans
    and their input projections still run in float32, see models/blocks.py)."""
    check_precision(precision)
    if precision == "fp32":
        yield
        return
    with torch.autocast(device_type=device.type, dtype=_AUTOCAST_DTYPES[precision]):
        yield


def time_calls(
    fn: Callable[[], object], repeats: int, warmup: int, device: torch.device
) -> list[float]:
    """Milliseconds of each of `repeats` calls of fn after `warmup` untimed calls.

    On CUDA every call sits between two CUDA events, after a synchronize, and is read
    after a final synchronize, so the numbers are GPU execution time, not launch time.
    Elsewhere: perf_counter around each call.
    """
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        events = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))
        torch.cuda.synchronize(device)
        return [s.elapsed_time(e) for s, e in events]
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0, 100]) of a non-empty list."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100.0 * (len(ordered) - 1))))
    return ordered[index]


# Fewer samples than this make the nearest-rank p90 the maximum, so it is not reported.
MIN_SAMPLES_FOR_P90 = 10


def summarize_ms(times: list[float], key: str) -> dict[str, float | int | None]:
    """{key}: mean, {key}_std (None for one sample), {key}_median, {key}_p90 (None below
    MIN_SAMPLES_FOR_P90 samples), {key}_n."""
    return {
        key: round(statistics.fmean(times), 4),
        f"{key}_std": round(statistics.stdev(times), 4) if len(times) > 1 else None,
        f"{key}_median": round(statistics.median(times), 4),
        f"{key}_p90": (
            round(percentile(times, 90.0), 4) if len(times) >= MIN_SAMPLES_FOR_P90 else None
        ),
        f"{key}_n": len(times),
    }


def is_unsupported_kernel(exc: BaseException) -> bool:
    """True for SDPA's refusal of a pinned backend for these inputs ("No available
    kernel"): a property of the backend and shape, recorded as unsupported rather than
    treated as a bug or retried."""
    return isinstance(exc, RuntimeError) and "no available kernel" in str(exc).lower()


def resolve_variant(label: str, variants: dict[str, dict[str, Any]] | None) -> tuple[str, dict]:
    """(architecture, kwargs) of a benchmark variant: variants[label] with an optional
    `backbone` key (default: the label itself, so bare architecture names still work)."""
    spec = dict((variants or {}).get(label, {}))
    return str(spec.pop("backbone", label)), spec


def state_bytes(state: DecodeState) -> int:
    """Bytes held by a decode state (KV caches, recurrent states, conv histories, tables)."""
    tensors = [t for block in state.blocks for t in block.values()]
    tensors += list(state.tables.values())
    return sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))
