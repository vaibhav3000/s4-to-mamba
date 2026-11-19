"""Efficiency benchmark: latency, memory and throughput as functions of sequence length.

Measures, per (architecture, sequence length):

- ``train_step_ms``: forward + backward + optimizer step of a small language
  model (embedding, backbone, linear LM head over VOCAB tokens, cross-entropy)
  over `batch_size` sequences; peak CUDA memory recorded alongside.
- ``infer_ms``: full forward pass including the LM head, no grad, batch 1.
- ``decode_ms``: autoregressive one-token-at-a-time generation cost.
  * mamba / s4d: true incremental decode with O(1) recurrent state via
    ``block.step()`` (state carried between calls).
  * mamba2 / mamba3: sequential-scan recurrence over the whole sequence,
    reported as per-token cost (the recurrence IS the decode computation).
  * transformer: full prefix recompute — the educational implementation has no
    KV cache, so this is its honest per-token cost. Labeled accordingly.
- ``n_parameters`` and throughput in tokens/s.

All timings use CUDA events around synchronized regions; nothing is estimated
or copied from papers. Every number in the report can be regenerated with:

    python scripts/run_efficiency.py --config configs/efficiency.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import nn

from ..models import build_backbone
from ..utils import (
    dump_json,
    peak_memory_stats,
    reset_peak_memory,
    resolve_device,
    seed_everything,
)

DEFAULT_KWARGS: dict[str, dict] = {
    "transformer": dict(d_model=256, n_layers=2, n_heads=4),
    "s4d": dict(d_model=256, n_layers=2, d_state=16),
    "mamba": dict(d_model=256, n_layers=2, d_state=16),
    "mamba2": dict(d_model=256, n_layers=2, d_state=32, headdim=64, scan_mode="chunked"),
    "mamba3": dict(d_model=256, n_layers=2, d_state=32, headdim=64),
}

VOCAB = 32


def _time_gpu(
    fn: Callable[[], None], repeats: int, warmup: int = 2, device: str = "cuda"
) -> tuple[float, float | None]:
    """Return (mean_ms, std_ms) of fn(), CUDA-event timed when possible.

    std_ms is None when there is a single timed repeat.
    """
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    else:
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
    t = torch.tensor(times)
    return float(t.mean()), (float(t.std()) if len(times) > 1 else None)


def _record_memory(entry: dict, phase: str, stats: dict[str, float | bool] | None) -> None:
    """Store peak allocated/reserved memory, device total and the spill flag for a phase.

    peak_<phase>_mb keeps its historical meaning (peak allocated). spilled_<phase> is
    True when the peak reserved memory exceeded the device's total memory, which on
    Windows means the driver paged to system memory and the timing is not a VRAM one.
    """
    if stats is None:
        return
    entry[f"peak_{phase}_mb"] = stats["peak_allocated_mb"]
    entry[f"peak_{phase}_reserved_mb"] = stats["peak_reserved_mb"]
    entry["device_total_mb"] = stats["device_total_mb"]
    entry[f"spilled_{phase}"] = stats["spilled"]


def _is_oom(exc: BaseException) -> bool:
    """True only for allocator out-of-memory errors: torch.OutOfMemoryError, or an
    accelerator/runtime error whose message says "out of memory". Anything else
    (illegal address, launch failure, a shape bug) is not a hardware limit."""
    return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _release_cuda_memory(device: torch.device) -> None:
    """Return cached blocks to the driver after an OOM; no-op off CUDA."""
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()


class _StepBlock(Protocol):
    """A block with an O(1)-state decode path (Mamba-1 and S4D blocks)."""

    def init_state(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]: ...

    def step(self, x_t: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor: ...


def _decode_stepwise(
    backbone: nn.Module, embed: nn.Module, head: nn.Module, seq_len: int, device: torch.device
) -> None:
    """True incremental decode for backbones whose blocks expose step()."""
    with torch.no_grad():
        blocks = backbone.blocks
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError(f"{type(backbone).__name__} has no ModuleList of blocks to step")
        steppers = [cast(_StepBlock, block) for block in blocks]
        states = [block.init_state(1, device) for block in steppers]
        for _ in range(seq_len):
            token = torch.randint(1, VOCAB, (1, 1), device=device)
            h = embed(token)
            for block, state in zip(steppers, states, strict=True):
                h = block.step(h, state)
            head(h)


def _decode_sequential(
    backbone: nn.Module, embed: nn.Module, head: nn.Module, seq_len: int, device: torch.device
) -> None:
    """Run the recurrent scan over a full sequence (per-token cost = time/T)."""
    with torch.no_grad():
        tokens = torch.randint(1, VOCAB, (1, seq_len), device=device)
        head(backbone(embed(tokens)))


def benchmark_single(
    name: str,
    seq_len: int,
    batch_size: int,
    repeats: int,
    device: torch.device,
    model_kwargs: dict | None = None,
) -> dict:
    kwargs = dict(DEFAULT_KWARGS.get(name, {}))
    if model_kwargs:
        kwargs.update(model_kwargs)
    seed_everything(0)
    backbone = build_backbone(name, kwargs).to(device)
    embed = nn.Embedding(VOCAB, kwargs["d_model"]).to(device)
    head = nn.Linear(kwargs["d_model"], VOCAB).to(device)  # LM head: logits over VOCAB
    lm_modules = (embed, backbone, head)
    entry: dict = {
        "model": name,
        "seq_len": seq_len,
        "batch_size": batch_size,
        # Backbone only, so the count compares mixers; the embedding and LM head add
        # the same 2 * VOCAB * d_model + VOCAB for every model of a given width.
        "n_parameters": sum(p.numel() for p in backbone.parameters()),
        "n_parameters_with_embed_and_head": sum(
            p.numel() for m in lm_modules for p in m.parameters()
        ),
        "model_kwargs": kwargs,
    }

    # ---- training step: embedding -> backbone -> LM head -> cross-entropy ----
    try:
        tokens = torch.randint(1, VOCAB, (batch_size, seq_len), device=device)
        params = [p for m in lm_modules for p in m.parameters()]
        opt = torch.optim.AdamW(params, lr=1e-4)
        target = torch.randint(1, VOCAB, (batch_size, seq_len), device=device)

        def train_step() -> None:
            logits = head(backbone(embed(tokens)))  # (B, T, VOCAB)
            loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        reset_peak_memory(device)
        mean_ms, std_ms = _time_gpu(train_step, repeats, device=device.type)
        entry["train_step_ms"] = round(mean_ms, 2)
        entry["train_step_ms_std"] = round(std_ms, 2) if std_ms is not None else None
        _record_memory(entry, "train", peak_memory_stats(device))
        entry["train_tokens_per_s"] = round(batch_size * seq_len / (mean_ms / 1000))
        del opt, tokens, target
    except Exception as exc:
        if not _is_oom(exc):
            raise
        entry["oom_train"] = True
        _release_cuda_memory(device)
        return entry

    # Inference and decode are timed in eval mode (dropout off), as at deployment,
    # and without the training gradients occupying memory.
    for m in lm_modules:
        m.eval()
        m.zero_grad(set_to_none=True)

    # ---- full forward, no grad, batch 1 ----
    try:
        tokens = torch.randint(1, VOCAB, (1, seq_len), device=device)

        def infer() -> None:
            with torch.no_grad():
                head(backbone(embed(tokens)))

        reset_peak_memory(device)
        mean_ms, _ = _time_gpu(infer, repeats, device=device.type)
        entry["infer_ms"] = round(mean_ms, 2)
        _record_memory(entry, "infer", peak_memory_stats(device))
        del tokens
    except Exception as exc:
        if not _is_oom(exc):
            raise
        entry["oom_infer"] = True
        _release_cuda_memory(device)
        return entry

    # ---- decode ----
    try:
        rep_d = max(repeats // 2, 1)
        if name in ("mamba", "s4d"):

            def decode_steps() -> None:
                _decode_stepwise(backbone, embed, head, seq_len, device)

            entry["decode_mode"] = "stepwise (O(1) state)"
            mean_ms, _ = _time_gpu(decode_steps, rep_d, warmup=1, device=device.type)
            entry["decode_ms"] = round(mean_ms, 2)
            entry["decode_per_token_ms"] = round(mean_ms / seq_len, 4)
        elif name in ("mamba2", "mamba3"):

            def decode_full() -> None:
                _decode_sequential(backbone, embed, head, seq_len, device)

            entry["decode_mode"] = "sequential scan over full sequence (per-token = time/T)"
            mean_ms, _ = _time_gpu(decode_full, rep_d, warmup=1, device=device.type)
            entry["decode_per_token_ms"] = round(mean_ms / seq_len, 4)
        else:  # transformer: no KV cache, so each new token re-runs the full prefix
            tokens = torch.randint(1, VOCAB, (1, seq_len), device=device)

            def decode_tr() -> None:
                with torch.no_grad():
                    head(backbone(embed(tokens)))

            entry["decode_mode"] = "full-prefix recompute per new token (no KV cache)"
            mean_ms, _ = _time_gpu(decode_tr, rep_d, warmup=1, device=device.type)
            # Generating ONE more token costs a full forward over the prefix.
            entry["decode_per_token_ms"] = round(mean_ms, 4)
    except Exception as exc:
        if not _is_oom(exc):
            raise
        entry["oom_decode"] = True
        _release_cuda_memory(device)

    return entry


def run_efficiency(cfg: dict) -> dict:
    device = resolve_device(cfg.get("device", "auto"))
    out_path = Path(cfg["out_path"])
    entries: list[dict] = []
    done: set[tuple[str, int]] = set()
    if out_path.exists():
        # Resume support: keep prior measurements (e.g. an expensive transformer
        # sweep) and only run what is missing.
        try:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
            entries = list(prior.get("entries", []))
            done = {(e.get("model"), e.get("seq_len")) for e in entries}
        except (json.JSONDecodeError, OSError):
            entries, done = [], set()
    results = {
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "date": time.strftime("%Y-%m-%d"),
        },
        "config": cfg,
        "entries": entries,
    }
    for name in cfg["models"]:
        # Optional per-model length cap: the educational sequential scans
        # (mamba, mamba3) have python-loop step counts proportional to T, so
        # very long sequences are measured only for the vectorized forms.
        limit = cfg.get("model_length_limits", {}).get(name)
        lengths = [L for L in cfg["seq_lengths"] if limit is None or L <= limit]
        for L in lengths:
            if (name, L) in done:
                print(f"skipping {name} @ T={L} (already measured)", flush=True)
                continue
            print(f"benchmarking {name} @ T={L} ...", flush=True)
            try:
                entry = benchmark_single(
                    name, L, cfg.get("batch_size", 8), cfg.get("repeats", 5), device
                )
            except Exception as exc:
                if not _is_oom(exc):
                    # Not a hardware limit: record it and stop, so a code bug or a
                    # broken CUDA context is never plotted as an OOM.
                    error = f"{type(exc).__name__}: {exc}"[:300]
                    results["entries"].append({"model": name, "seq_len": L, "error": error})
                    dump_json(results, out_path)
                    raise RuntimeError(
                        f"{name} at T={L} failed with a non-OOM error; sweep aborted "
                        f"(recorded in {out_path})"
                    ) from exc
                entry = {"model": name, "seq_len": L, "oom": True}
            results["entries"].append(entry)
            dump_json(results, out_path)
            if any(entry.get(k) for k in ("oom", "oom_train", "oom_infer", "oom_decode")):
                # A context that cannot free memory after an OOM raises here and
                # stops the sweep, instead of poisoning the next measurements.
                _release_cuda_memory(device)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML config for the efficiency benchmark")
    args = ap.parse_args()
    from ..utils import load_config

    cfg = load_config(args.config)
    run_efficiency(cfg)


if __name__ == "__main__":
    main()
