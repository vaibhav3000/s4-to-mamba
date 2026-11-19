"""Efficiency benchmark: latency, memory and throughput as functions of sequence length.

Measures, per (architecture, sequence length):

- ``train_step_ms``: forward + backward + optimizer step of a small language
  model (embedding, backbone, linear LM head over VOCAB tokens, cross-entropy)
  over `batch_size` sequences, in fp32. Peak allocated and peak reserved CUDA
  memory, the memory available when the phase started and the device total are
  recorded, and ``spilled_train`` flags a reserved peak within a 7% safety margin of
  what was available, or under 2% of the card free at the end of the phase (on
  Windows the driver then pages to system memory, so the timing is not a VRAM
  number).
- ``infer_ms``: full forward pass including the LM head, no grad, batch 1, in
  eval mode with the training gradients dropped; memory as above.
- decode cost per generated token, labeled by ``decode_mode``:
  * mamba / s4d: true incremental decode with an O(1) recurrent state via
    ``block.step()``; ``decode_per_token_ms`` = time for T steps / T.
  * transformer: this implementation has no KV cache, so one more token re-runs
    the whole prefix; ``decode_per_token_ms`` = one full forward over T tokens.
  * mamba2 / mamba3: no step() path, so no decode cost is measured.
    ``prefill_per_token_ms`` = one full forward over T tokens / T is recorded
    instead; it is a prefill throughput number, not a decode latency.
- ``n_parameters`` (backbone only), ``n_parameters_with_embed_and_head`` and
  throughput in tokens/s.

Timings use CUDA events around synchronized regions after warm-up (perf_counter
on CPU); nothing is estimated or copied from papers. Only out-of-memory errors
are recorded as OOM; any other error is recorded and stops the sweep. Every
entry stores its provenance (git commit, dirty flag, source hash, torch, CUDA,
Python, platform, GPU, NVIDIA driver, timestamp, a hash of the measurement
settings in the config, and the GPU clocks, temperature and power at sweep start
when nvidia-smi is available), and a resumed sweep reuses an entry only if it
was measured with the same batch size, repeat count, source, torch build, GPU
and config settings (--force overrides). Run with:

    ssmbench-efficiency --config configs/efficiency.yaml

The committed results/efficiency_gpu.json predates these rules and the model
fixes; see results/PROVENANCE.md.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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
    load_config,
    nvidia_smi_query,
    peak_memory_stats,
    reset_peak_memory,
    resolve_device,
    run_provenance,
    seed_everything,
)

DEFAULT_KWARGS: dict[str, dict] = {
    "transformer": {"d_model": 256, "n_layers": 2, "n_heads": 4},
    "s4d": {"d_model": 256, "n_layers": 2, "d_state": 16},
    "mamba": {"d_model": 256, "n_layers": 2, "d_state": 16},
    "mamba2": {"d_model": 256, "n_layers": 2, "d_state": 32, "headdim": 64, "scan_mode": "chunked"},
    "mamba3": {"d_model": 256, "n_layers": 2, "d_state": 32, "headdim": 64},
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


def _record_memory(entry: dict, phase: str, stats: dict[str, float | bool | None] | None) -> None:
    """Store peak allocated/reserved memory, device total, availability and spill flag.

    peak_<phase>_mb keeps its historical meaning (peak allocated). spilled_<phase> is
    True when the phase most likely ran past the card's memory (see
    utils.peak_memory_stats: a reserved peak within a safety margin of the memory
    available at the start, or almost no free memory at the end), which on Windows
    means the driver paged to system memory and the timing is not a VRAM one.
    """
    if stats is None:
        return
    entry[f"peak_{phase}_mb"] = stats["peak_allocated_mb"]
    entry[f"peak_{phase}_reserved_mb"] = stats["peak_reserved_mb"]
    entry["device_total_mb"] = stats["device_total_mb"]
    entry[f"available_at_{phase}_start_mb"] = stats["available_at_start_mb"]
    entry[f"free_at_{phase}_end_mb"] = stats["free_at_end_mb"]
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
    backbone: nn.Module, embed: nn.Module, head: nn.Module, tokens: torch.Tensor
) -> None:
    """True incremental decode of tokens (1, T) for backbones whose blocks expose
    step(). The timed session includes one init_state() per block (for S4D, the
    one-off discretization) and T steps; the tokens are generated beforehand."""
    with torch.no_grad():
        blocks = backbone.blocks
        if not isinstance(blocks, nn.ModuleList):
            raise TypeError(f"{type(backbone).__name__} has no ModuleList of blocks to step")
        steppers = [cast(_StepBlock, block) for block in blocks]
        states = [block.init_state(1, tokens.device) for block in steppers]
        for t in range(tokens.shape[1]):
            h = embed(tokens[:, t : t + 1])
            for block, state in zip(steppers, states, strict=True):
                h = block.step(h, state)
            head(h)


def _prefill(backbone: nn.Module, embed: nn.Module, head: nn.Module, tokens: torch.Tensor) -> None:
    """One full forward over a prompt of shape (1, T)."""
    with torch.no_grad():
        head(backbone(embed(tokens)))


def _measure_train_step(
    embed: nn.Module,
    backbone: nn.Module,
    head: nn.Module,
    batch_size: int,
    seq_len: int,
    repeats: int,
    device: torch.device,
) -> dict:
    """Time AdamW steps of embed -> backbone -> head with cross-entropy.

    The tokens, targets and optimizer state live only inside this call, so they are
    freed before inference is timed.
    """
    tokens = torch.randint(1, VOCAB, (batch_size, seq_len), device=device)
    target = torch.randint(1, VOCAB, (batch_size, seq_len), device=device)
    params = [p for m in (embed, backbone, head) for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-4)

    def train_step() -> None:
        logits = head(backbone(embed(tokens)))  # (B, T, VOCAB)
        loss = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), target.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    available_at_start = reset_peak_memory(device)
    mean_ms, std_ms = _time_gpu(train_step, repeats, device=device.type)
    result = {
        "train_step_ms": round(mean_ms, 2),
        "train_step_ms_std": round(std_ms, 2) if std_ms is not None else None,
    }
    _record_memory(result, "train", peak_memory_stats(device, available_at_start))
    result["train_tokens_per_s"] = round(batch_size * seq_len / (mean_ms / 1000))
    return result


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
        entry.update(
            _measure_train_step(embed, backbone, head, batch_size, seq_len, repeats, device)
        )
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

        available_at_start = reset_peak_memory(device)
        mean_ms, _ = _time_gpu(infer, repeats, device=device.type)
        entry["infer_ms"] = round(mean_ms, 2)
        _record_memory(entry, "infer", peak_memory_stats(device, available_at_start))
    except Exception as exc:
        if not _is_oom(exc):
            raise
        entry["oom_infer"] = True
        _release_cuda_memory(device)
        return entry

    # ---- decode ----
    try:
        rep_d = max(repeats // 2, 1)
        prompt = torch.randint(1, VOCAB, (1, seq_len), device=device)  # outside timing
        if name in ("mamba", "s4d"):

            def decode_steps() -> None:
                _decode_stepwise(backbone, embed, head, prompt)

            entry["decode_mode"] = "stepwise (O(1) state)"
            mean_ms, _ = _time_gpu(decode_steps, rep_d, warmup=1, device=device.type)
            entry["decode_ms"] = round(mean_ms, 2)
            entry["decode_per_token_ms"] = round(mean_ms / seq_len, 4)
        elif name in ("mamba2", "mamba3"):

            def prefill() -> None:
                _prefill(backbone, embed, head, prompt)

            entry["decode_mode"] = (
                "not measured: no step() path; prefill_per_token_ms is one full forward / T"
            )
            mean_ms, _ = _time_gpu(prefill, rep_d, warmup=1, device=device.type)
            entry["prefill_per_token_ms"] = round(mean_ms / seq_len, 4)
        else:  # transformer: no KV cache, so each new token re-runs the full prefix

            def decode_tr() -> None:
                _prefill(backbone, embed, head, prompt)

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


class ProvenanceMismatchError(RuntimeError):
    """The results file holds entries measured with different code."""


# Config keys that only choose which points a sweep runs; they do not change how a
# point is measured, so the config hash that resume compares leaves them out.
POINT_SELECTION_KEYS = frozenset({"out_path", "models", "seq_lengths", "model_length_limits"})
# nvidia-smi fields recorded once at the start of a sweep (best effort).
GPU_STATE_FIELDS = [
    "clocks.sm",
    "clocks.mem",
    "clocks.max.sm",
    "clocks.max.mem",
    "temperature.gpu",
    "power.draw",
    "pstate",
]
# Run settings, stored on every entry, that must match for a prior entry to be reused.
RESUME_SETTING_KEYS = ("batch_size", "repeats")
# Provenance fields that must match too.
RESUME_PROVENANCE_KEYS = ("source_sha256", "torch", "cuda", "gpu", "config_sha256")


def config_sha256(cfg: dict) -> str:
    """SHA-256 of the measurement settings in cfg (every key except POINT_SELECTION_KEYS)."""
    measured = {k: v for k, v in cfg.items() if k not in POINT_SELECTION_KEYS}
    return hashlib.sha256(json.dumps(measured, sort_keys=True, default=str).encode()).hexdigest()


def gpu_state(device: torch.device) -> dict[str, str] | None:
    """Clocks, temperature, power draw and P-state of the GPU from nvidia-smi, or None
    off CUDA or when nvidia-smi is unavailable. Never raises."""
    if device.type != "cuda":
        return None
    return nvidia_smi_query(GPU_STATE_FIELDS)


def _resume_mismatch(entry: dict, provenance: dict, settings: dict) -> str | None:
    """Why entry cannot be reused in this run, or None if it can."""
    for key in RESUME_SETTING_KEYS:
        if entry.get(key) != settings[key]:
            return key
    prior = entry.get("provenance") or {}
    for key in RESUME_PROVENANCE_KEYS:
        if prior.get(key) != provenance[key]:
            return key
    return None


def _load_prior_entries(
    out_path: Path, provenance: dict, settings: dict, force: bool
) -> list[dict]:
    """Entries already in out_path that may be kept when resuming a sweep.

    Resuming is for finishing an interrupted sweep of the SAME setup: the same
    batch_size and repeats, and the same package source, torch, CUDA, GPU and
    measurement settings in the config (RESUME_PROVENANCE_KEYS). Any other entry, including those written before
    provenance existed, makes this raise unless force is set. Error entries are
    dropped so those points are measured again.
    """
    if not out_path.exists():
        return []
    prior = json.loads(out_path.read_text(encoding="utf-8"))
    entries = [e for e in prior.get("entries", []) if "error" not in e]
    dropped = len(prior.get("entries", [])) - len(entries)
    if dropped:
        print(f"re-measuring {dropped} point(s) that ended in an error", flush=True)
    foreign = []
    for e in entries:
        reason = _resume_mismatch(e, provenance, settings)
        if reason is not None:
            foreign.append(f"{e.get('model')}@{e.get('seq_len')} ({reason} differs)")
    if foreign and not force:
        raise ProvenanceMismatchError(
            f"{out_path} holds {len(foreign)} entries measured with other code, another "
            f"environment or other settings, or without provenance "
            f"({', '.join(foreign[:5])}{', ...' if len(foreign) > 5 else ''}). "
            "Resuming would mix them in one table. Move the file away to start a clean "
            "sweep, or pass --force to merge anyway."
        )
    if foreign:
        print(f"warning: merging {len(foreign)} mismatched entries (--force)", flush=True)
    return entries


def run_efficiency(cfg: dict, force: bool = False) -> dict:
    """Run (or resume) the sweep in cfg and write cfg["out_path"] after every point.

    Every entry carries the provenance of the run that measured it; "environment"
    describes the most recent run. force=True merges with entries from other code.
    """
    device = resolve_device(cfg.get("device", "auto"))
    out_path = Path(cfg["out_path"])
    provenance = {
        **run_provenance(device),
        "config_sha256": config_sha256(cfg),
        "gpu_state_at_sweep_start": gpu_state(device),
    }
    settings = {"batch_size": cfg.get("batch_size", 8), "repeats": cfg.get("repeats", 5)}
    entries = _load_prior_entries(out_path, provenance, settings, force)
    done = {(str(e["model"]), int(e["seq_len"])) for e in entries}
    results = {
        "environment": {**provenance, "date": time.strftime("%Y-%m-%d")},
        "config": cfg,
        "entries": entries,
    }
    for name in cfg["models"]:
        # Optional per-model length cap: the educational sequential scans
        # (mamba, mamba3) have python-loop step counts proportional to T, so
        # very long sequences are measured only for the vectorized forms.
        limit = cfg.get("model_length_limits", {}).get(name)
        lengths = [L for L in cfg["seq_lengths"] if limit is None or limit >= L]
        for L in lengths:
            if (name, L) in done:
                print(f"skipping {name} @ T={L} (already measured)", flush=True)
                continue
            print(f"benchmarking {name} @ T={L} ...", flush=True)
            try:
                entry = benchmark_single(
                    name, L, settings["batch_size"], settings["repeats"], device
                )
            except Exception as exc:
                if not _is_oom(exc):
                    # Not a hardware limit: record it and stop, so a code bug or a
                    # broken CUDA context is never plotted as an OOM.
                    error = f"{type(exc).__name__}: {exc}"[:300]
                    results["entries"].append(
                        {
                            "model": name,
                            "seq_len": L,
                            "error": error,
                            **settings,
                            "provenance": provenance,
                        }
                    )
                    dump_json(results, out_path)
                    raise RuntimeError(
                        f"{name} at T={L} failed with a non-OOM error; sweep aborted "
                        f"(recorded in {out_path})"
                    ) from exc
                entry = {"model": name, "seq_len": L, "oom": True}
            entry.update(settings, provenance=provenance)
            results["entries"].append(entry)
            dump_json(results, out_path)
            if any(entry.get(k) for k in ("oom", "oom_train", "oom_infer", "oom_decode")):
                # A context that cannot free memory after an OOM raises here and
                # stops the sweep, instead of poisoning the next measurements.
                _release_cuda_memory(device)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="ssmbench-efficiency",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True, help="YAML config for the efficiency benchmark")
    ap.add_argument(
        "--force",
        action="store_true",
        help="resume into a results file that holds entries measured with other code",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    try:
        run_efficiency(cfg, force=args.force)
    except ProvenanceMismatchError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
