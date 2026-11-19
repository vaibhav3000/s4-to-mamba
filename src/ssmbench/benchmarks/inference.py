"""Inference benchmark: time to first token, decode latency and decode memory versus context.

For each (model variant, prompt length L), at one precision per sweep, a generation
session is: the prompt (batch_size, L) already on the device; embedding, ``prefill``
(the parallel form, returning the decode state), the LM head on the last position and
argmax give the first token; then ``new_tokens`` greedy decode steps, each one
``backbone.step`` on the previous token plus the head and argmax (docs/adr/0003).

Recorded per point, over ``repeats`` timed sessions after ``warmup`` untimed ones:

- ``ttft_ms``: time to first token (prompt on device to first token id), mean, std,
  median, n, and p90 when there are at least 10 sessions.
- ``decode_ms_per_token``: every decode step of every timed session, each timed with
  its own CUDA event pair (perf_counter on CPU): mean, std, median, p90, n. Steps run
  at contexts L to L + new_tokens; for the Transformer each step attends to all of them,
  for the SSMs the cost does not depend on L. Steps of one session are correlated, so
  the pooled spread is narrower than the spread between sessions.
- ``tokens_per_s``: batch_size * new_tokens divided by the median, over sessions, of the
  summed decode-step time (the prefill and first token are excluded).
- ``peak_allocated_mb`` / ``peak_reserved_mb``: the peak of the whole sessions, prefill
  included (weights, prompt activations, state), with the memory available at the start,
  free memory at the end and the spill flag (utils.peak_memory_stats). This is not decode
  memory.
- ``decode_state_bytes``: bytes of the decode state after the session (KV caches and
  RoPE tables, or recurrent states, conv histories and boundary inputs): the memory that
  decoding holds per sequence.
- ``decode_baseline_allocated_mb`` / ``decode_peak_above_baseline_mb`` (CUDA only): a
  separate untimed pass prefills, then opens a fresh peak window and decodes; the
  baseline is what is allocated when decoding starts (weights, prompt, state) and the
  second number is the transient memory decode steps add on top of it.
- ``attn_backend_prefill`` / ``attn_backend_step``: the SDPA kernel families a profiled
  prefill and step ran (Transformer variants; None for SSMs). A pinned backend that
  refuses the inputs is recorded as ``unsupported`` and the sweep continues.
- ``gpu_state``: SM and memory clocks, temperature, power and P-state when the point
  started (nvidia-smi, best effort).

Caveats for reading the numbers. At these model sizes (2 layers, d_model 256, batch 1)
a decode step is bound by Python and kernel-launch overhead rather than by memory
traffic, so per-token latency mostly counts kernels per step; the Transformer's growth
with context shows only where reading its KV cache costs more than that floor. Under
bf16 autocast the weight casts are cached for the whole session, so they fall inside the
time to first token and decode steps reuse them. The Transformer's KV cache follows the
autocast dtype (bf16) while SSM states stay float32, so decode-state sizes under bf16 are
not like-for-like.

Resume, provenance and OOM handling follow the efficiency sweep. Run with:

    ssmbench-inference --config configs/inference_bf16.yaml
"""

from __future__ import annotations

import argparse
import itertools
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from ..models import build_backbone
from ..models.attention import detect_attention_backend
from ..models.blocks import Backbone
from ..models.transformer_minimal import TransformerBackbone
from ..utils import (
    dump_json,
    load_config,
    peak_memory_stats,
    reset_peak_memory,
    resolve_device,
    run_provenance,
    seed_everything,
)
from .common import (
    check_precision,
    is_unsupported_kernel,
    precision_context,
    resolve_variant,
    state_bytes,
    summarize_ms,
)
from .efficiency import (
    DEFAULT_KWARGS,
    VOCAB,
    ProvenanceMismatchError,
    _is_oom,
    _load_prior_entries,
    _release_cuda_memory,
    config_sha256,
    gpu_state,
)


@dataclass
class Session:
    """One timed generation: TTFT, per-step milliseconds, the generated tokens
    (batch, new_tokens + 1) and the decode-state size at the end."""

    ttft_ms: float
    step_ms: list[float]
    tokens: torch.Tensor
    state_bytes: int


def generate_session(
    backbone: nn.Module,
    embed: nn.Module,
    head: nn.Module,
    prompt: torch.Tensor,
    new_tokens: int,
    device: torch.device,
    precision: str,
) -> Session:
    """Prefill the prompt, take the first token, then decode new_tokens greedy steps.

    Timestamps (CUDA events, or perf_counter off CUDA) mark the start, the first token
    and the end of every step; on CUDA they are read after one final synchronize.
    """
    if not isinstance(backbone, Backbone):
        raise TypeError(f"{type(backbone).__name__} has no decode interface")
    cuda = device.type == "cuda"
    marks: list = []

    def mark() -> None:
        if cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            marks.append(event)
        else:
            marks.append(time.perf_counter())

    if cuda:
        torch.cuda.synchronize(device)
    generated = []
    with torch.no_grad(), precision_context(device, precision):
        mark()
        features, state = backbone.prefill(embed(prompt), max_len=prompt.shape[1] + new_tokens)
        token = head(features[:, -1:]).argmax(-1)  # (B, 1)
        mark()
        generated.append(token)
        for _ in range(new_tokens):
            token = head(backbone.step(embed(token), state)).argmax(-1)
            mark()
            generated.append(token)
    if cuda:
        torch.cuda.synchronize(device)
        gaps = [a.elapsed_time(b) for a, b in itertools.pairwise(marks)]
    else:
        gaps = [(b - a) * 1000.0 for a, b in itertools.pairwise(marks)]
    return Session(
        ttft_ms=gaps[0],
        step_ms=gaps[1:],
        tokens=torch.cat(generated, dim=1),
        state_bytes=state_bytes(state),
    )


def _decode_memory(
    backbone: nn.Module,
    embed: nn.Module,
    head: nn.Module,
    prompt: torch.Tensor,
    new_tokens: int,
    device: torch.device,
    precision: str,
) -> dict[str, float] | None:
    """Untimed pass on CUDA: prefill, then a fresh peak window around the decode steps.

    Returns the memory allocated when decoding starts and the peak decode adds above
    it, in MiB; None off CUDA.
    """
    if device.type != "cuda" or not isinstance(backbone, Backbone):
        return None
    mib = 1024**2
    with torch.no_grad(), precision_context(device, precision):
        features, state = backbone.prefill(embed(prompt), max_len=prompt.shape[1] + new_tokens)
        token = head(features[:, -1:]).argmax(-1)
        del features
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        for _ in range(new_tokens):
            token = head(backbone.step(embed(token), state)).argmax(-1)
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
    return {
        "decode_baseline_allocated_mb": round(baseline / mib, 1),
        "decode_peak_above_baseline_mb": round((peak - baseline) / mib, 1),
    }


def measure_point(
    name: str,
    prompt_len: int,
    new_tokens: int,
    batch_size: int,
    repeats: int,
    warmup: int,
    device: torch.device,
    model_kwargs: dict | None = None,
    precision: str = "fp32",
) -> dict:
    """Measure one (architecture, prompt length) point; see the module docstring."""
    check_precision(precision)
    kwargs = dict(DEFAULT_KWARGS.get(name, {}))
    if name == "transformer":
        # Sinusoidal table sized to the session (prompt plus new tokens), so the
        # Transformer carries no 65536-position table that the SSMs do not have.
        kwargs["max_len"] = prompt_len + new_tokens
    if model_kwargs:
        kwargs.update(model_kwargs)
    seed_everything(0)
    backbone = build_backbone(name, kwargs).to(device).eval()
    embed = nn.Embedding(VOCAB, kwargs["d_model"]).to(device).eval()
    head = nn.Linear(kwargs["d_model"], VOCAB).to(device).eval()
    entry: dict = {
        "architecture": name,
        "seq_len": prompt_len,
        "new_tokens": new_tokens,
        "batch_size": batch_size,
        "precision": precision,
        "n_parameters": sum(p.numel() for p in backbone.parameters()),
        "model_kwargs": kwargs,
        "attn_backend_prefill": None,
        "attn_backend_step": None,
        "gpu_state": gpu_state(device),
    }
    if isinstance(backbone, TransformerBackbone):
        probe = torch.randint(1, VOCAB, (batch_size, min(prompt_len, 64)), device=device)
        holder: dict = {}

        def probe_prefill() -> None:
            with torch.no_grad(), precision_context(device, precision):
                holder["out"] = backbone.prefill(embed(probe), max_len=probe.shape[1] + 1)

        def probe_step() -> None:
            with torch.no_grad(), precision_context(device, precision):
                backbone.step(embed(probe[:, :1]), holder["out"][1])

        entry["attn_backend_prefill"] = detect_attention_backend(probe_prefill)
        entry["attn_backend_step"] = detect_attention_backend(probe_step)
        del holder

    prompt = torch.randint(1, VOCAB, (batch_size, prompt_len), device=device)
    available_at_start = reset_peak_memory(device)
    for _ in range(warmup):
        generate_session(backbone, embed, head, prompt, new_tokens, device, precision)
    sessions = [
        generate_session(backbone, embed, head, prompt, new_tokens, device, precision)
        for _ in range(repeats)
    ]
    stats = peak_memory_stats(device, available_at_start)
    decode_memory = _decode_memory(backbone, embed, head, prompt, new_tokens, device, precision)
    if decode_memory is not None:
        entry.update(decode_memory)
    entry.update(summarize_ms([s.ttft_ms for s in sessions], "ttft_ms"))
    entry.update(summarize_ms([t for s in sessions for t in s.step_ms], "decode_ms_per_token"))
    decode_total = statistics.median(sum(s.step_ms) for s in sessions)
    entry["tokens_per_s"] = round(batch_size * new_tokens / (decode_total / 1000.0), 2)
    entry["decode_state_bytes"] = sessions[-1].state_bytes
    if stats is not None:
        entry["peak_allocated_mb"] = stats["peak_allocated_mb"]
        entry["peak_reserved_mb"] = stats["peak_reserved_mb"]
        entry["device_total_mb"] = stats["device_total_mb"]
        entry["available_at_start_mb"] = stats["available_at_start_mb"]
        entry["free_at_end_mb"] = stats["free_at_end_mb"]
        entry["spilled"] = stats["spilled"]
    return entry


def run_inference(cfg: dict, force: bool = False) -> dict:
    """Run (or resume) the sweep in cfg, writing cfg["out_path"] after every point."""
    device = resolve_device(cfg.get("device", "auto"))
    out_path = Path(cfg["out_path"])
    precision = cfg.get("precision", "fp32")
    check_precision(precision)
    provenance = {
        **run_provenance(device),
        "config_sha256": config_sha256(cfg),
        "gpu_state_at_sweep_start": gpu_state(device),
    }
    settings = {"batch_size": cfg.get("batch_size", 1), "repeats": cfg.get("repeats", 5)}
    entries = _load_prior_entries(out_path, provenance, settings, force)
    done = {(str(e["model"]), int(e["seq_len"])) for e in entries}
    results = {
        "environment": {**provenance, "date": time.strftime("%Y-%m-%d")},
        "config": cfg,
        "entries": entries,
    }
    for label in cfg["models"]:
        architecture, model_kwargs = resolve_variant(label, cfg.get("variants"))
        limit = cfg.get("model_length_limits", {}).get(label)
        for L in [L for L in cfg["seq_lengths"] if limit is None or limit >= L]:
            if (label, L) in done:
                print(f"skipping {label} @ L={L} (already measured)", flush=True)
                continue
            print(f"inference {label} @ L={L} ({precision}) ...", flush=True)
            try:
                entry = measure_point(
                    architecture,
                    L,
                    cfg.get("new_tokens", 64),
                    settings["batch_size"],
                    settings["repeats"],
                    cfg.get("warmup", 1),
                    device,
                    model_kwargs=model_kwargs,
                    precision=precision,
                )
            except Exception as exc:
                if is_unsupported_kernel(exc):
                    entry = {
                        "architecture": architecture,
                        "seq_len": L,
                        "unsupported": f"{type(exc).__name__}: {exc}"[:300],
                    }
                    entry.update(model=label, **settings, provenance=provenance)
                    results["entries"].append(entry)
                    dump_json(results, out_path)
                    continue
                if not _is_oom(exc):
                    results["entries"].append(
                        {
                            "model": label,
                            "seq_len": L,
                            "error": f"{type(exc).__name__}: {exc}"[:300],
                            **settings,
                            "provenance": provenance,
                        }
                    )
                    dump_json(results, out_path)
                    raise RuntimeError(
                        f"{label} at L={L} failed with a non-OOM error; sweep aborted "
                        f"(recorded in {out_path})"
                    ) from exc
                entry = {"architecture": architecture, "seq_len": L, "oom": True}
                _release_cuda_memory(device)
            entry["model"] = label
            entry.update(settings, provenance=provenance)
            results["entries"].append(entry)
            dump_json(results, out_path)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="ssmbench-inference",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True, help="YAML config for the inference benchmark")
    ap.add_argument("--force", action="store_true", help="merge with entries from other code")
    args = ap.parse_args()
    try:
        run_inference(load_config(args.config), force=args.force)
    except ProvenanceMismatchError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
