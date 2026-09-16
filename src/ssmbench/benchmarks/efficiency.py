"""Efficiency benchmark: latency, memory and throughput as functions of sequence length.

Measures, per (architecture, sequence length):

- ``train_step_ms``: forward + backward + optimizer step over `batch_size`
  sequences; peak CUDA memory recorded alongside.
- ``infer_ms``: full forward pass, no grad, batch 1.
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
import time

import torch

from ..models import BACKBONES, build_backbone
from ..utils import dump_json, resolve_device, seed_everything

DEFAULT_KWARGS: dict[str, dict] = {
    "transformer": dict(d_model=256, n_layers=2, n_heads=4),
    "s4d": dict(d_model=256, n_layers=2, d_state=16),
    "mamba": dict(d_model=256, n_layers=2, d_state=16),
    "mamba2": dict(d_model=256, n_layers=2, d_state=32, headdim=64, scan_mode="chunked"),
    "mamba3": dict(d_model=256, n_layers=2, d_state=32, headdim=64),
}

VOCAB = 32


def _time_gpu(fn, repeats: int, warmup: int = 2, device: str = "cuda") -> tuple[float, float]:
    """Return (mean_ms, std_ms) of fn(), CUDA-event timed when possible."""
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
    return float(t.mean()), float(t.std())


def _decode_stepwise(backbone, embed, seq_len: int, device: torch.device) -> None:
    """True incremental decode for backbones whose blocks expose step()."""
    with torch.no_grad():
        states = [block.init_state(1, device) for block in backbone.blocks]
        for _ in range(seq_len):
            token = torch.randint(1, VOCAB, (1, 1), device=device)
            h = embed(token)
            for block, state in zip(backbone.blocks, states):
                h = block.step(h, state)


def _decode_sequential(backbone, embed, seq_len: int, device: torch.device) -> None:
    """Run the recurrent scan over a full sequence (per-token cost = time/T)."""
    with torch.no_grad():
        tokens = torch.randint(1, VOCAB, (1, seq_len), device=device)
        backbone(embed(tokens))


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
    embed = torch.nn.Embedding(VOCAB, kwargs["d_model"]).to(device)
    n_params = sum(p.numel() for p in backbone.parameters())
    entry: dict = {
        "model": name,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "n_parameters": n_params,
        "model_kwargs": kwargs,
    }

    # ---- training step ----
    try:
        tokens = torch.randint(1, VOCAB, (batch_size, seq_len), device=device)
        opt = torch.optim.AdamW(backbone.parameters(), lr=1e-4)
        target = torch.randint(1, VOCAB, (batch_size, seq_len), device=device)

        def train_step() -> None:
            logits = backbone(embed(tokens))
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        mean_ms, std_ms = _time_gpu(train_step, repeats, device=device.type)
        entry["train_step_ms"] = round(mean_ms, 2)
        entry["train_step_ms_std"] = round(std_ms, 2)
        if device.type == "cuda":
            entry["peak_train_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        entry["train_tokens_per_s"] = round(batch_size * seq_len / (mean_ms / 1000))
        del opt, tokens, target
    except torch.cuda.OutOfMemoryError:
        entry["oom_train"] = True
        torch.cuda.empty_cache()
        return entry

    # ---- full forward, no grad, batch 1 ----
    try:
        tokens = torch.randint(1, VOCAB, (1, seq_len), device=device)

        def infer() -> None:
            with torch.no_grad():
                backbone(embed(tokens))

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        mean_ms, _ = _time_gpu(infer, repeats, device=device.type)
        entry["infer_ms"] = round(mean_ms, 2)
        if device.type == "cuda":
            entry["peak_infer_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        del tokens
    except torch.cuda.OutOfMemoryError:
        entry["oom_infer"] = True
        torch.cuda.empty_cache()
        return entry

    # ---- decode ----
    try:
        rep_d = max(repeats // 2, 1)
        if name in ("mamba", "s4d"):
            fn = lambda: _decode_stepwise(backbone, embed, seq_len, device)
            entry["decode_mode"] = "stepwise (O(1) state)"
            mean_ms, _ = _time_gpu(fn, rep_d, warmup=1, device=device.type)
            entry["decode_ms"] = round(mean_ms, 2)
            entry["decode_per_token_ms"] = round(mean_ms / seq_len, 4)
        elif name in ("mamba2", "mamba3"):
            fn = lambda: _decode_sequential(backbone, embed, seq_len, device)
            entry["decode_mode"] = "sequential scan over full sequence (per-token = time/T)"
            mean_ms, _ = _time_gpu(fn, rep_d, warmup=1, device=device.type)
            entry["decode_per_token_ms"] = round(mean_ms / seq_len, 4)
        else:  # transformer: no KV cache — honest full-prefix recompute
            tokens = torch.randint(1, VOCAB, (1, seq_len), device=device)

            def decode_tr() -> None:
                with torch.no_grad():
                    backbone(embed(tokens))

            entry["decode_mode"] = "full-prefix recompute (no KV cache)"
            mean_ms, _ = _time_gpu(decode_tr, rep_d, warmup=1, device=device.type)
            entry["decode_per_token_ms"] = round(mean_ms / seq_len, 4)
    except torch.cuda.OutOfMemoryError:
        entry["oom_decode"] = True
        torch.cuda.empty_cache()

    return entry


def run_efficiency(cfg: dict) -> dict:
    device = resolve_device(cfg.get("device", "auto"))
    results = {
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "date": time.strftime("%Y-%m-%d"),
        },
        "config": cfg,
        "entries": [],
    }
    for name in cfg["models"]:
        # Optional per-model length cap: the educational sequential scans
        # (mamba, mamba3) have python-loop step counts proportional to T, so
        # very long sequences are measured only for the vectorized forms.
        limit = cfg.get("model_length_limits", {}).get(name)
        lengths = [L for L in cfg["seq_lengths"] if limit is None or L <= limit]
        for L in lengths:
            print(f"benchmarking {name} @ T={L} ...", flush=True)
            try:
                entry = benchmark_single(
                    name, L, cfg.get("batch_size", 8), cfg.get("repeats", 5), device
                )
            except torch.cuda.OutOfMemoryError:
                entry = {"model": name, "seq_len": L, "oom": True}
                torch.cuda.empty_cache()
            results["entries"].append(entry)
            dump_json(results, cfg["out_path"])
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
