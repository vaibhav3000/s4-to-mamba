"""torch.compile readiness of every architecture (ssmbench-compile).

For each model variant:

- ``forward``: torch._dynamo.explain of the training-mode forward pass: graph count,
  graph-break count and the break reasons. tests/test_compile.py pins zero breaks.
- ``step``: the same for one decode step after a prefill, eval mode, no grad.
- ``aot_eager``: torch.compile(backend="aot_eager") (Dynamo and AOTAutograd, no code
  generation) against eager: the largest absolute differences of the output and of the
  input gradient.
- ``inductor``: when Triton is importable on a CUDA device, torch.compile with the default
  inductor backend, its compile time (first call) reported separately, and the eager and
  compiled forward medians after warm-up; otherwise the reason it was not run (the
  Windows environment has no Triton).

Run with:

    ssmbench-compile --config configs/compile.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch._dynamo
from torch import nn

from ..models import build_backbone
from ..models.blocks import Backbone, DecodeState
from ..utils import dump_json, load_config, resolve_device, run_provenance, seed_everything
from .common import resolve_variant, time_calls


def graph_report(fn: Callable[..., Any], *args: Any) -> dict[str, Any]:
    """Graph count, graph-break count and break reasons of fn(*args) under Dynamo."""
    torch._dynamo.reset()
    explanation = torch._dynamo.explain(fn)(*args)
    return {
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "break_reasons": [str(r.reason).splitlines()[0] for r in explanation.break_reasons],
    }


def compiled_difference(
    module: nn.Module, x: torch.Tensor, backend: str = "aot_eager"
) -> dict[str, float]:
    """Largest absolute differences between eager and torch.compile(backend) for the
    output of module(x) and the gradient of output.sum() with respect to x."""
    torch._dynamo.reset()
    x_eager = x.detach().clone().requires_grad_(True)
    x_comp = x.detach().clone().requires_grad_(True)
    y_eager = module(x_eager)
    y_eager.sum().backward()
    compiled = torch.compile(module, backend=backend)
    y_comp = compiled(x_comp)
    y_comp.sum().backward()
    assert x_eager.grad is not None and x_comp.grad is not None
    return {
        "max_abs_output_diff": float((y_comp - y_eager).abs().max()),
        "max_abs_grad_diff": float((x_comp.grad - x_eager.grad).abs().max()),
    }


def _inductor(module: nn.Module, x: torch.Tensor, device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"status": "not run: inductor timing is measured on CUDA only"}
    if importlib.util.find_spec("triton") is None:
        return {"status": "not run: Triton is not installed in this environment"}
    torch._dynamo.reset()
    module.eval()
    compiled = torch.compile(module)
    with torch.no_grad():
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        compiled(x)
        torch.cuda.synchronize(device)
        compile_s = time.perf_counter() - t0
        eager = time_calls(lambda: module(x), repeats=30, warmup=10, device=device)
        comp = time_calls(lambda: compiled(x), repeats=30, warmup=10, device=device)
    return {
        "status": "measured",
        "first_call_seconds": round(compile_s, 2),
        "eager_forward_ms_median": round(statistics.median(eager), 4),
        "compiled_forward_ms_median": round(statistics.median(comp), 4),
        "repeats": 30,
        "warmup": 10,
    }


class _Step(nn.Module):
    """backbone.step as a module call, so Dynamo explains it like a forward."""

    def __init__(self, backbone: Backbone) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, emb_t: torch.Tensor, state: DecodeState) -> torch.Tensor:
        return self.backbone.step(emb_t, state)


def check_variant(
    name: str, kwargs: dict[str, Any], seq_len: int, device: torch.device
) -> dict[str, Any]:
    """forward, step, aot_eager and inductor results of one architecture (batch 2)."""
    seed_everything(0)
    backbone = build_backbone(name, kwargs).to(device)
    d_model = kwargs["d_model"]
    x = torch.randn(2, seq_len, d_model, device=device)
    backbone.train()
    forward = graph_report(backbone, x)
    aot = compiled_difference(backbone, x)
    backbone.eval()
    with torch.no_grad():
        _, state = backbone.prefill(x[:, :-1], max_len=seq_len)
    step = graph_report(_Step(backbone), x[:, -1:], state)
    return {
        "architecture": name,
        "model_kwargs": kwargs,
        "seq_len": seq_len,
        "forward": forward,
        "step": step,
        "aot_eager": aot,
        "inductor": _inductor(backbone, x, device),
    }


def run_compile(cfg: dict[str, Any]) -> dict[str, Any]:
    """Check every variant in cfg["models"] and write cfg["out_path"]."""
    device = resolve_device(cfg.get("device", "auto"))
    results: dict[str, Any] = {
        "environment": {**run_provenance(device), "date": time.strftime("%Y-%m-%d")},
        "config": cfg,
        "entries": [],
    }
    for label in cfg["models"]:
        name, kwargs = resolve_variant(label, cfg.get("variants"))
        print(f"compile check {label} ...", flush=True)
        entry = check_variant(name, kwargs, cfg.get("seq_len", 256), device)
        entry["model"] = label
        results["entries"].append(entry)
        dump_json(results, Path(cfg["out_path"]))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="ssmbench-compile",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run_compile(load_config(args.config))


if __name__ == "__main__":
    main()
