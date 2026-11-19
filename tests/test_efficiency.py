"""CPU tests for the efficiency benchmark's bookkeeping (no timing claims)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from ssmbench.benchmarks import efficiency

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
    assert "train_step_ms" in entry and "infer_ms" in entry and "decode_per_token_ms" in entry
    # _time_gpu: 2 warmup + 2 timed train steps, then inference and decode calls.
    assert modes[:4] == [True] * 4
    assert len(modes) > 4 and not any(modes[4:])
