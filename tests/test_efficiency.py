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


def test_single_repeat_reports_no_std() -> None:
    mean_ms, std_ms = efficiency._time_gpu(lambda: None, repeats=1, warmup=0, device="cpu")
    assert mean_ms >= 0 and std_ms is None
