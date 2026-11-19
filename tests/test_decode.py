"""Decode contract: prefill over a prompt, then one step per token, equals forward().

Every backbone exposes init_state / prefill / step (docs/design/step2.md section 3.1).
The checks run in float64, where the parallel and recurrent forms agree to rounding, so
the tolerance is rtol = atol = 1e-10.
"""

from __future__ import annotations

import pytest
import torch

from ssmbench.models import build_backbone
from ssmbench.models.blocks import Backbone, DecodeState
from ssmbench.models.mamba2_minimal import Mamba2Block
from ssmbench.models.mamba_minimal import MambaBlock

RTOL = ATOL = 1e-10

# case id -> (architecture, kwargs). Chunk size 4 makes T = 11 span three chunks with a
# partial last one; two groups exercise the B/C group expansion in the step path too.
DECODE_CASES: dict[str, tuple[str, dict]] = {
    "s4d": ("s4d", {"d_model": 8, "n_layers": 2, "d_state": 4}),
    "mamba": ("mamba", {"d_model": 8, "n_layers": 2, "d_state": 4}),
    "mamba2-chunked": (
        "mamba2",
        {"d_model": 8, "n_layers": 2, "d_state": 4, "headdim": 4, "n_groups": 2, "chunk_size": 4},
    ),
    "mamba2-sequential": (
        "mamba2",
        {"d_model": 8, "n_layers": 2, "d_state": 4, "headdim": 4, "scan_mode": "sequential"},
    ),
}


def _backbone(case: str, seed: int = 0) -> Backbone:
    torch.manual_seed(seed)
    name, kwargs = DECODE_CASES[case]
    backbone = build_backbone(name, kwargs).double().eval()
    assert isinstance(backbone, Backbone)
    # Non-trivial skip and gate weights so every term of the block shows up in the output.
    with torch.no_grad():
        for n, p in backbone.named_parameters():
            if n.endswith(".D"):
                p.uniform_(-1.0, 1.0)
    return backbone


def _decode(backbone: Backbone, x: torch.Tensor, k: int) -> torch.Tensor:
    """Prefill x[:, :k] (or start from an empty state when k = 0), then step the rest."""
    B, T, _ = x.shape
    outs = []
    if k == 0:
        state = backbone.init_state(B, x.device, max_len=T, dtype=x.dtype)
    else:
        y, state = backbone.prefill(x[:, :k], max_len=T)
        assert y.shape == (B, k, x.shape[-1])
        outs.append(y)
    assert isinstance(state, DecodeState)
    for t in range(k, T):
        y_t = backbone.step(x[:, t : t + 1], state)
        assert y_t.shape == (B, 1, x.shape[-1]) and y_t.dtype == x.dtype
        outs.append(y_t)
    return torch.cat(outs, dim=1)


def _width(case: str) -> int:
    return DECODE_CASES[case][1]["d_model"]


@pytest.mark.parametrize("case", list(DECODE_CASES))
@pytest.mark.parametrize("k", [0, 1, 4, 5, 10])
@pytest.mark.parametrize("batch", [1, 3])
def test_prefill_then_step_matches_forward(case: str, k: int, batch: int) -> None:
    """Fails if prefill returns the wrong final state or step drifts from forward()."""
    backbone = _backbone(case)
    torch.manual_seed(1)
    x = torch.randn(batch, 11, _width(case), dtype=torch.float64)
    with torch.no_grad():
        expected = backbone(x)
        got = _decode(backbone, x, k)
    torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", list(DECODE_CASES))
def test_state_position_counts_tokens(case: str) -> None:
    backbone = _backbone(case)
    x = torch.randn(2, 6, _width(case), dtype=torch.float64)
    with torch.no_grad():
        _, state = backbone.prefill(x[:, :4], max_len=6)
        assert state.position == 4
        backbone.step(x[:, 4:5], state)
    assert state.position == 5


@pytest.mark.parametrize("case", ["mamba", "mamba2-chunked"])
def test_step_recurrence_runs_in_float32_under_bf16_autocast(case: str) -> None:
    """The one-token update (state and readout) stays float32 under bf16 autocast, like the
    parallel scans; without autocast_disabled its einsum readout would come out bf16."""
    torch.manual_seed(0)
    name, kwargs = DECODE_CASES[case]
    backbone = build_backbone(name, kwargs).eval()
    assert isinstance(backbone, Backbone)
    x = torch.randn(2, 5, _width(case))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        _, state = backbone.prefill(x[:, :3])
        backbone.step(x[:, 3:4], state)
        block = backbone.blocks[0]
        h = state.blocks[0]["h"]
        if isinstance(block, MambaBlock):
            d_inner = block.d_inner
            y, h_new = block._recurrent_update(
                h,
                torch.randn(2, d_inner),
                torch.rand(2, d_inner),
                torch.randn(2, 4),
                torch.randn(2, 4),
            )
        else:
            assert isinstance(block, Mamba2Block)
            H, P, N = block.n_heads, block.headdim, block.d_state
            y, h_new = block._recurrent_update(
                h,
                torch.randn(2, H, P),
                -torch.rand(2, H),
                torch.randn(2, H, N),
                torch.randn(2, H, N),
            )
    assert h.dtype == torch.float32 and h_new.dtype == torch.float32
    assert y.dtype == torch.float32
