"""Attention backends, RoPE and the Transformer variants (docs/adr/0004).

float64 comparisons use rtol = atol = 1e-10.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ssmbench.models import build_backbone
from ssmbench.models.attention import (
    ATTENTION_BACKENDS,
    apply_rope,
    attention,
    detect_attention_backend,
    naive_attention,
    rope_angles,
)
from ssmbench.models.transformer_minimal import TransformerBackbone

RTOL = ATOL = 1e-10


def _qkv(T: int = 7, S: int | None = None, seed: int = 0) -> tuple[torch.Tensor, ...]:
    g = torch.Generator().manual_seed(seed)
    S = T if S is None else S
    q = torch.randn(2, 3, T, 4, generator=g, dtype=torch.float64)
    k = torch.randn(2, 3, S, 4, generator=g, dtype=torch.float64)
    v = torch.randn(2, 3, S, 4, generator=g, dtype=torch.float64)
    return q, k, v


@pytest.mark.parametrize("causal", [True, False])
def test_naive_attention_matches_sdpa_math(causal: bool) -> None:
    q, k, v = _qkv()
    expected = attention(q, k, v, causal=causal, backend="math")
    torch.testing.assert_close(
        naive_attention(q, k, v, causal=causal), expected, rtol=RTOL, atol=ATOL
    )


def test_single_query_against_a_cache_needs_no_mask() -> None:
    """A decode step attends one query to every cached key: equal to the last row of
    causal attention over the full sequence."""
    q, k, v = _qkv(T=6)
    full = naive_attention(q, k, v, causal=True)
    last = naive_attention(q[:, :, -1:], k, v, causal=False)
    torch.testing.assert_close(last, full[:, :, -1:], rtol=RTOL, atol=ATOL)


def test_pinned_math_backend_is_detected() -> None:
    q, k, v = _qkv()
    used = detect_attention_backend(lambda: attention(q, k, v, causal=True, backend="math"))
    assert used == ["math"]


def test_naive_backend_dispatches_no_sdpa_kernel() -> None:
    q, k, v = _qkv()
    assert detect_attention_backend(lambda: naive_attention(q, k, v, causal=True)) == []


def test_pinned_backend_raises_instead_of_falling_back() -> None:
    """The memory-efficient kernel has no CPU implementation: pinning it on CPU must
    raise, not silently run another kernel."""
    q, k, v = (t.float() for t in _qkv())
    with pytest.raises(RuntimeError):
        attention(q, k, v, causal=True, backend="efficient")


def test_unknown_backend_is_rejected() -> None:
    q, k, v = _qkv()
    with pytest.raises(ValueError, match="backend"):
        attention(q, k, v, causal=True, backend="fast")
    assert set(ATTENTION_BACKENDS) == {"auto", "flash", "efficient", "cudnn", "math", "naive"}


def test_rope_scores_depend_only_on_the_offset() -> None:
    """<R(m) q, R(n) k> = <R(m + s) q, R(n + s) k> for every shift s, and R(m) preserves
    the norm. float64, 1e-10."""
    g = torch.Generator().manual_seed(1)
    q = torch.randn(1, 1, 1, 8, generator=g, dtype=torch.float64)
    k = torch.randn(1, 1, 1, 8, generator=g, dtype=torch.float64)

    def rotated(x: torch.Tensor, pos: int) -> torch.Tensor:
        cos, sin = rope_angles(torch.tensor([pos]), 8, dtype=torch.float64)
        return apply_rope(x, cos, sin)

    base = (rotated(q, 5) * rotated(k, 2)).sum()
    for shift in (1, 17, 1000):
        shifted = (rotated(q, 5 + shift) * rotated(k, 2 + shift)).sum()
        torch.testing.assert_close(shifted, base, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(rotated(q, 123).norm(), q.norm(), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(rotated(q, 0), q, rtol=0.0, atol=0.0)


def _transformer(**kwargs: object) -> TransformerBackbone:
    torch.manual_seed(0)
    backbone = build_backbone(
        "transformer", {"d_model": 8, "n_layers": 2, "n_heads": 2, **kwargs}
    ).double()
    assert isinstance(backbone, TransformerBackbone)
    return backbone.eval()


@pytest.mark.parametrize("pos_encoding", ["sinusoidal", "rope"])
def test_backends_agree_through_the_backbone(pos_encoding: str) -> None:
    x = torch.randn(2, 9, 8, dtype=torch.float64)
    with torch.no_grad():
        outs = {
            b: _transformer(pos_encoding=pos_encoding, attn_backend=b)(x)
            for b in ("auto", "math", "naive")
        }
    torch.testing.assert_close(outs["naive"], outs["math"], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(outs["auto"], outs["math"], rtol=RTOL, atol=ATOL)


def test_rope_variant_has_no_position_table_and_the_same_parameters() -> None:
    rope = _transformer(pos_encoding="rope")
    sinusoidal = _transformer()
    assert sum(p.numel() for p in rope.parameters()) == sum(
        p.numel() for p in sinusoidal.parameters()
    )
    assert getattr(rope, "pe", None) is None
    x = torch.randn(1, 5, 8, dtype=torch.float64)
    with torch.no_grad():
        assert not torch.allclose(rope(x), sinusoidal(x))


def test_rope_odd_head_dim_is_rejected() -> None:
    with pytest.raises(ValueError, match="even"):
        build_backbone(
            "transformer", {"d_model": 6, "n_layers": 1, "n_heads": 2, "pos_encoding": "rope"}
        )


def test_decode_past_max_len_raises() -> None:
    backbone = _transformer()
    x = torch.randn(1, 4, 8, dtype=torch.float64)
    with torch.no_grad():
        _, state = backbone.prefill(x, max_len=5)
        backbone.step(x[:, :1], state)
        with pytest.raises(ValueError, match="max_len"):
            backbone.step(x[:, :1], state)


def test_kv_cache_follows_the_key_dtype_under_bf16_autocast() -> None:
    torch.manual_seed(0)
    backbone = build_backbone("transformer", {"d_model": 8, "n_layers": 1, "n_heads": 2}).eval()
    assert isinstance(backbone, TransformerBackbone)
    x = torch.randn(1, 4, 8)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        _, state = backbone.prefill(x, max_len=6)
        backbone.step(x[:, :1], state)
    assert state.blocks[0]["k"].dtype == torch.bfloat16
    assert state.blocks[0]["k"].shape == (1, 2, 6, 4)


def test_sdpa_math_reference_is_float64_capable() -> None:
    """The math backend is the float64 oracle used above; guard that it really computed
    in float64 (a float32 path would miss 1e-10 by orders of magnitude)."""
    q, k, v = _qkv()
    out = attention(q, k, v, causal=True, backend="math")
    ref = (
        F.softmax(
            (q @ k.transpose(-2, -1)) / 2.0
            + torch.full((7, 7), float("-inf"), dtype=torch.float64).triu(1),
            dim=-1,
        )
        @ v
    )
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
