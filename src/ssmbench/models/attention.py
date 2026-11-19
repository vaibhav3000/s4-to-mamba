"""Attention kernels, rotary position embedding and attention-backend detection.

Backends (docs/adr/0004):

- ``auto``: F.scaled_dot_product_attention with PyTorch's own kernel choice.
- ``flash``, ``efficient``, ``cudnn``, ``math``: SDPA pinned to one kernel family with
  sdpa_kernel([backend]). A pinned backend that cannot run the inputs raises; it never
  falls back to another kernel.
- ``naive``: softmax(Q K^T / sqrt(d) + causal mask) V written out in the input dtype.
  It is the correctness reference and is labelled as such wherever it is measured.

detect_attention_backend() reports which SDPA kernel families actually executed, read
from the aten op names in a profiler trace.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

_PINNED: dict[str, SDPBackend] = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "math": SDPBackend.MATH,
}
ATTENTION_BACKENDS = ("auto", *_PINNED, "naive")

# aten op of each SDPA kernel family -> label recorded with results.
_SDPA_OPS = {
    "aten::_scaled_dot_product_flash_attention": "flash",
    "aten::_scaled_dot_product_flash_attention_for_cpu": "flash_cpu",
    "aten::_scaled_dot_product_efficient_attention": "efficient",
    "aten::_scaled_dot_product_cudnn_attention": "cudnn",
    "aten::_scaled_dot_product_attention_math": "math",
}


def naive_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool
) -> torch.Tensor:
    """Reference attention. q: (B, H, L, d), k and v: (B, H, S, d) -> (B, H, L, d).

    causal=True requires L == S (position i attends to j <= i).
    """
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])  # (B, H, L, S)
    if causal:
        L, S = q.shape[-2], k.shape[-2]
        if L != S:
            raise ValueError(f"causal naive attention needs equal lengths, got {L} and {S}")
        future = torch.ones(L, S, dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool, backend: str
) -> torch.Tensor:
    """Scaled dot-product attention with the named backend (see the module docstring)."""
    if backend == "naive":
        return naive_attention(q, k, v, causal=causal)
    if backend == "auto":
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    if backend not in _PINNED:
        raise ValueError(f"unknown attention backend {backend!r}; use one of {ATTENTION_BACKENDS}")
    with sdpa_kernel([_PINNED[backend]]):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def detect_attention_backend(fn: Callable[[], object]) -> list[str]:
    """Sorted labels of the SDPA kernel families that ran inside fn() (empty if none).

    Reads aten op names from a CPU-side profiler trace, which records the dispatched op
    for CUDA tensors too.
    """
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
    return sorted({_SDPA_OPS[e.name] for e in (prof.events() or []) if e.name in _SDPA_OPS})


def rope_angles(
    positions: torch.Tensor, d_head: int, base: float = 10000.0, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin tables of rotary embedding for integer positions (T,): each (T, d_head).

    Frequency i is base^(-2i / d_head), repeated for the two halves (rotate-half
    convention). Angles are formed in float64 so position 32767 keeps its precision,
    then cast to dtype (default float32).
    """
    if d_head % 2:
        raise ValueError(f"RoPE needs an even head dimension, got {d_head}")
    inv_freq = base ** (
        -torch.arange(0, d_head, 2, dtype=torch.float64, device=positions.device) / d_head
    )
    angles = positions.to(torch.float64).unsqueeze(-1) * inv_freq  # (T, d_head / 2)
    angles = torch.cat([angles, angles], dim=-1)
    out_dtype = dtype or torch.float32
    return angles.cos().to(out_dtype), angles.sin().to(out_dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x (..., T, d_head) by the angles in cos/sin (T, d_head); returns x's dtype.

    The rotation is computed in the dtype of cos (float32 for bf16 inputs).
    """
    xf = x.to(cos.dtype)
    x1, x2 = xf.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return (xf * cos + rotated * sin).to(x.dtype)
