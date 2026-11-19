"""Educational causal Transformer baseline.

Purpose in this study: a *quality and scaling baseline* with softmax attention,
built from pre-norm blocks and `F.scaled_dot_product_attention` with
is_causal=True. On CUDA in float32, the trainer's default (amp off) and the only
mode of the efficiency benchmark, PyTorch can dispatch SDPA to its
memory-efficient kernel but not to FlashAttention, which needs fp16 or bf16.
There is no KV cache. Wall-clock comparisons against the SSMs are therefore
against this fp32 baseline, not against a best-case attention implementation.

What it exposes for the study:
- O(T^2) compute in the attention score matrix: visible as latency growth with T
  even though the memory-efficient kernel keeps activation memory closer to linear.
- The need for explicit positional information: an absolute sinusoidal table is
  added once to the token embeddings (no RoPE, ALiBi or NoPE variant). LTI/SSM
  layers need no such table because their recurrence already encodes time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import sinusoidal_positions


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        # Memory-efficient fused kernel; is_causal handles the triangular mask.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(y)


class TransformerBlock(nn.Module):
    """Pre-norm block: x + attn(LN(x)); x + mlp(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_mlp: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_mlp, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class TransformerBackbone(nn.Module):
    pe: torch.Tensor  # (max_len, d_model) sinusoidal table, a non-persistent buffer

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_mlp: int | None = None,
        dropout: float = 0.0,
        max_len: int = 65536,
    ) -> None:
        super().__init__()
        d_mlp = d_mlp or 4 * d_model
        self.d_model = d_model
        self.max_len = max_len
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_mlp, dropout) for _ in range(n_layers)]
        )
        # Positions are added once per forward; SSM baselines need no equivalent.
        self.register_buffer(
            "pe", sinusoidal_positions(d_model, max_len, torch.device("cpu")), persistent=False
        )

    def forward(self, emb: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        T = emb.shape[1]
        if self.max_len < T:
            raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len}")
        x = emb + self.pe[:T].to(dtype=emb.dtype)
        for block in self.blocks:
            x = block(x, mask)
        return x
