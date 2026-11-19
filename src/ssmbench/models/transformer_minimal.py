"""Educational causal Transformer baseline.

Purpose in this study: a *quality and scaling baseline* with softmax attention,
built from pre-norm blocks. Two settings define a Transformer variant:

- ``pos_encoding``: ``"sinusoidal"`` adds the classic sin/cos table to the token
  embeddings once; ``"rope"`` rotates queries and keys per head by their absolute
  position (rotary embedding, rotate-half convention, base 10000) and adds nothing to
  the embeddings. SSM layers need neither, because their recurrence encodes time.
- ``attn_backend``: ``"auto"`` lets PyTorch choose the SDPA kernel; ``"flash"``,
  ``"efficient"``, ``"cudnn"`` and ``"math"`` pin one kernel family (a pinned kernel
  that cannot run the inputs raises instead of falling back); ``"naive"`` is the
  written-out reference. FlashAttention and cuDNN attention need fp16 or bf16 on CUDA,
  so float32 runs use the memory-efficient or math kernels. See models/attention.py and
  docs/adr/0004.

Generation uses a KV cache: prefill() runs causal attention over the prompt and stores
every layer's keys and values in caches preallocated to max_len positions; step() writes
one position and attends one query to the cached prefix. Compute per decode step grows
linearly with the context, and the cache holds 2 * n_layers * max_len * d_model values
per sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import ATTENTION_BACKENDS, apply_rope, attention, rope_angles
from .blocks import Backbone, DecodeState, sinusoidal_positions

POS_ENCODINGS = ("sinusoidal", "rope")


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        pos_encoding: str = "sinusoidal",
        attn_backend: str = "auto",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        if pos_encoding not in POS_ENCODINGS:
            raise ValueError(f"pos_encoding must be one of {POS_ENCODINGS}, got {pos_encoding!r}")
        if attn_backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"attn_backend must be one of {ATTENTION_BACKENDS}, got {attn_backend!r}"
            )
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if pos_encoding == "rope" and self.d_head % 2:
            raise ValueError(f"RoPE needs an even head dimension, got {self.d_head}")
        self.rope = pos_encoding == "rope"
        self.attn_backend = attn_backend
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def _qkv(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, T, D), positions: (T,) -> q, k, v each (B, H, T, d_head)."""
        B, T, _ = x.shape
        q, k, v = (
            t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            for t in self.qkv(x).chunk(3, dim=-1)
        )
        if self.rope:
            cos, sin = rope_angles(
                positions, self.d_head, dtype=torch.promote_types(q.dtype, torch.float32)
            )
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        return q, k, v

    def _merge(self, y: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = y.shape
        return self.proj(y.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        positions = torch.arange(x.shape[1], device=x.device)
        q, k, v = self._qkv(x, positions)
        return self._merge(attention(q, k, v, causal=True, backend=self.attn_backend))

    def prefill(
        self, x: torch.Tensor, max_len: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """forward(x) plus K/V caches of max_len positions holding the prompt's keys and
        values (dtype of the keys, so bf16 under bf16 autocast)."""
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device)
        q, k, v = self._qkv(x, positions)
        y = self._merge(attention(q, k, v, causal=True, backend=self.attn_backend))
        shape = (B, self.n_heads, max_len, self.d_head)
        cache_k = torch.zeros(shape, dtype=k.dtype, device=x.device)
        cache_v = torch.zeros(shape, dtype=v.dtype, device=x.device)
        cache_k[:, :, :T] = k
        cache_v[:, :, :T] = v
        return y, {"k": cache_k, "v": cache_v}

    def step(
        self, x_t: torch.Tensor, state: dict[str, torch.Tensor], position: int
    ) -> torch.Tensor:
        """Attend the token at `position` (x_t: (B, 1, D)) to the cache; writes its K/V."""
        q, k, v = self._qkv(x_t, torch.tensor([position], device=x_t.device))
        state["k"][:, :, position : position + 1] = k
        state["v"][:, :, position : position + 1] = v
        keys = state["k"][:, :, : position + 1]
        values = state["v"][:, :, : position + 1]
        # One query attends to every cached position, so no mask is needed (a causal
        # mask on a 1 x (t + 1) score matrix would be upper-left aligned and wrong).
        y = attention(
            q, keys.to(q.dtype), values.to(q.dtype), causal=False, backend=self.attn_backend
        )
        return self._merge(y)


class TransformerBlock(nn.Module):
    """Pre-norm block: x + attn(LN(x)); x + mlp(LN(x))."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        dropout: float = 0.0,
        pos_encoding: str = "sinusoidal",
        attn_backend: str = "auto",
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, pos_encoding, attn_backend)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_mlp, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

    def prefill(
        self, x: torch.Tensor, max_len: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y, state = self.attn.prefill(self.ln1(x), max_len)
        x = x + self.dropout(y)
        return x + self.dropout(self.mlp(self.ln2(x))), state

    def step(
        self, x_t: torch.Tensor, state: dict[str, torch.Tensor], position: int
    ) -> torch.Tensor:
        x_t = x_t + self.dropout(self.attn.step(self.ln1(x_t), state, position))
        return x_t + self.dropout(self.mlp(self.ln2(x_t)))


class TransformerBackbone(Backbone):
    pe: torch.Tensor | None  # (max_len, d_model) sinusoidal table, or None with RoPE

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_mlp: int | None = None,
        dropout: float = 0.0,
        max_len: int = 65536,
        pos_encoding: str = "sinusoidal",
        attn_backend: str = "auto",
    ) -> None:
        super().__init__()
        d_mlp = d_mlp or 4 * d_model
        self.d_model = d_model
        self.max_len = max_len
        self.pos_encoding = pos_encoding
        self.attn_backend = attn_backend
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_mlp, dropout, pos_encoding, attn_backend)
                for _ in range(n_layers)
            ]
        )
        if pos_encoding == "sinusoidal":
            # Positions are added once per forward; a non-persistent buffer.
            self.register_buffer(
                "pe", sinusoidal_positions(d_model, max_len, torch.device("cpu")), persistent=False
            )
        else:
            self.pe = None

    def _positioned(self, emb: torch.Tensor, start: int) -> torch.Tensor:
        end = start + emb.shape[1]
        if self.max_len < end:
            raise ValueError(f"Sequence length {end} exceeds max_len {self.max_len}")
        if self.pe is None:
            return emb
        return emb + self.pe[start:end].to(dtype=emb.dtype)

    def forward(self, emb: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self._positioned(emb, 0)
        for block in self.blocks:
            x = block(x, mask)
        return x

    def _transformer_blocks(self) -> list[TransformerBlock]:
        return [b for b in self.blocks if isinstance(b, TransformerBlock)]

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        max_len: int = 0,
        dtype: torch.dtype | None = None,
    ) -> DecodeState:
        """Empty KV caches of max_len positions (max_len must be positive)."""
        if max_len <= 0:
            raise ValueError("a Transformer decode state needs max_len > 0 for its KV cache")
        layers = self._transformer_blocks()
        cache_dtype = dtype or layers[0].ln1.weight.dtype
        blocks = []
        for block in layers:
            shape = (batch_size, block.attn.n_heads, max_len, block.attn.d_head)
            blocks.append(
                {
                    "k": torch.zeros(shape, dtype=cache_dtype, device=device),
                    "v": torch.zeros(shape, dtype=cache_dtype, device=device),
                }
            )
        return DecodeState(blocks=blocks, position=0, max_len=max_len)

    def prefill(self, emb: torch.Tensor, max_len: int = 0) -> tuple[torch.Tensor, DecodeState]:
        T = emb.shape[1]
        max_len = max(max_len, T)
        x = self._positioned(emb, 0)
        states = []
        for block in self._transformer_blocks():
            x, block_state = block.prefill(x, max_len)
            states.append(block_state)
        return x, DecodeState(blocks=states, position=T, max_len=max_len)

    def step(self, emb_t: torch.Tensor, state: DecodeState) -> torch.Tensor:
        if emb_t.ndim != 3 or emb_t.shape[1] != 1:
            raise ValueError(f"step() takes one token of shape (B, 1, D), got {tuple(emb_t.shape)}")
        position = state.position
        if position >= state.max_len:
            raise ValueError(
                f"the KV cache holds max_len={state.max_len} positions and is full; "
                "prefill or init_state with a larger max_len"
            )
        x = self._positioned(emb_t, position)
        for block, block_state in zip(self._transformer_blocks(), state.blocks, strict=True):
            x = block.step(x, block_state, position)
        state.position += 1
        return x
