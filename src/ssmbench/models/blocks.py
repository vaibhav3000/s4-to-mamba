"""Shared building blocks used by every architecture.

All five backbones (Transformer, S4D, Mamba, Mamba-2, Mamba-3) share the same
outer skeleton so that comparisons isolate the mixer, not the plumbing:

    sequence tasks (SequenceClassifier, IMDb):
        token embedding -> [block x n_layers] -> final norm -> masked mean pool -> linear head
    token tasks (TokenLevelClassifier, parity and the copy tasks):
        token embedding -> [block x n_layers] -> final norm -> per-token linear head

The blocks themselves differ in size (the Transformer block has an MLP, the SSM
blocks do not), so equal width and depth do not mean equal parameter counts.
"""

from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ParamSpec, Protocol, TypeVar, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

P = ParamSpec("P")
R = TypeVar("R")


class SequenceClassifier(nn.Module):
    """Wraps a sequence backbone with an embedding, pooling head and classifier.

    Parameters
    ----------
    backbone:
        Module with signature (emb: (B, T, d_model) float embeddings,
        mask: (B, T) bool or None) -> (B, T, d_model) features.
    vocab_size:
        Number of input tokens.
    d_model:
        Backbone width (also the embedding width).
    n_classes:
        Output classes.
    padding_idx:
        Token id used for padding; excluded from pooling via the mask.
    """

    def __init__(
        self,
        backbone: nn.Module,
        vocab_size: int,
        d_model: int,
        n_classes: int = 2,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.backbone = backbone
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.padding_idx = padding_idx

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, T) -> logits: (B, n_classes)."""
        mask = tokens.ne(self.padding_idx) if self.padding_idx is not None else None
        h = self.backbone(self.embed(tokens), mask)
        h = self.norm(h)
        pooled = masked_mean(h, mask)
        return self.head(pooled)


class TokenLevelClassifier(nn.Module):
    """Wraps a backbone for per-token classification (parity, selective copy, exact copy).

    backbone: (embeddings (B, T, D), mask) -> features (B, T, D)
    logits: (B, T, n_classes)
    """

    def __init__(
        self,
        backbone: nn.Module,
        vocab_size: int,
        d_model: int,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.backbone = backbone
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.backbone(self.embed(tokens), None)
        return self.head(self.norm(h))


def masked_mean(h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool over the time dimension honouring a boolean validity mask.

    h: (B, T, D), mask: (B, T) bool. Without a mask this is a plain mean.
    """
    if mask is None:
        return h.mean(dim=1)
    mask = mask.unsqueeze(-1).to(h.dtype)
    return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def autocast_disabled(fn: Callable[P, R]) -> Callable[P, R]:
    """Run an SSM scan with autocast off on the devices of its tensor arguments.

    Under bf16 autocast the einsums inside a scan would otherwise run as bf16
    matmuls even though their inputs were cast to float32; with this decorator the
    whole scan runs in scan_dtype(...).

    The usual case, every tensor on one device type, enters torch.autocast with a
    plain `with`, which torch.compile traces without a graph break (autocast entered
    through an ExitStack is not traceable).
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        tensors = [a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)]
        device_types = sorted({t.device.type for t in tensors})
        if len(device_types) <= 1:
            device_type = device_types[0] if device_types else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                return fn(*args, **kwargs)
        with contextlib.ExitStack() as stack:
            for device_type in device_types:
                stack.enter_context(torch.autocast(device_type=device_type, enabled=False))
            return fn(*args, **kwargs)

    return wrapper


def scan_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Real dtype for an SSM scan: the promoted input dtype, never below float32.

    float32 and lower-precision inputs (bf16/fp16 from autocast) scan in float32,
    float64 inputs scan in float64, so float64 reference and gradcheck tests see
    float64 arithmetic end to end. The scans are wrapped in autocast_disabled, so
    autocast cannot move their einsums back to bf16.
    """
    dtype = torch.float32
    for t in tensors:
        dtype = torch.promote_types(dtype, t.dtype)
    return dtype


def complex_dtype_for(real_dtype: torch.dtype) -> torch.dtype:
    """complex128 for float64 scans, complex64 otherwise."""
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def expand_groups_to_heads(x: torch.Tensor, n_groups: int, n_heads: int) -> torch.Tensor:
    """Expand grouped SSM inputs (..., G*N) to per-head inputs (..., H, N).

    The H heads form G contiguous groups of H/G heads, and every head in group g
    reads the full N-vector of group g (Mamba-2's n_groups semantics; G = 1 means
    all heads share one B and one C).
    """
    if n_heads % n_groups != 0 or x.shape[-1] % n_groups != 0:
        raise ValueError(
            f"cannot split {x.shape[-1]} features into {n_groups} groups for {n_heads} heads"
        )
    grouped = x.unflatten(-1, (n_groups, x.shape[-1] // n_groups))  # (..., G, N)
    return grouped.repeat_interleave(n_heads // n_groups, dim=-2)  # (..., H, N)


def sinusoidal_positions(d_model: int, max_len: int, device: torch.device) -> torch.Tensor:
    """Classic sin/cos positional table, shape (max_len, d_model)."""
    pos = torch.arange(max_len, device=device).unsqueeze(1).float()
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
    return pe


# -------------------- decode interface (docs/design/step2.md, section 3.1) --------------------


@dataclass
class DecodeState:
    """Everything a backbone carries between decode steps.

    blocks: one state dict per block (KV cache for attention, recurrent state, short-conv
    history and boundary input for the SSMs). position: tokens consumed so far.
    max_len: the longest sequence the state was sized for (only the KV cache uses it).
    tables: per-sequence lookup tables a backbone builds once (the Transformer's RoPE
    cos/sin rows for every position up to max_len). Callers treat the state as opaque
    and pass it back to step().
    """

    blocks: list[dict[str, torch.Tensor]]
    position: int = 0
    max_len: int = 0
    tables: dict[str, torch.Tensor] = field(default_factory=dict)


class DecodeBlock(Protocol):
    """A residual block with a parallel prefill and an O(1)-state recurrent step."""

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype | None = None
    ) -> dict[str, torch.Tensor]: ...

    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...

    def step(self, x_t: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor: ...


class Backbone(nn.Module):
    """Base class of every backbone: forward(emb, mask) plus the decode interface.

    forward(emb (B, T, D), mask) -> (B, T, D) runs the parallel form.
    init_state(batch_size, device, max_len, dtype) -> an empty DecodeState.
    prefill(emb (B, T, D), max_len) -> (features (B, T, D), DecodeState after T tokens).
    step(emb_t (B, 1, D), state) -> features (B, 1, D); advances state by one token.

    prefill followed by step over the remaining tokens reproduces forward() position by
    position (tests/test_decode.py). Decode is inference only: prefill and step run
    without autograd. This default composes the blocks' own prefill and step; the
    Transformer overrides it to handle positions and its KV cache.
    """

    blocks: nn.ModuleList

    def _decode_blocks(self) -> list[DecodeBlock]:
        return [cast(DecodeBlock, block) for block in self.blocks]

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        max_len: int = 0,
        dtype: torch.dtype | None = None,
    ) -> DecodeState:
        blocks = [b.init_state(batch_size, device, dtype=dtype) for b in self._decode_blocks()]
        return DecodeState(blocks=blocks, position=0, max_len=max_len)

    @torch.no_grad()
    def prefill(self, emb: torch.Tensor, max_len: int = 0) -> tuple[torch.Tensor, DecodeState]:
        x = emb
        states = []
        for block in self._decode_blocks():
            x, block_state = block.prefill(x)
            states.append(block_state)
        T = emb.shape[1]
        return x, DecodeState(blocks=states, position=T, max_len=max(max_len, T))

    @torch.no_grad()
    def step(self, emb_t: torch.Tensor, state: DecodeState) -> torch.Tensor:
        if emb_t.ndim != 3 or emb_t.shape[1] != 1:
            raise ValueError(f"step() takes one token of shape (B, 1, D), got {tuple(emb_t.shape)}")
        x = emb_t
        for block, block_state in zip(self._decode_blocks(), state.blocks, strict=True):
            x = block.step(x, block_state)
        state.position += 1
        return x


def short_conv(conv: nn.Conv1d, x_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal depthwise convolution followed by SiLU, as in the Mamba blocks.

    x_in: (B, T, C). Returns (out (B, T, C), history (B, C, K - 1)), where history holds
    the last K - 1 inputs (zeros where the sequence is shorter), i.e. the decode state
    that short_conv_step() continues from. history is a copy in the conv weight's
    dtype: it owns K - 1 columns of storage instead of viewing the padded prompt, and
    a state made under autocast can be stepped without it.
    """
    T = x_in.shape[1]
    K = conv.kernel_size[0]
    padded = F.pad(x_in.transpose(1, 2), (K - 1, 0))  # (B, C, T + K - 1)
    out = F.silu(conv(padded)[..., :T]).transpose(1, 2)
    history = padded[..., T:].to(dtype=conv.weight.dtype, copy=True)
    return out, history


def short_conv_step(
    conv: nn.Conv1d, history: torch.Tensor, x_t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of short_conv. history: (B, C, K - 1), x_t: (B, C).

    Returns (out (B, C), new history (B, C, K - 1)).
    """
    window = torch.cat([history, x_t.unsqueeze(-1)], dim=-1)  # (B, C, K)
    out = F.silu(conv(window)[..., -1])
    return out, window[..., 1:]
