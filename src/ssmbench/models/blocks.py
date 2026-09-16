"""Shared building blocks used by every architecture.

All five backbones (Transformer, S4D, Mamba, Mamba-2, Mamba-3) share the same
outer skeleton so that comparisons isolate the mixer, not the plumbing:

    token embedding -> [block x n_layers] -> final norm -> masked mean pool -> linear head
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SequenceClassifier(nn.Module):
    """Wraps a sequence backbone with an embedding, pooling head and classifier.

    Parameters
    ----------
    backbone:
        Module with signature (x: (B, T) int64 tokens, mask: (B, T) bool)
        -> (B, T, d_model) features.
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
    """Wraps a backbone for per-token classification (used by the parity task).

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


def causal_mask(T: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular boolean mask of shape (T, T); True marks disallowed attention."""
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
