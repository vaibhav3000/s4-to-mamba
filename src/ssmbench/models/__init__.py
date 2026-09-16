"""Model registry: build any backbone by name from a config dict."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .mamba2_minimal import Mamba2Backbone
from .mamba3_minimal import Mamba3Backbone
from .mamba_minimal import MambaBackbone
from .s4_minimal import S4DBackbone
from .transformer_minimal import TransformerBackbone

BACKBONES: dict[str, type[nn.Module]] = {
    "transformer": TransformerBackbone,
    "s4d": S4DBackbone,
    "mamba": MambaBackbone,
    "mamba2": Mamba2Backbone,
    "mamba3": Mamba3Backbone,
}

# Constructor argument allow-list per backbone, so configs can share a schema
# without leaking another model's kwargs.
_KWARGS: dict[str, set[str]] = {
    "transformer": {"d_model", "n_layers", "n_heads", "d_mlp", "dropout", "max_len"},
    "s4d": {"d_model", "n_layers", "d_state", "dropout", "init"},
    "mamba": {"d_model", "n_layers", "d_state", "d_conv", "expand"},
    "mamba2": {
        "d_model", "n_layers", "d_state", "headdim", "expand", "n_groups",
        "chunk_size", "scan_mode",
    },
    "mamba3": {"d_model", "n_layers", "d_state", "headdim", "expand", "n_groups"},
}


def build_backbone(name: str, cfg: dict[str, Any]) -> nn.Module:
    """Instantiate backbone `name` with the allowed subset of `cfg`."""
    if name not in BACKBONES:
        raise KeyError(f"Unknown backbone '{name}'. Available: {sorted(BACKBONES)}")
    allowed = _KWARGS[name]
    kwargs = {k: v for k, v in cfg.items() if k in allowed}
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"Config keys {sorted(unknown)} are not used by backbone '{name}'")
    return BACKBONES[name](**kwargs)
