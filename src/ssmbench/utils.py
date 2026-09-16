"""Shared utilities: seeding, device selection, config loading, parameter counting."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and all CUDA devices) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    """Map a config string to a torch.device. 'auto' picks CUDA when available."""
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config. Relative paths inside the config are
    resolved by the caller; this function only parses."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must parse to a mapping, got {type(cfg)}")
    return cfg


def config_to_jsonable(cfg: dict[str, Any]) -> dict[str, Any]:
    """Convert a config (possibly containing dataclasses) to a JSON-serialisable dict."""
    out: dict[str, Any] = {}
    for k, v in cfg.items():
        if is_dataclass(v) and not isinstance(v, type):
            out[k] = asdict(v)  # type: ignore[arg-type]
        elif isinstance(v, dict):
            out[k] = config_to_jsonable(v)
        else:
            out[k] = v
    return out


def dump_json(obj: Any, path: str | Path) -> None:
    """Write JSON with stable formatting, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
