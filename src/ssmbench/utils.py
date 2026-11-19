"""Shared utilities: seeding, device selection, config loading, parameter counting."""

from __future__ import annotations

import gc
import hashlib
import json
import platform
import random
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
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


def reset_peak_memory(device: torch.device) -> None:
    """Start a fresh peak-memory window on a CUDA device; no-op elsewhere.

    Frees cached blocks first so the reserved peak reflects only what runs after
    this call, not memory left over from an earlier model in the same process.
    """
    if device.type != "cuda":
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def peak_memory_stats(device: torch.device) -> dict[str, float | bool] | None:
    """Peak memory since the last reset_peak_memory(device), in MiB; None off CUDA.

    Returns peak allocated and peak reserved memory, the device's total memory, and
    ``spilled``: whether the peak reserved memory exceeded the device total. On
    Windows (WDDM) the driver can back allocations with system memory instead of
    failing, so a measurement taken while spilled includes PCIe paging and is not
    a VRAM number.
    """
    if device.type != "cuda":
        return None
    mib = 1024**2
    reserved = torch.cuda.max_memory_reserved(device)
    total = torch.cuda.get_device_properties(device).total_memory
    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / mib, 1),
        "peak_reserved_mb": round(reserved / mib, 1),
        "device_total_mb": round(total / mib, 1),
        "spilled": reserved > total,
    }


PACKAGE_DIR = Path(__file__).resolve().parent


def source_sha256(package_dir: Path = PACKAGE_DIR) -> str:
    """SHA-256 of the package source: every .py file under package_dir in sorted
    relative-path order, with CRLF normalized to LF so Windows and Linux checkouts
    of the same code hash the same."""
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        digest.update(path.relative_to(package_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    """Output of a git command run in the package directory; None when git is not
    installed or the package is not inside a git checkout (provenance then records
    the field as unknown)."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=PACKAGE_DIR,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def run_provenance(device: torch.device) -> dict[str, Any]:
    """Where and with which code a measurement was taken; stored with every result.

    git_dirty is True when the package source differs from git_sha (result files
    written during a run do not count); source_sha256 identifies the exact code
    even without git.
    """
    status = _git("status", "--porcelain", "--untracked-files=no", "--", ".")
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "source_sha256": source_sha256(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def count_parameters(model: torch.nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config. Relative paths inside the config are
    resolved by the caller; this function only parses."""
    with open(path, encoding="utf-8") as f:
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
