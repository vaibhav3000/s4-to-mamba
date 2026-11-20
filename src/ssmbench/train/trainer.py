"""Config-driven training loop with deterministic seeding and JSON results.

Supports two task types:
- "classification": sequence-level labels (IMDb). Metrics: accuracy, AUC.
- "token_classification": per-token targets with -100 ignore (copy, parity).
  Metrics: token accuracy and, for selective copy, full-sequence success rate.

Every run (one model variant, one run seed) writes <out_dir>/seed<k>.json with
the resolved config, the per-epoch validation history, final metrics on every test
evaluation set, the parameter count, the attention backend that ran (Transformers),
memory and provenance. These files are the single source of truth for the result
tables. See docs/design/step2.md section 4 and docs/adr/0001, 0008.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..data.imdb import get_imdb_loaders
from ..data.loaders import TaskLoaders
from ..data.synthetic import (
    get_exact_copy_loaders,
    get_parity_loaders,
    get_selective_copy_loaders,
)
from ..models import build_backbone
from ..models.attention import detect_attention_backend
from ..models.blocks import SequenceClassifier, TokenLevelClassifier
from ..models.transformer_minimal import TransformerBackbone
from ..utils import (
    config_to_jsonable,
    count_parameters,
    dump_json,
    peak_memory_stats,
    reset_peak_memory,
    resolve_device,
    run_provenance,
    seed_everything,
    source_sha256,
)


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    scheduler: str = "cosine"  # "cosine" | "none"
    warmup_frac: float = 0.05
    # bfloat16 autocast on CUDA. The SSM projections that produce delta, B, C (and
    # Mamba-3's lambda) and the scans themselves still run in float32; every other
    # layer follows autocast. No committed config enables it.
    amp: bool = False
    seed: int = 42
    optimizer: str = "adamw"
    # torch.use_deterministic_algorithms for the run (docs/adr/0008).
    deterministic: bool = False


def _env_info(device: torch.device) -> dict:
    info = {
        "torch": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


# Parameters that set the SSM dynamics get no weight decay: the state matrix A
# (decay rates and, for Mamba-3, rotation frequencies), the S4D step size and the
# skip D. Decaying them would pull A toward -1 and the rotations toward 0. This
# follows the reference implementations, which exempt the same kind of parameters
# (S4D registers log_dt, A_real and A_imag with weight decay 0; Mamba marks A_log
# and D as no-weight-decay) and treat B and C as ordinary weights, so B and C are
# decayed here too. Every 1-D parameter, including each D vector, the S4D step size
# and Mamba-2's per-head A_log, is already exempt by the ndim rule below.
SSM_NO_DECAY_PARAMS = frozenset(
    {
        "log_A_real",  # S4D
        "A_imag",  # S4D
        "dt_param",  # S4D step size (1-D)
        "A_log",  # Mamba-1 (d_inner, N) and Mamba-2 (H,)
        "a_real_log",  # Mamba-3
        "theta",  # Mamba-3 rotation frequencies
        "D",  # Mamba-1/2/3 and S4D skip
    }
)


def build_param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """AdamW parameter groups: weight decay only on weights with ndim >= 2 that are
    not SSM dynamics parameters (see SSM_NO_DECAY_PARAMS); biases, norms and every
    other 1-D parameter get none. Parameters are matched by their attribute name."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        (no_decay if p.ndim <= 1 or leaf in SSM_NO_DECAY_PARAMS else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_model(cfg: dict) -> nn.Module:
    """Assemble SequenceClassifier/TokenLevelClassifier from the config."""
    mcfg = cfg["model"]
    backbone = build_backbone(mcfg["name"], mcfg.get("kwargs", {}))
    d_model = mcfg["kwargs"]["d_model"]
    task = cfg["task"]["type"]
    if task == "classification":
        return SequenceClassifier(
            backbone,
            vocab_size=cfg["task"]["vocab_size"],
            d_model=d_model,
            n_classes=cfg["task"].get("n_classes", 2),
            padding_idx=cfg["task"].get("padding_idx", 0),
        )
    if task == "token_classification":
        return TokenLevelClassifier(
            backbone,
            vocab_size=cfg["task"]["vocab_size"],
            d_model=d_model,
            n_classes=cfg["task"].get("n_classes", 2),
        )
    raise ValueError(f"Unknown task type {task!r}")


def get_loaders(cfg: dict) -> TaskLoaders:
    """Training, validation and named test loaders of the task in cfg["data"].

    Every dataset is fixed by the data seed cfg["data"]["seed"]; the run seed in
    cfg["train"] never changes which examples a run sees (docs/adr/0001).
    """
    dcfg = cfg["data"]
    name = dcfg["name"]
    if name == "imdb":
        loaders, _vocab = get_imdb_loaders(
            max_features=dcfg.get("max_features", 20000),
            max_len=dcfg.get("max_len", 1024),
            batch_size=dcfg.get("batch_size", 32),
            train_subset=dcfg.get("train_subset"),
            val_size=dcfg.get("val_size", 2500),
            test_subset=dcfg.get("test_subset"),
            seed=dcfg.get("seed", 42),
        )
        return loaders
    if name == "selective_copy":
        return get_selective_copy_loaders(dcfg)
    if name == "parity":
        return get_parity_loaders(dcfg)
    if name == "exact_copy":
        return get_exact_copy_loaders(dcfg)
    raise ValueError(f"Unknown dataset {name!r}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, task: str, device: torch.device) -> dict:
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_labels, all_preds, all_probs = [], [], []
    seq_correct, seq_total = 0, 0
    for batch in loader:
        tokens = batch["input_ids"].to(device)
        if task == "classification":
            labels = batch["label"].to(device)
            logits = model(tokens)
            loss = F.cross_entropy(logits, labels)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(logits.argmax(-1).cpu().tolist())
        else:
            targets = batch["targets"].to(device)
            logits = model(tokens)  # (B, T, C)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
            )
            mask = targets.ne(-100)
            preds = logits.argmax(-1)
            correct = (preds == targets) & mask
            all_preds.extend(preds[mask].cpu().tolist())
            all_labels.extend(targets[mask].cpu().tolist())
            # Sequence success: every scored position correct (selective copy).
            per_seq = correct.sum(dim=1) == mask.sum(dim=1)
            seq_correct += int(per_seq.sum())
            seq_total += int(per_seq.numel())
        total_loss += float(loss)
        n_batches += 1

    metrics: dict = {"loss": total_loss / max(n_batches, 1)}
    if task == "classification":
        metrics["accuracy"] = float(np.mean(np.array(all_preds) == np.array(all_labels)))
        try:
            metrics["auc"] = float(roc_auc_score(all_labels, all_probs))
        except ValueError:
            metrics["auc"] = None  # single-class batch (degenerate subsets)
    else:
        metrics["token_accuracy"] = float(np.mean(np.array(all_preds) == np.array(all_labels)))
        metrics["sequence_success"] = seq_correct / max(seq_total, 1)
    return metrics


# Run-config keys that say where a run executes or writes, not what it computes.
RUN_LOCATION_KEYS = frozenset({"device", "checkpoint_dir"})


def run_config_sha256(cfg: dict) -> str:
    """SHA-256 of everything in a run config that determines its result."""
    measured = {k: v for k, v in cfg.items() if k not in RUN_LOCATION_KEYS}
    payload = json.dumps(config_to_jsonable(measured), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


@contextlib.contextmanager
def deterministic_algorithms(enabled: bool) -> Iterator[None]:
    """Turn on torch's deterministic algorithms for the block, then restore the previous
    setting. CUBLAS_WORKSPACE_CONFIG must reach cuBLAS before its first handle is made;
    ssmbench-train sets it at start-up, and this sets it too for library callers."""
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def _rng_state(device: torch.device) -> dict[str, Any]:
    """Every RNG a training epoch draws from, in a torch.load(weights_only=True) form."""
    _, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "numpy": {
            "keys": torch.from_numpy(np.asarray(keys).astype(np.int64)),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached": float(cached),
        },
        "python": random.getstate(),
    }


def _set_rng_state(state: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    n = state["numpy"]
    keys = n["keys"].numpy().astype(np.uint32)
    np.random.set_state(("MT19937", keys, n["pos"], n["has_gauss"], n["cached"]))
    random.setstate(state["python"])


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """torch.save into a temporary file in the same directory, then os.replace, so an
    interrupted save never leaves a truncated checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _detect_backend(model: nn.Module, loader: DataLoader, device: torch.device) -> list[str] | None:
    """SDPA kernel families one training-mode forward runs, for Transformer backbones
    (None for SSMs). Uses the first validation batch (a sequential loader, so no RNG is
    drawn) and runs no backward."""
    backbone = getattr(model, "backbone", None)
    if not isinstance(backbone, TransformerBackbone):
        return None
    tokens = next(iter(loader))["input_ids"].to(device)
    model.train()
    return detect_attention_backend(lambda: model(tokens))


def run_training(cfg: dict, out_dir: str | Path) -> dict:
    """Train one model variant with one run seed; write <out_dir>/seed<k>.json.

    cfg: task, data, train (TrainConfig fields; train.seed is the run seed), model
    {name, kwargs}, device, and optionally name, config (the experiment config's name),
    label (the model variant) and checkpoint_dir. With checkpoint_dir, every epoch ends
    with a checkpoint (model, optimizer, scheduler, history and every RNG state); a run
    that finds a checkpoint written with the same run config and source resumes from it
    and ends bitwise identical to an uninterrupted run. The checkpoint is deleted once
    the result file is written.
    """
    out_dir = Path(out_dir)
    tcfg = TrainConfig(**cfg["train"])
    cfg_hash = run_config_sha256(cfg)
    with deterministic_algorithms(tcfg.deterministic):
        results = _train(cfg, tcfg, cfg_hash)
    dump_json(results, out_dir / f"seed{tcfg.seed}.json")
    if cfg.get("checkpoint_dir"):
        (Path(cfg["checkpoint_dir"]) / "last.pt").unlink(missing_ok=True)
    return results


def _train(cfg: dict, tcfg: TrainConfig, cfg_hash: str) -> dict:
    seed_everything(tcfg.seed)
    device = resolve_device(cfg.get("device", "auto"))
    # One process may train several runs in a row (ssmbench-train loops over variants
    # and seeds), so the peak-memory window starts here.
    available_at_start = reset_peak_memory(device)

    model = build_model(cfg).to(device)
    loaders = get_loaders(cfg)
    task = cfg["task"]["type"]
    attention_backend = _detect_backend(model, loaders.val, device)

    opt = torch.optim.AdamW(build_param_groups(model, tcfg.weight_decay), lr=tcfg.lr)
    steps_per_epoch = len(loaders.train)
    total_steps = steps_per_epoch * tcfg.epochs
    warmup = max(1, int(tcfg.warmup_frac * total_steps))

    def lr_lambda(step: int) -> float:
        if tcfg.scheduler == "none":
            return 1.0
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history: list[dict] = []
    start_epoch, resumed_from, elapsed_before = 0, None, 0.0
    ckpt_path = Path(cfg["checkpoint_dir"]) / "last.pt" if cfg.get("checkpoint_dir") else None
    source = source_sha256()
    if ckpt_path is not None and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if ckpt.get("config_sha256") == cfg_hash and ckpt.get("source_sha256") == source:
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            sched.load_state_dict(ckpt["scheduler"])
            history = list(ckpt["history"])
            start_epoch = resumed_from = int(ckpt["epoch"])
            elapsed_before = float(ckpt["elapsed_seconds"])
            _set_rng_state(ckpt["rng"], device)
            print(f"[{cfg.get('name')}] resuming after epoch {start_epoch}", flush=True)
        else:
            print(
                f"[{cfg.get('name')}] ignoring {ckpt_path}: written with other settings or code",
                flush=True,
            )

    t0 = time.time()
    for epoch in range(start_epoch, tcfg.epochs):
        model.train()
        ep_loss, ep_batches = 0.0, 0
        ep_start = time.time()
        for batch in loaders.train:
            tokens = batch["input_ids"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=tcfg.amp and device.type == "cuda",
            ):
                if task == "classification":
                    logits = model(tokens)
                    loss = F.cross_entropy(logits, batch["label"].to(device))
                else:
                    targets = batch["targets"].to(device)
                    logits = model(tokens)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
                    )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            sched.step()
            ep_loss += float(loss.detach())
            ep_batches += 1
        val_metrics = evaluate(model, loaders.val, task, device)
        entry = {
            "epoch": epoch + 1,
            "train_loss": ep_loss / max(ep_batches, 1),
            "epoch_seconds": round(time.time() - ep_start, 2),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(entry)
        shown = {k: round(v, 4) if isinstance(v, float) else v for k, v in val_metrics.items()}
        print(
            f"[{cfg.get('name')} seed {tcfg.seed}] epoch {epoch + 1}/{tcfg.epochs} "
            f"loss={entry['train_loss']:.4f} val={shown}",
            flush=True,
        )
        if ckpt_path is not None:
            _save_checkpoint(
                ckpt_path,
                {
                    "epoch": epoch + 1,
                    "config_sha256": cfg_hash,
                    "source_sha256": source,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "history": history,
                    "elapsed_seconds": elapsed_before + time.time() - t0,
                    "rng": _rng_state(device),
                },
            )

    # Test evaluation sets: scored once, after the last epoch (no selection on them).
    final_evals = {
        name: evaluate(model, loader, task, device) for name, loader in loaders.tests.items()
    }
    # Peak over this process's part of the run: build, training, every evaluation.
    memory = peak_memory_stats(device, available_at_start)
    jsonable = config_to_jsonable(cfg)
    return {
        "name": cfg.get("name"),
        "config": cfg.get("config"),
        "label": cfg.get("label", cfg["model"]["name"]),
        "seed": tcfg.seed,
        "task": task,
        "model": cfg["model"],
        "train_config": jsonable["train"],
        "data_config": jsonable["data"],
        "n_parameters": count_parameters(model),
        "history": history,
        "final_evals": final_evals,
        "attention_backend": attention_backend,
        "resumed_from_epoch": resumed_from,
        "wall_time_seconds": round(elapsed_before + time.time() - t0, 1),
        "environment": _env_info(device),
        "provenance": {**run_provenance(device), "config_sha256": cfg_hash},
        "peak_gpu_mb": memory["peak_allocated_mb"] if memory else None,
        "peak_gpu_reserved_mb": memory["peak_reserved_mb"] if memory else None,
        "device_total_mb": memory["device_total_mb"] if memory else None,
        "gpu_available_at_start_mb": memory["available_at_start_mb"] if memory else None,
        "gpu_free_at_end_mb": memory["free_at_end_mb"] if memory else None,
        "gpu_memory_spilled": memory["spilled"] if memory else None,
    }
