"""Config-driven training loop with deterministic seeding and JSON results.

Supports two task types:
- "classification": sequence-level labels (IMDb). Metrics: accuracy, AUC.
- "token_classification": per-token targets with -100 ignore (copy, parity).
  Metrics: token accuracy and, for selective copy, full-sequence success rate.

Every run writes <out_dir>/<run_name>/results.json containing the config
snapshot, per-epoch history, final metrics, parameter count and environment
information — the single source of truth for the paper's result tables.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..data.imdb import get_imdb_loaders
from ..data.synthetic import get_parity_loaders, get_selective_copy_loaders
from ..models import build_backbone
from ..models.blocks import SequenceClassifier, TokenLevelClassifier
from ..utils import config_to_jsonable, count_parameters, dump_json, resolve_device, seed_everything


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    scheduler: str = "cosine"  # "cosine" | "none"
    warmup_frac: float = 0.05
    amp: bool = False          # bfloat16 autocast on CUDA
    seed: int = 42
    optimizer: str = "adamw"


def _env_info(device: torch.device) -> dict:
    info = {
        "torch": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda
    return info


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


def get_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    dcfg = cfg["data"]
    name = dcfg["name"]
    if name == "imdb":
        train, val, _vocab, _ = get_imdb_loaders(
            max_features=dcfg.get("max_features", 20000),
            max_len=dcfg.get("max_len", 1024),
            batch_size=dcfg.get("batch_size", 32),
            train_subset=dcfg.get("train_subset"),
            val_subset=dcfg.get("val_subset"),
            seed=cfg["train"].get("seed", 42),
        )
        return train, val
    if name == "selective_copy":
        train, val, _ = get_selective_copy_loaders(dcfg)
        return train, val
    if name == "parity":
        train, val, _ = get_parity_loaders(dcfg)
        return train, val
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
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
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


def run_training(cfg: dict, out_dir: str | Path) -> dict:
    """Train one model on one task; write results.json; return the results dict."""
    out_dir = Path(out_dir)
    tcfg = TrainConfig(**cfg["train"])
    seed_everything(tcfg.seed)
    device = resolve_device(cfg.get("device", "auto"))

    model = build_model(cfg).to(device)
    train_loader, val_loader = get_loaders(cfg)
    task = cfg["task"]["type"]

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": tcfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=tcfg.lr,
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * tcfg.epochs
    warmup = max(1, int(tcfg.warmup_frac * total_steps))

    def lr_lambda(step: int) -> float:
        if tcfg.scheduler == "none":
            return 1.0
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history: list[dict] = []
    t0 = time.time()
    for epoch in range(tcfg.epochs):
        model.train()
        ep_loss, ep_batches = 0.0, 0
        ep_start = time.time()
        for batch in train_loader:
            tokens = batch["input_ids"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=tcfg.amp and device.type == "cuda"):
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
            ep_loss += float(loss)
            ep_batches += 1
        val_metrics = evaluate(model, val_loader, task, device)
        entry = {
            "epoch": epoch + 1,
            "train_loss": ep_loss / max(ep_batches, 1),
            "epoch_seconds": round(time.time() - ep_start, 2),
            **val_metrics,
        }
        history.append(entry)
        print(
            f"[{cfg['name']}] epoch {epoch+1}/{tcfg.epochs} "
            f"loss={entry['train_loss']:.4f} val={ {k: round(v, 4) if isinstance(v, float) else v for k, v in val_metrics.items()} }",
            flush=True,
        )

    if device.type == "cuda":
        peak_gpu_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    else:
        peak_gpu_mb = None
    results = {
        "name": cfg["name"],
        "task": task,
        "model": cfg["model"],
        "train_config": config_to_jsonable(cfg)["train"],
        "data_config": config_to_jsonable(cfg)["data"],
        "n_parameters": count_parameters(model),
        "history": history,
        "wall_time_seconds": round(time.time() - t0, 1),
        "environment": _env_info(device),
        "peak_gpu_mb": peak_gpu_mb,
    }
    dump_json(results, out_dir / f"{cfg['name']}.json")
    return results
