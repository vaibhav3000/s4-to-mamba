"""IMDb sentiment classification data pipeline.

Tokenisation is deliberately simple and fully reproducible: lowercase, regex
word split, a vocabulary of the `max_features` most frequent training words.
0 = pad, 1 = unk. Dynamic padding per batch; the classifier masks padding via
padding_idx.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterator

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

_PAD, _UNK = 0, 1
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class IMDBDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_len: int) -> None:
        self.encoded = [
            torch.tensor(
                [_UNK if w not in vocab else vocab[w] for w in _tokenize(t)][:max_len]
                or [_UNK],
                dtype=torch.long,
            )
            for t in texts
        ]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.encoded[idx], "label": self.labels[idx]}


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Right-pad input_ids with 0 to the longest sequence in the batch."""
    lengths = [b["input_ids"].numel() for b in batch]
    max_len = max(lengths)
    tokens = torch.full((len(batch), max_len), _PAD, dtype=torch.long)
    for i, (b, L) in enumerate(zip(batch, lengths)):
        tokens[i, :L] = b["input_ids"]
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": tokens, "label": labels}


def build_vocab(texts: list[str], max_features: int) -> dict[str, int]:
    counter = Counter(tok for t in texts for tok in _tokenize(t))
    # +2 reserves ids 0 (pad) and 1 (unk).
    most_common = [w for w, _ in counter.most_common(max_features - 2)]
    return {w: i + 2 for i, w in enumerate(most_common)}


def get_imdb_loaders(
    max_features: int = 20000,
    max_len: int = 1024,
    batch_size: int = 32,
    train_subset: int | None = None,
    val_subset: int | None = None,
    num_workers: int = 0,
    seed: int = 42,
    cache_dir: str | Path | None = None,
) -> tuple[DataLoader, DataLoader, dict[str, int], int]:
    """Build train/val DataLoaders over the standard IMDb split.

    Returns (train_loader, val_loader, vocab, n_classes).
    `train_subset`/`val_subset` cap the number of samples (smoke tests / CPU runs).
    """
    ds = load_dataset("imdb", cache_dir=str(cache_dir) if cache_dir else None)
    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    val_texts = ds["test"]["text"]
    val_labels = ds["test"]["label"]

    if train_subset:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(train_texts), generator=g)[:train_subset].tolist()
        train_texts = [train_texts[i] for i in idx]
        train_labels = [train_labels[i] for i in idx]
    if val_subset:
        val_texts = val_texts[:val_subset]
        val_labels = val_labels[:val_subset]

    vocab = build_vocab(train_texts, max_features)
    train_ds = IMDBDataset(train_texts, train_labels, vocab, max_len)
    val_ds = IMDBDataset(val_texts, val_labels, vocab, max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
        num_workers=num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
        num_workers=num_workers,
    )
    return train_loader, val_loader, vocab, 2


def infinite_loader(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch
