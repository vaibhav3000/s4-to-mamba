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

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from .loaders import TaskLoaders

_PAD, _UNK = 0, 1
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class IMDBDataset(Dataset):
    def __init__(
        self, texts: list[str], labels: list[int], vocab: dict[str, int], max_len: int
    ) -> None:
        self.encoded = [
            torch.tensor(
                [vocab.get(w, _UNK) for w in _tokenize(t)][:max_len] or [_UNK],
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
    for i, (b, L) in enumerate(zip(batch, lengths, strict=True)):
        tokens[i, :L] = b["input_ids"]
    labels = torch.stack([b["label"] for b in batch])
    return {"input_ids": tokens, "label": labels}


def build_vocab(texts: list[str], max_features: int) -> dict[str, int]:
    counter = Counter(tok for t in texts for tok in _tokenize(t))
    # +2 reserves ids 0 (pad) and 1 (unk).
    most_common = [w for w, _ in counter.most_common(max_features - 2)]
    return {w: i + 2 for i, w in enumerate(most_common)}


def split_train_indices(
    n: int, train_size: int, val_size: int, seed: int
) -> tuple[list[int], list[int]]:
    """Disjoint training and validation indices into a split of n examples, from one seeded
    permutation: the first train_size go to training, the next val_size to validation."""
    if train_size + val_size > n:
        raise ValueError(f"train_size {train_size} + val_size {val_size} exceeds {n} examples")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return perm[:train_size], perm[train_size : train_size + val_size]


def get_imdb_loaders(
    max_features: int = 20000,
    max_len: int = 1024,
    batch_size: int = 32,
    train_subset: int | None = None,
    val_size: int = 2500,
    test_subset: int | None = None,
    num_workers: int = 0,
    seed: int = 42,
    cache_dir: str | Path | None = None,
) -> tuple[TaskLoaders, dict[str, int]]:
    """Loaders over the standard IMDb splits, and the vocabulary.

    Training and validation are disjoint draws from the training split (validation holds
    val_size reviews; training takes train_subset, or the rest); the vocabulary is built
    from the training reviews only. The single test set, "test", is the test split (its
    first test_subset reviews when given). seed is the data seed.
    """
    ds = load_dataset("imdb", cache_dir=str(cache_dir) if cache_dir else None)
    texts, labels = ds["train"]["text"], ds["train"]["label"]
    test_texts, test_labels = ds["test"]["text"], ds["test"]["label"]
    train_size = train_subset if train_subset else len(texts) - val_size
    train_idx, val_idx = split_train_indices(len(texts), train_size, val_size, seed)
    if test_subset:
        test_texts, test_labels = test_texts[:test_subset], test_labels[:test_subset]

    train_texts = [texts[i] for i in train_idx]
    vocab = build_vocab(train_texts, max_features)

    def loader(subset_texts: list[str], subset_labels: list[int], shuffle: bool) -> DataLoader:
        return DataLoader(
            IMDBDataset(subset_texts, subset_labels, vocab, max_len),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate,
            num_workers=num_workers,
        )

    loaders = TaskLoaders(
        train=loader(train_texts, [labels[i] for i in train_idx], shuffle=True),
        val=loader([texts[i] for i in val_idx], [labels[i] for i in val_idx], shuffle=False),
        tests={"test": loader(list(test_texts), list(test_labels), shuffle=False)},
    )
    return loaders, vocab
