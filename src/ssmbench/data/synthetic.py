"""Synthetic long-range tasks: Selective Copying and Parity.

Selective Copying (following the Mamba paper's task design)
----------------------------------------------------------
Each sequence is a stream of noise tokens in which K "marker -> value" pairs
are embedded; a separator token then requests the model to emit the K
remembered values, in order. The model must (a) detect markers, (b) store their
values in state, (c) filter noise, and (d) release them on cue. LTI models
cannot solve it (they cannot gate noise out of the state); it is the canonical
demonstration of *selectivity*.

Vocabulary: 0 = pad, 1 = marker, 2 = separator, 3..V-1 = content/noise values.

Targets use -100 (ignore_index) everywhere except the K positions after the
separator, where the target is the next remembered value (token-level task).

Parity (state tracking; the motivating task for Mamba-3's complex states)
-------------------------------------------------------------------------
Binary input tokens; the target at position t is the XOR (parity) of inputs
1..t. A model whose transition matrix has only real non-negative eigenvalues
(decay-only dynamics) cannot represent the required rotation h_t = R(pi x_t)
h_{t-1}; complex-valued transitions can (Mamba-3 paper, Section 3.2).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IGNORE_INDEX = -100
PAD, MARKER, SEPARATOR = 0, 1, 2


class SelectiveCopyDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        seq_len: int = 64,
        min_pairs: int = 3,
        max_pairs: int = 5,
        vocab_size: int = 20,
        seed: int = 42,
    ) -> None:
        if vocab_size < 6:
            raise ValueError("vocab_size must leave room for pad/marker/separator + values")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.min_pairs = min_pairs
        self.max_pairs = max_pairs
        self.vocab_size = vocab_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.RandomState(self.seed + idx)
        V = self.vocab_size
        k = rng.randint(self.min_pairs, self.max_pairs + 1)
        # Reserve: k answer positions + 1 sep + k markers + k values; noise fills the rest.
        noise_slots = self.seq_len - 3 * k - 1
        if noise_slots < 0:
            raise ValueError(f"seq_len {self.seq_len} too short for {k} pairs")

        values = rng.randint(3, V, size=k)
        # Build the noise stream and insert marker->value pairs at random slots.
        stream = rng.randint(3, V, size=noise_slots + k).tolist()  # noise + room for values
        pair_pos = sorted(rng.choice(len(stream), size=k, replace=False).tolist())
        inputs: list[int] = []
        remembered: list[int] = []
        vi = 0
        for si, tok in enumerate(stream):
            if vi < k and si == pair_pos[vi]:
                inputs.append(MARKER)
                inputs.append(int(values[vi]))
                remembered.append(int(values[vi]))
                vi += 1
            else:
                inputs.append(int(tok))
        inputs.append(SEPARATOR)
        if len(inputs) != self.seq_len - k:
            raise AssertionError(
                f"selective copy construction error: built {len(inputs)} tokens, "
                f"expected {self.seq_len - k}"
            )
        # Answer block: k positions of pad the model must fill with remembered values.
        inputs = inputs + [PAD] * k
        inputs = inputs + [PAD] * (self.seq_len - len(inputs))

        targets = torch.full((self.seq_len,), IGNORE_INDEX, dtype=torch.long)
        answer_start = len(inputs) - k
        targets[answer_start: answer_start + k] = torch.tensor(remembered, dtype=torch.long)
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "targets": targets,
        }


class ParityDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int = 128, seed: int = 42) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.RandomState(self.seed + idx)
        bits = rng.randint(0, 2, size=self.seq_len)
        parity = np.cumsum(bits) % 2
        return {
            "input_ids": torch.tensor(bits, dtype=torch.long),
            "targets": torch.tensor(parity, dtype=torch.long),
        }


def collate_token_level(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack fixed-length token-level samples (no padding needed)."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "targets": torch.stack([b["targets"] for b in batch]),
    }


def get_selective_copy_loaders(cfg: dict) -> tuple[DataLoader, DataLoader, int]:
    train = SelectiveCopyDataset(num_samples=cfg["train_samples"], seed=cfg.get("seed", 42), **cfg["train"])
    test = SelectiveCopyDataset(num_samples=cfg["test_samples"], seed=cfg.get("seed", 42) + 1_000_000, **cfg["test"])
    bs = cfg["batch_size"]
    return (
        DataLoader(train, batch_size=bs, shuffle=True, collate_fn=collate_token_level),
        DataLoader(test, batch_size=bs, shuffle=False, collate_fn=collate_token_level),
        cfg["train"]["vocab_size"],
    )


def get_parity_loaders(cfg: dict) -> tuple[DataLoader, DataLoader, int]:
    train = ParityDataset(num_samples=cfg["train_samples"], seed=cfg.get("seed", 42), **cfg["train"])
    test = ParityDataset(num_samples=cfg["test_samples"], seed=cfg.get("seed", 42) + 1_000_000, **cfg["test"])
    bs = cfg["batch_size"]
    return (
        DataLoader(train, batch_size=bs, shuffle=True, collate_fn=collate_token_level),
        DataLoader(test, batch_size=bs, shuffle=False, collate_fn=collate_token_level),
        2,
    )
