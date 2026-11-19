"""Synthetic long-range tasks: Selective Copying, Parity, and Exact Copying.

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

Exact Copying (Repeat After Me formulation; Jelassi et al., ICML 2024)
----------------------------------------------------------------------
The prompt is `<bos> <string> <sep>` and the model must emit `<string> <eos>`
at the positions after the separator. Unlike selective copying, the amount of
information that must survive grows LINEARLY with the string length (every
token must be reproduced), and there is no noise to filter. This is the
formulation under which Jelassi et al. find Transformers dominate fixed-state
SSMs: attention can re-read any prompt position, while an SSM must squeeze the
whole string through its fixed-size state.

Vocabulary: 0 = pad (also the eval padding '*'), 1 = bos, 2 = sep, 3 = eos,
4..4+V-1 = string letters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .loaders import (
    TEST_SEED_OFFSET,
    VAL_SEED_OFFSET,
    TaskLoaders,
    eval_seed,
)

IGNORE_INDEX = -100
PAD, MARKER, SEPARATOR = 0, 1, 2
# Exact-copy vocabulary (separate namespace from selective copy).
EC_PAD, EC_BOS, EC_SEP, EC_EOS = 0, 1, 2, 3


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
        targets[answer_start : answer_start + k] = torch.tensor(remembered, dtype=torch.long)
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


def _loader(dataset: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_token_level
    )


def get_selective_copy_loaders(cfg: dict) -> TaskLoaders:
    """Training, validation (training distribution) and two test sets: "id" (training
    distribution, its own data seed) and "ood" (cfg["test"]: longer, more pairs)."""
    seed, bs = cfg.get("seed", 42), cfg["batch_size"]
    n_test = cfg["test_samples"]
    train = SelectiveCopyDataset(num_samples=cfg["train_samples"], seed=seed, **cfg["train"])
    val = SelectiveCopyDataset(
        num_samples=cfg.get("val_samples", n_test), seed=seed + VAL_SEED_OFFSET, **cfg["train"]
    )
    ood = SelectiveCopyDataset(num_samples=n_test, seed=seed + TEST_SEED_OFFSET, **cfg["test"])
    in_dist = SelectiveCopyDataset(
        num_samples=n_test, seed=seed + 3 * TEST_SEED_OFFSET, **cfg["train"]
    )
    return TaskLoaders(
        train=_loader(train, bs, shuffle=True),
        val=_loader(val, bs),
        tests={"id": _loader(in_dist, bs), "ood": _loader(ood, bs)},
    )


def get_parity_loaders(cfg: dict) -> TaskLoaders:
    """Training, validation (training length) and test sets: one per length in
    cfg["eval_lengths"] named "T=<length>", or the single cfg["test"] set named "test"."""
    seed, bs = cfg.get("seed", 42), cfg["batch_size"]
    n_test = cfg["test_samples"]
    train = ParityDataset(num_samples=cfg["train_samples"], seed=seed, **cfg["train"])
    val = ParityDataset(
        num_samples=cfg.get("val_samples", n_test), seed=seed + VAL_SEED_OFFSET, **cfg["train"]
    )
    if "eval_lengths" in cfg:
        tests = {
            f"T={L}": _loader(ParityDataset(n_test, seq_len=L, seed=eval_seed(seed, L)), bs)
            for L in cfg["eval_lengths"]
        }
    else:
        tests = {
            "test": _loader(
                ParityDataset(num_samples=n_test, seed=seed + TEST_SEED_OFFSET, **cfg["test"]), bs
            )
        }
    return TaskLoaders(train=_loader(train, bs, shuffle=True), val=_loader(val, bs), tests=tests)


class ExactCopyDataset(Dataset):
    """Repeat After Me style copying (Jelassi et al., ICML 2024).

    One example per fixed-length sequence:
        input  = [bos] + string + [sep] + pad...
        target = ignore  + ignore + [string + eos] at the positions right
                 after the separator, ignore on padding.

    Every string token must be reproduced, so the information to retain grows
    linearly with ``string_len`` (contrast SelectiveCopyDataset, where only
    ``k`` marker->value pairs matter regardless of noise length).

    With ``min_len`` set, each example draws its string length uniformly from
    [min_len, string_len] (variable-length training); otherwise every string has
    exactly ``string_len`` letters, drawn exactly as before min_len existed.
    """

    def __init__(
        self,
        num_samples: int,
        string_len: int,
        seq_len: int,
        alphabet_size: int = 26,
        seed: int = 42,
        min_len: int | None = None,
    ) -> None:
        if alphabet_size < 2 or alphabet_size > 26:
            raise ValueError("alphabet_size must be in [2, 26]")
        if min_len is not None and not 1 <= min_len <= string_len:
            raise ValueError(f"min_len must be in [1, {string_len}], got {min_len}")
        self.num_samples = num_samples
        self.string_len = string_len
        self.min_len = min_len
        self.alphabet_size = alphabet_size
        self.seed = seed
        required = 2 + string_len + 1 + string_len + 1  # bos + s + sep + s + eos
        if seq_len < required:
            raise ValueError(
                f"seq_len {seq_len} too small for string_len {string_len} (needs >= {required})"
            )
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.RandomState(self.seed + idx)
        L = self.string_len
        if self.min_len is not None:
            L = int(rng.randint(self.min_len, self.string_len + 1))
        letters = rng.randint(0, self.alphabet_size, size=L) + 4  # 4..3+V
        tokens = np.full(self.seq_len, EC_PAD, dtype=np.int64)
        tokens[0] = EC_BOS
        tokens[1 : 1 + L] = letters
        tokens[1 + L] = EC_SEP

        targets = np.full(self.seq_len, IGNORE_INDEX, dtype=np.int64)
        targets[2 + L : 2 + 2 * L] = letters  # reproduce the string after sep
        targets[2 + 2 * L] = EC_EOS
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
        }


def get_exact_copy_loaders(cfg: dict) -> TaskLoaders:
    """Training, validation (training distribution) and one test set per string length in
    cfg["eval_lengths"], named "L=<length>".

    Training strings have exactly cfg["train"]["max_len"] letters, or, with
    cfg["train"]["min_len"], lengths uniform in [min_len, max_len]. Every loader shares one
    sequence length (2 * max(eval_lengths) + 8) so the same trained model is evaluated
    identically at every string length, including lengths beyond the training range.
    """
    seed, bs = cfg.get("seed", 42), cfg["batch_size"]
    seq_len = 2 * max(cfg["eval_lengths"]) + 8
    alphabet = cfg["alphabet_size"]
    max_len, min_len = cfg["train"]["max_len"], cfg["train"].get("min_len")

    def training_distribution(n: int, data_seed: int) -> ExactCopyDataset:
        return ExactCopyDataset(n, max_len, seq_len, alphabet, data_seed, min_len=min_len)

    train = training_distribution(cfg["train_samples"], seed)
    val = training_distribution(cfg.get("val_samples", cfg["eval_samples"]), seed + VAL_SEED_OFFSET)
    tests = {
        f"L={L}": _loader(
            ExactCopyDataset(cfg["eval_samples"], L, seq_len, alphabet, eval_seed(seed, L)), bs
        )
        for L in cfg["eval_lengths"]
    }
    return TaskLoaders(train=_loader(train, bs, shuffle=True), val=_loader(val, bs), tests=tests)
