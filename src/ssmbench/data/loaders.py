"""The loaders of one task: training, validation and named test evaluation sets."""

from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass

from torch.utils.data import DataLoader

# Data-seed offsets. A dataset draws example i from RandomState(base + i), so every
# evaluation set gets a base far from the others: no two sets share an example seed
# (sets of different lengths sharing a seed would share their first tokens).
VAL_SEED_OFFSET = 500_000
TEST_SEED_OFFSET = 1_000_000  # the single legacy test set; length-indexed sets use eval_seed


def eval_seed(seed: int, length: int) -> int:
    """Data seed of the test set at `length`: seed + 1e6 * (length + 1). Distinct per length,
    and above every training, validation and legacy test seed for sets under 1e6 examples."""
    return seed + 1_000_000 * (length + 1)


@dataclass
class TaskLoaders:
    """train: shuffled training batches. val: the validation set (training distribution, own
    data seed), scored every epoch. tests: test evaluation sets by name, scored once after
    training (docs/adr/0001)."""

    train: DataLoader
    val: DataLoader
    tests: dict[str, DataLoader]


def dataset_size(loader: DataLoader) -> int:
    """Number of examples behind a loader."""
    dataset = loader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError(f"{type(dataset).__name__} has no length")
    return len(dataset)
