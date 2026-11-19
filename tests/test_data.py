"""Evaluation sets and validation splits of every task (docs/adr/0001)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ssmbench.data.imdb import split_train_indices
from ssmbench.data.loaders import dataset_size
from ssmbench.data.synthetic import (
    EC_EOS,
    IGNORE_INDEX,
    ExactCopyDataset,
    get_exact_copy_loaders,
    get_parity_loaders,
    get_selective_copy_loaders,
)
from ssmbench.train.trainer import get_loaders


def _first_inputs(loader: torch.utils.data.DataLoader) -> torch.Tensor:
    return torch.stack([loader.dataset[i]["input_ids"] for i in range(dataset_size(loader))])


PARITY = {
    "name": "parity",
    "batch_size": 4,
    "seed": 3,
    "train_samples": 16,
    "val_samples": 8,
    "test_samples": 8,
    "train": {"seq_len": 12},
    "eval_lengths": [12, 24],
}


def test_parity_has_a_validation_set_and_one_test_set_per_length() -> None:
    loaders = get_parity_loaders(PARITY)
    assert set(loaders.tests) == {"T=12", "T=24"}
    assert loaders.tests["T=24"].dataset[0]["input_ids"].shape == (24,)
    val = _first_inputs(loaders.val)
    assert val.shape == (8, 12)
    test12 = _first_inputs(loaders.tests["T=12"])
    assert not torch.equal(val, test12)  # different data seeds, same distribution
    train = _first_inputs(loaders.train)
    assert not any(torch.equal(v, t) for v in val for t in train[:4])


SELCOPY = {
    "name": "selective_copy",
    "batch_size": 4,
    "seed": 5,
    "train_samples": 16,
    "val_samples": 8,
    "test_samples": 8,
    "train": {"seq_len": 32, "min_pairs": 2, "max_pairs": 3, "vocab_size": 12},
    "test": {"seq_len": 48, "min_pairs": 4, "max_pairs": 5, "vocab_size": 12},
}


def test_selective_copy_scores_in_and_out_of_distribution() -> None:
    loaders = get_selective_copy_loaders(SELCOPY)
    assert set(loaders.tests) == {"id", "ood"}
    assert loaders.tests["id"].dataset[0]["input_ids"].shape == (32,)
    assert loaders.tests["ood"].dataset[0]["input_ids"].shape == (48,)
    id_set = loaders.tests["id"].dataset
    n_answers_id = {
        int((id_set[i]["targets"] != IGNORE_INDEX).sum())
        for i in range(dataset_size(loaders.tests["id"]))
    }
    assert n_answers_id <= {2, 3}
    assert not torch.equal(_first_inputs(loaders.val), _first_inputs(loaders.tests["id"]))


EXACT = {
    "name": "exact_copy",
    "alphabet_size": 6,
    "batch_size": 4,
    "seed": 7,
    "train_samples": 32,
    "val_samples": 8,
    "eval_samples": 8,
    "train": {"max_len": 6},
    "eval_lengths": [3, 6, 9],
}


def test_exact_copy_fixed_length_training_strings_all_have_max_len() -> None:
    loaders = get_exact_copy_loaders(EXACT)
    assert set(loaders.tests) == {"L=3", "L=6", "L=9"}
    for i in range(dataset_size(loaders.train)):
        targets = loaders.train.dataset[i]["targets"]
        assert int((targets != IGNORE_INDEX).sum()) == 6 + 1  # string plus eos


def test_exact_copy_variable_length_training_covers_the_range() -> None:
    cfg = {**EXACT, "train": {"min_len": 1, "max_len": 6}}
    loaders = get_exact_copy_loaders(cfg)
    lengths = set()
    for i in range(dataset_size(loaders.train)):
        ex = loaders.train.dataset[i]
        scored = ex["targets"] != IGNORE_INDEX
        n = int(scored.sum()) - 1
        lengths.add(n)
        # The scored block is the string (copied from the prompt) followed by eos.
        assert torch.equal(ex["targets"][scored][:-1], ex["input_ids"][1 : 1 + n])
        assert int(ex["targets"][scored][-1]) == EC_EOS
    assert lengths <= set(range(1, 7)) and len(lengths) >= 4
    # Validation follows the training distribution: variable lengths too.
    val_lengths = {
        int((loaders.val.dataset[i]["targets"] != IGNORE_INDEX).sum()) - 1
        for i in range(dataset_size(loaders.val))
    }
    assert len(val_lengths) > 1


def test_fixed_length_exact_copy_examples_are_unchanged() -> None:
    """Adding variable lengths must not change the fixed-length examples (same draws)."""
    ds = ExactCopyDataset(num_samples=4, string_len=5, seq_len=20, alphabet_size=6, seed=11)
    rng = torch.from_numpy(np.random.RandomState(11 + 2).randint(0, 6, size=5) + 4)
    assert torch.equal(ds[2]["input_ids"][1:6], rng)


def test_split_train_indices_is_disjoint_and_seeded() -> None:
    train, val = split_train_indices(100, train_size=40, val_size=10, seed=1)
    assert len(train) == 40 and len(val) == 10
    assert not set(train) & set(val)
    assert split_train_indices(100, 40, 10, seed=1) == (train, val)
    assert split_train_indices(100, 40, 10, seed=2) != (train, val)


def test_trainer_get_loaders_returns_named_test_sets() -> None:
    loaders = get_loaders({"data": PARITY, "train": {"seed": 0}})
    assert set(loaders.tests) == {"T=12", "T=24"}
    assert dataset_size(loaders.val) == 8


def test_legacy_single_test_configs_still_load() -> None:
    """A config without val_samples or eval_lengths keeps its one test set, named "test",
    and validates on a separate draw of the training distribution."""
    cfg = {k: v for k, v in PARITY.items() if k not in ("val_samples", "eval_lengths")}
    cfg["test"] = {"seq_len": 12}
    loaders = get_parity_loaders(cfg)
    assert set(loaders.tests) == {"test"}
    assert dataset_size(loaders.val) == cfg["test_samples"]


def test_imdb_loads_the_canonical_dataset_at_a_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current huggingface_hub rejects the legacy id "imdb"; the loader asks for
    stanfordnlp/imdb at a fixed revision so every run reads the same data."""
    from ssmbench.data import imdb

    calls: list[tuple[str, dict]] = []

    def fake_load(name: str, **kwargs: object) -> dict:
        calls.append((name, dict(kwargs)))
        split = {"text": ["good film", "bad film", "fine", "awful"], "label": [1, 0, 1, 0]}
        return {"train": split, "test": split}

    monkeypatch.setattr(imdb, "load_dataset", fake_load)
    loaders, _ = imdb.get_imdb_loaders(batch_size=2, train_subset=2, val_size=2)
    assert calls[0][0] == "stanfordnlp/imdb"
    assert calls[0][1]["revision"] == imdb.IMDB_REVISION and len(imdb.IMDB_REVISION) == 40
    assert dataset_size(loaders.train) == 2 and dataset_size(loaders.val) == 2
