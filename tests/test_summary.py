"""Multi-seed summaries of training runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from ssmbench.analysis.summary import SummaryError, chance_levels, summarize_dir, summarize_runs


def _run(label: str, seed: int, acc: float, source: str = "abc", d_model: int = 16) -> dict:
    return {
        "config": "parity",
        "label": label,
        "seed": seed,
        "task": "token_classification",
        "model": {"name": "mamba2", "kwargs": {"d_model": d_model}},
        "train_config": {"epochs": 2, "seed": seed},
        "data_config": {"name": "parity", "seed": 0},
        "n_parameters": 1234,
        "final_evals": {"T=8": {"token_accuracy": acc, "loss": 1.0 - acc}},
        "wall_time_seconds": 10.0 + seed,
        "attention_backend": None,
        "provenance": {"source_sha256": source, "git_sha": "deadbeef"},
    }


def test_mean_std_and_values_per_variant_and_eval_set() -> None:
    summary = summarize_runs([_run("m", 0, 0.5), _run("m", 1, 0.7), _run("m", 2, 0.9)])
    stats = summary["variants"]["m"]["evals"]["T=8"]["token_accuracy"]
    assert stats["n"] == 3 and stats["values"] == [0.5, 0.7, 0.9]
    assert math.isclose(stats["mean"], 0.7, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(stats["std"], 0.2, rel_tol=0, abs_tol=1e-12)  # sample std, ddof = 1
    assert stats["min"] == 0.5 and stats["max"] == 0.9
    variant = summary["variants"]["m"]
    assert variant["seeds"] == [0, 1, 2] and variant["n_parameters"] == 1234
    assert variant["architecture"] == "mamba2"
    assert summary["source_sha256"] == "abc"


def test_single_seed_has_no_std() -> None:
    stats = summarize_runs([_run("m", 0, 0.5)])["variants"]["m"]["evals"]["T=8"]["token_accuracy"]
    assert stats["n"] == 1 and stats["std"] is None


def test_runs_from_different_code_are_refused() -> None:
    with pytest.raises(SummaryError, match="source"):
        summarize_runs([_run("m", 0, 0.5), _run("m", 1, 0.6, source="other")])


def test_seeds_of_one_variant_must_share_their_settings() -> None:
    with pytest.raises(SummaryError, match="settings"):
        summarize_runs([_run("m", 0, 0.5), _run("m", 1, 0.6, d_model=32)])


def test_duplicate_seeds_are_refused() -> None:
    with pytest.raises(SummaryError, match="seed"):
        summarize_runs([_run("m", 0, 0.5), _run("m", 0, 0.6)])


def test_chance_levels_follow_the_task_definition() -> None:
    assert chance_levels({"name": "parity"}) == {"token_accuracy": 0.5}
    assert chance_levels({"name": "imdb"}) == {"accuracy": 0.5, "auc": 0.5}
    sel = chance_levels({"name": "selective_copy", "train": {"vocab_size": 20}})
    assert math.isclose(sel["token_accuracy"], 1 / 17)
    ec = chance_levels({"name": "exact_copy", "alphabet_size": 26})
    assert math.isclose(ec["token_accuracy"], 1 / 26)


def test_summarize_dir_reads_every_seed_file_and_writes_summary(tmp_path: Path) -> None:
    for label, seed, acc in [("a", 0, 0.5), ("a", 1, 0.6), ("b", 0, 0.9)]:
        path = tmp_path / label / f"seed{seed}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_run(label, seed, acc)), encoding="utf-8")
    summary = summarize_dir(tmp_path)
    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert written == summary
    assert set(summary["variants"]) == {"a", "b"}
    assert summary["variants"]["a"]["evals"]["T=8"]["token_accuracy"]["n"] == 2
    assert summary["chance"] == {"token_accuracy": 0.5}
