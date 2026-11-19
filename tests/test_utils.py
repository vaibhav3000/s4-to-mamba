"""Shared utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssmbench.utils import dump_json


class _Unserializable:
    def __str__(self) -> str:
        raise RuntimeError("cannot serialize")


def test_dump_json_replaces_the_file_atomically(tmp_path: Path) -> None:
    """A write that fails part-way must leave the previous file intact and no temp
    file behind, so a crash can never leave a truncated results file to resume from."""
    path = tmp_path / "results.json"
    dump_json({"entries": [1, 2]}, path)
    with pytest.raises(RuntimeError, match="cannot serialize"):
        dump_json({"entries": [1, 2, _Unserializable()]}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"entries": [1, 2]}
    assert [p.name for p in tmp_path.iterdir()] == ["results.json"]


def test_dump_json_creates_parents_and_ends_with_newline(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "out.json"
    dump_json({"x": 1.5}, path)
    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == {"x": 1.5} and text.endswith("\n")
