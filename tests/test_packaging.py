"""Packaging guards for docs/adr/0007: torch 2.14.1 through conflicting uv extras."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_torch_extras_pin_one_build_each_and_conflict() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    assert extras["cpu"] == ["torch==2.14.1"] and extras["cu130"] == ["torch==2.14.1"]
    assert "torch>=2.14,<2.15" in project["project"]["dependencies"]
    uv = project["tool"]["uv"]
    assert uv["conflicts"] == [[{"extra": "cpu"}, {"extra": "cu130"}]]
    sources = {s["extra"]: s["index"] for s in uv["sources"]["torch"]}
    indexes = {i["name"]: i for i in uv["index"]}
    assert indexes[sources["cpu"]]["url"].endswith("/whl/cpu")
    assert indexes[sources["cu130"]]["url"].endswith("/whl/cu130")
    assert all(i["explicit"] for i in indexes.values())
    assert "pytest>=9.2" in project["dependency-groups"]["dev"]


def test_lock_holds_both_torch_builds() -> None:
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'version = "2.14.1+cpu"' in lock and 'version = "2.14.1+cu130"' in lock
