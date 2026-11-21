"""Generated tables, README blocks and figures (docs/adr/0006)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from fake_results import write_fake_results

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_every_table_says_pending_without_results(tmp_path: Path) -> None:
    tables = _script("make_tables")
    md = tables.render_markdown(tmp_path)
    tex = tables.render_latex(tmp_path)
    assert set(md) == set(tex) == set(tables.TABLES)
    for name in tables.TABLES:
        assert "Pending" in md[name] and "Pending" in tex[name], name


def test_training_tables_show_mean_std_seeds_parameters_and_chance(tmp_path: Path) -> None:
    write_fake_results(tmp_path)
    md = _script("make_tables").render_markdown(tmp_path)
    parity = md["parity"]
    assert "| Mamba-3 |" in parity and "1,002" in parity
    assert "72.0 ± 2.0" in parity  # mean and sample std over three seeds, in percent
    assert "n = 3" in parity and "chance 50.0" in parity
    assert parity.index("Transformer") < parity.index("S4D") < parity.index("Mamba-3")


def test_benchmark_tables_mark_points_without_timings(tmp_path: Path) -> None:
    write_fake_results(tmp_path)
    md = _script("make_tables").render_markdown(tmp_path)
    eff = md["efficiency_bf16"]
    assert "spilled" in eff and "OOM" in eff and "skipped" in eff and "unsupported" in eff
    assert "10.0" in eff  # a median step time
    inf = md["inference_fp32"]
    assert "5.0" in inf and "0.50" in inf
    assert "Mamba-3" in md["numerics"] and "1.0e-07" in md["numerics"]
    assert "Triton" in md["compile"]


def test_benchmark_tables_state_parameter_counts(tmp_path: Path) -> None:
    write_fake_results(tmp_path)
    md = _script("make_tables").render_markdown(tmp_path)
    eff = md["efficiency_fp32"]
    assert eff.splitlines()[0].startswith("| Model | Parameters | T=1024")
    assert "| Transformer, mem.-efficient attention | 2,048 |" in eff
    assert "| S4D | 512 |" in eff
    assert "not matched" in eff
    inf = md["inference_bf16"]
    assert "| Mamba-3 | 768 |" in inf and "not matched" in inf


def test_latex_escapes_and_uses_booktabs(tmp_path: Path) -> None:
    write_fake_results(tmp_path)
    tex = _script("make_tables").render_latex(tmp_path)["parity"]
    assert "\\toprule" in tex and "\\bottomrule" in tex and "$\\pm$" in tex
    assert "±" not in tex and "_" not in tex.replace("\\_", "")


def test_readme_blocks_are_replaced_between_markers(tmp_path: Path) -> None:
    tables = _script("make_tables")
    text = "intro\n<!-- generated:parity -->\nold\n<!-- /generated:parity -->\noutro\n"
    new = tables.replace_blocks(text, {"parity": "| a |\n"})
    assert new == "intro\n<!-- generated:parity -->\n| a |\n<!-- /generated:parity -->\noutro\n"
    assert tables.replace_blocks(new, {"parity": "| a |\n"}) == new


def test_check_mode_reports_drift(tmp_path: Path) -> None:
    tables = _script("make_tables")
    write_fake_results(tmp_path / "results")
    (tmp_path / "README.md").write_text(
        "<!-- generated:parity -->\n<!-- /generated:parity -->\n", encoding="utf-8"
    )
    tables.write_all(tmp_path / "results", tmp_path)
    assert tables.check_all(tmp_path / "results", tmp_path) == []
    (tmp_path / "reports" / "tables" / "imdb.tex").write_text("typed by hand\n", encoding="utf-8")
    assert tables.check_all(tmp_path / "results", tmp_path) == ["reports/tables/imdb.tex"]


def test_committed_tables_match_the_committed_results() -> None:
    """No hand-typed number: the committed tables, SUMMARY.md and README blocks are exactly
    what make_tables.py generates from the committed results."""
    tables = _script("make_tables")
    assert tables.check_all(ROOT / "results", ROOT) == []


def test_figures_are_written_from_results_and_skipped_without(tmp_path: Path) -> None:
    figures = _script("make_figures")
    write_fake_results(tmp_path / "results")
    written = figures.make_all(tmp_path / "results", tmp_path / "out")
    names = {p.name for p in written}
    for stem in (
        "fig_parity",
        "fig_selective_copy",
        "fig_exact_copy",
        "fig_imdb",
        "fig_efficiency",
        "fig_inference",
        "fig_decode_memory",
        "fig_overview",
    ):
        assert f"{stem}.pdf" in names and f"{stem}.png" in names, stem
    # One drawing per figure, written once: the report and the README share these files.
    assert {p.parent for p in written} == {tmp_path / "out"}
    assert figures.make_all(tmp_path / "empty", tmp_path / "out2") == []


def test_environment_table_lists_the_software_and_hardware(tmp_path: Path) -> None:
    tables = _script("make_tables")
    (tmp_path / "ENVIRONMENT.json").write_text(
        '{"gpu": "Fake GPU", "gpu_total_mb": 6141.0, "nvidia_driver": "1.0", "torch": "2.14.1+cu130",'
        ' "cuda": "13.0", "cudnn": 91200, "python": "3.13.7", "platform": "Windows-11",'
        ' "git_sha": "abcdef1234567890", "uv_lock_sha256": "f00d"}',
        encoding="utf-8",
    )
    md = tables.render_markdown(tmp_path)["environment"]
    for text in ("Fake GPU", "6141", "2.14.1+cu130", "13.0", "91200", "3.13.7", "abcdef1"):
        assert text in md, text
