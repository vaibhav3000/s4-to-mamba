"""The GPU run plan, the smoke check and the environment manifest, exercised on CPU."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from ssmbench.utils import environment_manifest, load_config

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_names_the_environment_and_the_lock() -> None:
    manifest = environment_manifest()
    for key in ("python", "torch", "cuda", "cudnn", "git_sha", "source_sha256", "packages"):
        assert key in manifest, key
    assert manifest["packages"]["torch"].startswith("2.14")
    assert "hostname" not in manifest  # results are committed; no machine names


def test_plan_runs_every_job_in_order_with_logs(tmp_path: Path) -> None:
    plan = _script("run_gpu_plan")
    steps = plan.build_plan(seeds=[0, 1, 2])
    names = [s.name for s in steps]
    assert names[0] == "environment" and names[1] == "smoke"
    for config in ("parity", "selective_copy", "exact_copy", "exact_copy_varlen", "imdb"):
        assert f"train-{config}" in names
        train = next(s for s in steps if s.name == f"train-{config}")
        assert train.argv[train.argv.index("--seeds") :][:4] == ["--seeds", "0", "1", "2"]
        assert "--checkpoints" in train.argv and "cuda" in train.argv
    assert names.index("summarize") > names.index("train-imdb")
    for job in ("efficiency-fp32", "efficiency-bf16", "inference-fp32", "inference-bf16"):
        assert job in names
    assert names[-2:] == ["figures", "tables"]

    ran: list[str] = []

    def runner(argv: list[str], log: Path) -> int:
        ran.append(argv[0])
        log.write_text("ok\n", encoding="utf-8")
        return 1 if "efficiency" in " ".join(argv) else 0

    status = plan.run_plan(steps, tmp_path / "logs", runner=runner)
    assert status != 0
    assert (tmp_path / "logs" / "train-parity.log").read_text(encoding="utf-8") == "ok\n"
    assert not (tmp_path / "logs" / "inference-fp32.log").exists()  # stopped at the failure


def test_plan_can_start_from_a_named_job(tmp_path: Path) -> None:
    plan = _script("run_gpu_plan")
    steps = plan.select(plan.build_plan(seeds=[0]), start="summarize", only=None)
    assert steps[0].name == "summarize"
    only = plan.select(plan.build_plan(seeds=[0]), start=None, only=["smoke"])
    assert [s.name for s in only] == ["smoke"]


def test_smoke_check_trains_two_steps_twice_and_compares(tmp_path: Path) -> None:
    smoke = _script("gpu_smoke")
    cfg = load_config(ROOT / "configs" / "parity.yaml")
    cfg["data"] |= {"train_samples": 16, "val_samples": 8, "test_samples": 8, "batch_size": 8}
    for spec in cfg["models"].values():
        spec.pop("attn_backend", None)  # the efficient kernel has no CPU implementation
    report = smoke.smoke_training(cfg, device="cpu", steps=2)
    assert set(report) == set(cfg["models"])
    for label, row in report.items():
        assert row["deterministic"] is True, label
        assert row["finite"] is True, label


def test_smoke_check_records_attention_backends_without_failing() -> None:
    smoke = _script("gpu_smoke")
    report = smoke.smoke_attention(device="cpu", backends=["math", "efficient"], precision="fp32")
    assert report["math"]["ok"] is True and report["math"]["ran"] == ["math"]
    assert report["efficient"]["ok"] is False and report["efficient"]["unsupported"] is True
    assert json.loads(json.dumps(report)) == report  # written to results/logs as JSON
