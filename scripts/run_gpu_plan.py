"""Run the Step 2 GPU plan: every job in order, each logged, stopping at the first failure.

Every job resumes on its own (training runs skip finished seeds and continue from epoch
checkpoints; the sweeps skip measured points), so rerunning the plan after an
interruption continues where it stopped. Jobs (docs/design/step2.md section 8):

    environment          results/ENVIRONMENT.json
    smoke                two training steps per variant, twice (determinism), and each
                         pinned attention backend once, on CUDA
    train-<config>       ssmbench-train for the five training configs and every seed
    summarize            ssmbench-summarize over results/train/*
    efficiency-fp32/bf16 ssmbench-efficiency
    inference-fp32/bf16  ssmbench-inference
    numerics, compile    ssmbench-numerics, ssmbench-compile
    figures, tables      scripts/make_figures.py, scripts/make_tables.py

Usage (repository root, no other GPU job running, CUDA_VISIBLE_DEVICES unset):

    .venv/Scripts/python.exe scripts/run_gpu_plan.py --seeds 0 1 2
    .venv/Scripts/python.exe scripts/run_gpu_plan.py --from summarize
    .venv/Scripts/python.exe scripts/run_gpu_plan.py --only smoke

Logs go to results/logs/<job>.log (appended, one header per attempt).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIGS = ["parity", "selective_copy", "exact_copy", "exact_copy_varlen", "imdb"]
PY = sys.executable


@dataclass
class Step:
    name: str
    argv: list[str]


def build_plan(seeds: list[int]) -> list[Step]:
    """The ordered jobs of the GPU plan for the given run seeds."""
    seed_args = ["--seeds", *(str(s) for s in seeds)]
    steps = [
        Step("environment", [PY, str(ROOT / "scripts" / "write_environment.py")]),
        Step("smoke", [PY, str(ROOT / "scripts" / "gpu_smoke.py"), "--device", "cuda"]),
    ]
    for config in TRAIN_CONFIGS:
        steps.append(
            Step(
                f"train-{config}",
                [
                    PY,
                    "-m",
                    "ssmbench.train.cli",
                    "--config",
                    f"configs/{config}.yaml",
                    "--out",
                    "results/train",
                    *seed_args,
                    "--checkpoints",
                    "checkpoints",
                    "--device",
                    "cuda",
                ],
            )
        )
    summaries = [f"results/train/{c}" for c in TRAIN_CONFIGS]
    steps.append(Step("summarize", [PY, "-m", "ssmbench.analysis.summary", *summaries]))
    for precision in ("fp32", "bf16"):
        steps.append(
            Step(
                f"efficiency-{precision}",
                [
                    PY,
                    "-m",
                    "ssmbench.benchmarks.efficiency",
                    "--config",
                    f"configs/efficiency_{precision}.yaml",
                ],
            )
        )
    for precision in ("fp32", "bf16"):
        steps.append(
            Step(
                f"inference-{precision}",
                [
                    PY,
                    "-m",
                    "ssmbench.benchmarks.inference",
                    "--config",
                    f"configs/inference_{precision}.yaml",
                ],
            )
        )
    steps.append(
        Step(
            "numerics",
            [PY, "-m", "ssmbench.benchmarks.numerics", "--config", "configs/numerics.yaml"],
        )
    )
    steps.append(
        Step(
            "compile",
            [PY, "-m", "ssmbench.benchmarks.compile_check", "--config", "configs/compile.yaml"],
        )
    )
    steps.append(Step("figures", [PY, str(ROOT / "scripts" / "make_figures.py")]))
    steps.append(Step("tables", [PY, str(ROOT / "scripts" / "make_tables.py")]))
    return steps


def select(steps: list[Step], start: str | None, only: list[str] | None) -> list[Step]:
    """The jobs from `start` on, or exactly the jobs named in `only`."""
    names = [s.name for s in steps]
    for name in [start, *(only or [])]:
        if name is not None and name not in names:
            raise SystemExit(f"error: unknown job {name!r}; jobs: {', '.join(names)}")
    if only:
        return [s for s in steps if s.name in only]
    if start:
        return steps[names.index(start) :]
    return steps


def _run_logged(argv: list[str], log: Path) -> int:
    """Run argv with stdout and stderr appended to log; returns its exit code."""
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)}\n")
        fh.flush()
        result = subprocess.run(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, check=False)
        return result.returncode


def run_plan(
    steps: list[Step], log_dir: Path, runner: Callable[[list[str], Path], int] = _run_logged
) -> int:
    """Run the jobs in order; returns the first non-zero exit code, or 0."""
    log_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        print(f"[{time.strftime('%H:%M:%S')}] {step.name} ...", flush=True)
        code = runner(step.argv, log_dir / f"{step.name}.log")
        if code != 0:
            print(f"{step.name} failed with exit code {code}; see its log", flush=True)
            return code
    print("plan complete", flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--from", dest="start", default=None, help="start at this job")
    ap.add_argument("--only", nargs="+", default=None, help="run only these jobs")
    ap.add_argument("--dry-run", action="store_true", help="print the jobs and exit")
    args = ap.parse_args()
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raise SystemExit("error: CUDA_VISIBLE_DEVICES is set; unset it for the GPU plan")
    steps = select(build_plan(args.seeds), args.start, args.only)
    if args.dry_run:
        for step in steps:
            print(f"{step.name}: {' '.join(step.argv)}")
        return
    raise SystemExit(run_plan(steps, ROOT / "results" / "logs"))


if __name__ == "__main__":
    main()
