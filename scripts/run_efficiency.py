"""Run the sequence-length efficiency benchmark.

Usage:
    python scripts/run_efficiency.py --config configs/efficiency.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ssmbench.benchmarks import run_efficiency  # noqa: E402
from ssmbench.utils import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run_efficiency(cfg)


if __name__ == "__main__":
    main()
