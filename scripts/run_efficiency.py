"""Run the sequence-length efficiency benchmark.

Same as the ssmbench-efficiency console script (see
src/ssmbench/benchmarks/efficiency.py); kept so existing commands keep working.
Needs the package installed, for example with `pip install -e .`.

Usage:
    python scripts/run_efficiency.py --config configs/efficiency.yaml [--force]
"""

from ssmbench.benchmarks.efficiency import main

if __name__ == "__main__":
    main()
