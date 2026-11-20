"""Run all training experiments defined in a config file.

Same as the ssmbench-train console script (see src/ssmbench/train/cli.py);
kept so existing commands keep working. Needs the package installed, for
example with `pip install -e .`.

Usage:
    python scripts/run_training.py --config configs/imdb.yaml --out results/
"""

from ssmbench.train.cli import main

if __name__ == "__main__":
    main()
