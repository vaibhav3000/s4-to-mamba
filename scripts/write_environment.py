"""Write results/ENVIRONMENT.json: the environment the committed results were produced in.

Usage: .venv/Scripts/python.exe scripts/write_environment.py
"""

from __future__ import annotations

from pathlib import Path

from ssmbench.utils import dump_json, environment_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = environment_manifest()
    dump_json(manifest, ROOT / "results" / "ENVIRONMENT.json")
    print(f"torch {manifest['torch']}, CUDA {manifest['cuda']}, GPU {manifest['gpu']}")


if __name__ == "__main__":
    main()
