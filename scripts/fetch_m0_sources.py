#!/usr/bin/env python3
"""Fetch exact M0 source revisions into an ignored raw-data directory."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from build_m0_data import SWEBENCH_COMMIT, TINYBENCH_COMMIT


def clone_at(url: str, commit: str, destination: Path) -> None:
    if destination.exists():
        actual = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != commit:
            raise ValueError(f"{destination} is at {actual}, expected {commit}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--filter=blob:none", url, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", commit], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clone_at(
        "https://github.com/felipemaiapolo/tinyBenchmarks.git",
        TINYBENCH_COMMIT,
        args.raw_root / "tinyBenchmarks",
    )
    clone_at(
        "https://github.com/SWE-bench/experiments.git",
        SWEBENCH_COMMIT,
        args.raw_root / "swebench-experiments",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

