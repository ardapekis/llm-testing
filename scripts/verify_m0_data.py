#!/usr/bin/env python3
"""Verify the checked-in M0 artifacts without network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from irt_rank.data.artifact import verify_processed_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix_ids = ["mmlu_openllm", "swebench_verified"]
    audits = {
        matrix_id: verify_processed_matrix(args.processed_root / matrix_id)
        for matrix_id in matrix_ids
    }
    print(
        json.dumps(
            {
                matrix_id: {
                    "models": audit.models,
                    "items": audit.items,
                    "rows": audit.rows,
                    "correct": audit.correct,
                    "density": audit.density,
                    "sha256": audit.sha256,
                }
                for matrix_id, audit in audits.items()
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
