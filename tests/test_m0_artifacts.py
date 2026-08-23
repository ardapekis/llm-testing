from __future__ import annotations

from pathlib import Path

import pytest

from irt_rank.data.artifact import verify_processed_matrix

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("matrix_id", "models", "items", "rows"),
    [
        ("mmlu_openllm", 395, 14_042, 5_546_590),
        ("swebench_verified", 134, 500, 67_000),
    ],
)
def test_processed_m0_artifact(
    matrix_id: str,
    models: int,
    items: int,
    rows: int,
) -> None:
    audit = verify_processed_matrix(REPOSITORY / "data" / "processed" / matrix_id)

    assert audit.models == models
    assert audit.items == items
    assert audit.rows == rows
    assert audit.density == 1.0

