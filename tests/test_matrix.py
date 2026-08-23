from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from irt_rank.data.matrix import load_long_response_matrix


def _write_rows(path: Path, rows: list[tuple[str, str, bool]]) -> None:
    table = pa.table(
        {
            "model_id": [row[0] for row in rows],
            "item_id": [row[1] for row in rows],
            "correct": [row[2] for row in rows],
        }
    )
    pq.write_table(table, path)


def test_load_long_response_matrix_uses_explicit_sorted_axes(tmp_path: Path) -> None:
    path = tmp_path / "responses.parquet"
    _write_rows(
        path,
        [
            ("m2", "i2", True),
            ("m1", "i1", True),
            ("m2", "i1", False),
            ("m1", "i2", False),
        ],
    )

    matrix = load_long_response_matrix(path)

    assert matrix.model_ids == ("m1", "m2")
    assert matrix.item_ids == ("i1", "i2")
    assert matrix.responses.tolist() == [[1, 0], [0, 1]]


@pytest.mark.parametrize(
    "rows",
    [
        [("m1", "i1", True), ("m1", "i1", False)],
        [("m1", "i1", True), ("m2", "i2", False)],
    ],
)
def test_load_long_response_matrix_rejects_non_dense_or_duplicate_pairs(
    tmp_path: Path, rows: list[tuple[str, str, bool]]
) -> None:
    path = tmp_path / "responses.parquet"
    _write_rows(path, rows)

    with pytest.raises(ValueError):
        load_long_response_matrix(path)
