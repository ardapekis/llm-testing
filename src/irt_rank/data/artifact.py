"""Independent checks for processed M0 artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from irt_rank.data.m0 import sha256_file


@dataclass(frozen=True, slots=True)
class ArtifactAudit:
    """Facts recomputed from a Parquet file and its manifest."""

    models: int
    items: int
    rows: int
    correct: int
    density: float
    sha256: str


def verify_processed_matrix(matrix_dir: Path) -> ArtifactAudit:
    """Verify schema, checksum, density, binary validity, and pair uniqueness."""

    manifest_path = matrix_dir / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    artifact = cast(dict[str, Any], manifest["artifact"])
    statistics = cast(dict[str, int | float], manifest["statistics"])
    parquet_path = matrix_dir / cast(str, artifact["path"])

    actual_hash = sha256_file(parquet_path)
    if actual_hash != artifact["sha256"]:
        raise ValueError(f"artifact checksum mismatch for {parquet_path}")

    parquet = pq.ParquetFile(parquet_path)
    expected_schema = pa.schema(
        [
            pa.field("model_id", pa.string(), nullable=False),
            pa.field("item_id", pa.string(), nullable=False),
            pa.field("correct", pa.bool_(), nullable=False),
        ]
    )
    if parquet.schema_arrow != expected_schema:
        raise ValueError(f"unexpected schema for {parquet_path}: {parquet.schema_arrow}")

    table = parquet.read()
    if sum(column.null_count for column in table.columns) != 0:
        raise ValueError(f"null values in {parquet_path}")
    models = int(pc.count_distinct(table["model_id"]).as_py())
    items = int(pc.count_distinct(table["item_id"]).as_py())
    rows = table.num_rows
    correct = int(pc.sum(table["correct"]).as_py())
    possible = models * items
    density = rows / possible if possible else 0.0

    grouped = table.select(["model_id", "item_id"]).group_by("model_id").aggregate(
        [("item_id", "count_distinct")]
    )
    distinct_items_per_model = grouped["item_id_count_distinct"].to_pylist()
    if any(count != items for count in distinct_items_per_model):
        raise ValueError(f"duplicate or missing model-item pairs in {parquet_path}")

    recomputed = {
        "models": models,
        "items": items,
        "observed": rows,
        "possible": possible,
        "density": density,
        "correct": correct,
        "incorrect": rows - correct,
    }
    if recomputed != statistics:
        raise ValueError(f"manifest statistics mismatch for {parquet_path}")
    if manifest.get("m0_gate_pass") is not True:
        raise ValueError(f"manifest does not record an M0 pass for {parquet_path}")

    return ArtifactAudit(models, items, rows, correct, density, actual_hash)
