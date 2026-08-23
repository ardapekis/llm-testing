"""Load a validated long-format response artifact into explicit dense axes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

UIntArray = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class DenseResponseMatrix:
    """Stable model/item axes and their binary response matrix."""

    model_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    responses: UIntArray


def load_long_response_matrix(path: Path) -> DenseResponseMatrix:
    """Load pair-unique complete long data without relying on row order."""

    table = pq.read_table(path, columns=["model_id", "item_id", "correct"])
    model_values = table["model_id"].to_pylist()
    item_values = table["item_id"].to_pylist()
    correct_values = table["correct"].to_pylist()
    if any(value is None for value in (*model_values, *item_values, *correct_values)):
        raise ValueError("response artifact contains nulls")

    model_ids = tuple(sorted(set(model_values)))
    item_ids = tuple(sorted(set(item_values)))
    model_index = {model_id: index for index, model_id in enumerate(model_ids)}
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    missing = np.uint8(255)
    responses = np.full((len(model_ids), len(item_ids)), missing, dtype=np.uint8)

    for model_id, item_id, correct in zip(
        model_values, item_values, correct_values, strict=True
    ):
        row = model_index[model_id]
        column = item_index[item_id]
        if responses[row, column] != missing:
            raise ValueError(f"duplicate response pair: {model_id!r}, {item_id!r}")
        responses[row, column] = np.uint8(bool(correct))

    if bool(np.any(responses == missing)):
        raise ValueError("response artifact is not a complete dense matrix")
    return DenseResponseMatrix(model_ids, item_ids, responses)
