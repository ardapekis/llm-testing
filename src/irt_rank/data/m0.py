"""M0 response-matrix validation primitives.

The gate is deliberately independent of IRT fitting. No later milestone should consume a matrix
that fails these structural checks.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class MatrixStatistics:
    """Structural statistics used to decide the M0 gate."""

    models: int
    items: int
    observed: int
    possible: int
    density: float
    correct: int
    incorrect: int

    def as_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    def passes_m0(self) -> bool:
        """Return whether this matrix meets the user-specified M0 size/density gate."""

        return self.models >= 15 and self.items >= 500 and self.density >= 0.90


def audit_dense_arrays(
    model_ids: list[str],
    item_ids: list[str],
    correctness: npt.NDArray[np.generic],
) -> MatrixStatistics:
    """Validate a dense model-by-item binary matrix and return gate statistics.

    The transformation boundary rejects duplicate identifiers, non-binary responses, and shape
    mismatches. Missing responses must be represented before this function is called; silently
    coercing them to incorrect would make the replay semantics ambiguous.
    """

    if len(model_ids) != len(set(model_ids)):
        raise ValueError("model identifiers must be unique")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("item identifiers must be unique")
    if correctness.ndim != 2:
        raise ValueError(f"correctness must be two-dimensional, got {correctness.ndim}")

    expected_shape = (len(model_ids), len(item_ids))
    if correctness.shape != expected_shape:
        raise ValueError(
            f"correctness shape {correctness.shape} does not match model/item axes {expected_shape}"
        )
    if not bool(np.isfinite(correctness).all()):
        raise ValueError("correctness contains missing or non-finite values")
    if not bool(np.isin(correctness, (0, 1)).all()):
        raise ValueError("correctness values must be binary")

    observed = int(correctness.size)
    possible = len(model_ids) * len(item_ids)
    correct = int(np.count_nonzero(correctness))
    return MatrixStatistics(
        models=len(model_ids),
        items=len(item_ids),
        observed=observed,
        possible=possible,
        density=observed / possible if possible else 0.0,
        correct=correct,
        incorrect=observed - correct,
    )


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_keys(mapping: dict[str, Any], keys: set[str], *, context: str) -> None:
    """Fail loudly when an upstream artifact changes schema."""

    missing = keys.difference(mapping)
    if missing:
        raise ValueError(f"{context} is missing required keys: {sorted(missing)}")

