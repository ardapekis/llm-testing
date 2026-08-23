"""Gauss-Hermite quadrature on a standard-normal ability scale."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@lru_cache(maxsize=16)
def normal_quadrature(points: int) -> tuple[FloatArray, FloatArray]:
    """Return nodes and normalized weights for N(0, 1)."""

    if points < 7:
        raise ValueError("quadrature requires at least seven points")
    raw_nodes, raw_weights = np.polynomial.hermite.hermgauss(points)
    nodes = np.sqrt(2.0) * raw_nodes
    weights = raw_weights / np.sqrt(np.pi)
    weights /= weights.sum()
    return np.asarray(nodes, dtype=np.float64), np.asarray(weights, dtype=np.float64)

