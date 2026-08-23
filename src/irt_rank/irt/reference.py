"""Adapters for independent IRT reference implementations."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt

from irt_rank.irt.model import ItemParameters


def fit_girth_rasch_difficulty(
    responses: npt.NDArray[np.uint8],
    discrimination: float,
    *,
    quadrature_bounds: tuple[float, float],
    quadrature_points: int,
) -> npt.NDArray[np.float64]:
    """Fit fixed-discrimination Rasch difficulties with girth.

    Girth's internal response lookup uses integer array indexing. Preserve the
    validated uint8 representation instead of coercing to NumPy bool, which
    NumPy interprets as a Boolean mask.
    """

    from girth import rasch_mml

    result = cast(
        dict[str, Any],
        rasch_mml(
            responses.T,
            discrimination=discrimination,
            options={
                "quadrature_bounds": quadrature_bounds,
                "quadrature_n": quadrature_points,
            },
        ),
    )
    return np.asarray(result["Difficulty"], dtype=np.float64)


def fit_girth_2pl_parameters(
    responses: npt.NDArray[np.uint8],
    *,
    quadrature_bounds: tuple[float, float],
    quadrature_points: int,
    max_iterations: int,
) -> ItemParameters:
    """Fit 2PL item parameters with girth's joint marginal-ML routine."""

    from girth import twopl_mml

    result = cast(
        dict[str, Any],
        twopl_mml(
            responses.T.astype(np.int64),
            options={
                "estimate_distribution": False,
                "initial_guess": True,
                "max_iteration": max_iterations,
                "num_processors": 1,
                "quadrature_bounds": quadrature_bounds,
                "quadrature_n": quadrature_points,
                "use_LUT": False,
            },
        ),
    )
    discrimination = np.asarray(result["Discrimination"], dtype=np.float64)
    difficulty = np.asarray(result["Difficulty"], dtype=np.float64)
    if discrimination.ndim != 1 or difficulty.shape != discrimination.shape:
        raise ValueError("girth returned unexpected 2PL parameter shapes")
    if not bool(np.isfinite(discrimination).all() and np.isfinite(difficulty).all()):
        raise ValueError("girth returned non-finite 2PL parameters")
    return ItemParameters(discrimination, difficulty, np.zeros_like(difficulty))
