"""Adapters for independent IRT reference implementations."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt


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
