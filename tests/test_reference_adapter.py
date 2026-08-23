from __future__ import annotations

import numpy as np

from irt_rank.irt.reference import fit_girth_rasch_difficulty


def test_girth_adapter_preserves_numeric_binary_indexing() -> None:
    responses = np.asarray(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )

    difficulty = fit_girth_rasch_difficulty(
        responses,
        1.0,
        quadrature_bounds=(-6.0, 6.0),
        quadrature_points=21,
    )

    assert difficulty.shape == (responses.shape[1],)
    assert np.isfinite(difficulty).all()
