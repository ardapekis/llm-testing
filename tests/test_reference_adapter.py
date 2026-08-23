from __future__ import annotations

import numpy as np

from irt_rank.irt.model import ItemParameters
from irt_rank.irt.reference import fit_girth_2pl_parameters, fit_girth_rasch_difficulty
from irt_rank.irt.synthetic import generate_responses


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


def test_girth_2pl_adapter_returns_finite_item_parameters() -> None:
    rng = np.random.default_rng(22)
    abilities = rng.normal(size=120)
    parameters = ItemParameters(
        discrimination=np.asarray([0.8, 1.0, 1.2, 1.5]),
        difficulty=np.asarray([-1.0, -0.2, 0.5, 1.1]),
        guessing=np.zeros(4),
    )
    responses = generate_responses(abilities, parameters, seed=23)

    fitted = fit_girth_2pl_parameters(
        responses,
        quadrature_bounds=(-6.0, 6.0),
        quadrature_points=21,
        max_iterations=3,
    )

    assert fitted.items == parameters.items
    assert np.isfinite(fitted.discrimination).all()
    assert np.isfinite(fitted.difficulty).all()
