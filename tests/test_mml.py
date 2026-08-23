from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from irt_rank.irt.ability import estimate_eap
from irt_rank.irt.mml import MMLConfig, fit_mml
from irt_rank.irt.model import IRTModel, ItemParameters
from irt_rank.irt.synthetic import generate_responses


def test_one_pl_mml_recovers_small_synthetic_order() -> None:
    abilities = np.linspace(-1.5, 1.5, 40)
    true_parameters = ItemParameters(
        np.full(80, 2.0),
        np.linspace(-1.0, 1.0, 80),
        np.zeros(80),
    )
    responses = generate_responses(abilities, true_parameters, seed=7)

    result = fit_mml(
        responses,
        IRTModel.ONE_PL,
        config=MMLConfig(
            fixed_discrimination=2.0,
            quadrature_points=31,
            max_iterations=40,
            tolerance=1e-3,
        ),
    )
    estimates = estimate_eap(responses, result.parameters)

    assert result.converged
    difficulty_rmse = np.sqrt(
        np.mean(np.square(result.parameters.difficulty - true_parameters.difficulty))
    )
    assert difficulty_rmse < 0.35
    assert float(spearmanr(estimates.mean, abilities).statistic) > 0.95
    assert result.log_likelihood_history[-1] >= result.log_likelihood_history[0]


def test_two_pl_and_three_pl_paths_produce_valid_parameters() -> None:
    rng = np.random.default_rng(11)
    abilities = rng.normal(size=80)
    parameters = ItemParameters(
        rng.uniform(0.8, 1.6, size=5),
        rng.uniform(-0.8, 0.8, size=5),
        np.zeros(5),
    )
    responses = generate_responses(abilities, parameters, seed=12)
    config = MMLConfig(quadrature_points=21, max_iterations=3, tolerance=1e-2)

    two_pl = fit_mml(responses, IRTModel.TWO_PL, config=config)
    three_pl = fit_mml(responses, IRTModel.THREE_PL, config=config)

    assert np.all(two_pl.parameters.discrimination > 0)
    assert np.all(two_pl.parameters.guessing == 0)
    assert np.all(three_pl.parameters.discrimination > 0)
    assert np.all(three_pl.parameters.guessing >= 0)
    assert np.all(three_pl.parameters.guessing < config.maximum_guessing)
