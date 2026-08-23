from __future__ import annotations

import numpy as np
import pytest

from irt_rank.irt.agreement import (
    estimable_item_mask,
    evaluate_difficulty_agreement,
    link_reference_scale,
    parameter_agreement_metrics,
    sample_estimable_item_indices,
)
from irt_rank.irt.model import ItemParameters


def test_estimable_item_mask_excludes_constant_response_columns() -> None:
    responses = np.asarray(
        [
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
        ],
        dtype=np.uint8,
    )

    assert estimable_item_mask(responses).tolist() == [True, False, False, True]


def test_lightweight_item_sample_is_stable_and_only_uses_estimable_items() -> None:
    item_ids = ("a", "b", "c", "d", "e", "f")
    estimable = np.asarray([True, False, True, True, False, True])
    first = sample_estimable_item_indices(item_ids, estimable, sample_size=3, salt="fixed")
    second = sample_estimable_item_indices(item_ids, estimable, sample_size=3, salt="fixed")

    assert first.tolist() == second.tolist()
    assert first.size == 3
    assert estimable[first].all()


def test_stocking_lord_link_recovers_known_affine_scale() -> None:
    reference = ItemParameters(
        discrimination=np.asarray([0.7, 1.0, 1.3, 1.7]),
        difficulty=np.asarray([-1.2, -0.2, 0.6, 1.4]),
        guessing=np.zeros(4),
    )
    expected_slope = 1.35
    expected_intercept = -0.45
    local = ItemParameters(
        discrimination=reference.discrimination / expected_slope,
        difficulty=expected_slope * reference.difficulty + expected_intercept,
        guessing=np.zeros(4),
    )

    linked = link_reference_scale(
        local,
        reference,
        theta_grid=np.linspace(-4.0, 4.0, 161),
    )

    assert linked.slope == pytest.approx(expected_slope, abs=1e-3)
    assert linked.intercept == pytest.approx(expected_intercept, abs=1e-3)
    assert linked.parameters.discrimination == pytest.approx(local.discrimination, abs=1e-3)
    assert linked.parameters.difficulty == pytest.approx(local.difficulty, abs=1e-3)
    assert linked.tcc_rmse < 1e-5

    metrics = parameter_agreement_metrics(
        local,
        linked.parameters,
        theta_grid=np.linspace(-4.0, 4.0, 161),
    )
    assert metrics.difficulty_rmse < 1e-3
    assert metrics.log_discrimination_rmse < 1e-3
    assert metrics.icc_rmse < 1e-5


def test_reference_agreement_passes_inside_monte_carlo_tolerance() -> None:
    result = evaluate_difficulty_agreement(
        [-1.0, 0.0, 1.0],
        [-0.98, 0.01, 1.02],
        [0.02, 0.03, 0.04, 0.05],
        [True, True, True, True],
        minimum_spearman=0.99,
        rmse_quantile=0.95,
        require_all_converged=True,
    )

    assert result.passed
    assert result.bootstrap_rmse_quantile == 0.05
    assert result.gate_checks == {
        "difficulty_spearman": True,
        "rmse_within_monte_carlo": True,
        "all_bootstrap_local_fits_converged": True,
    }


def test_reference_agreement_rejects_rank_or_convergence_failure() -> None:
    result = evaluate_difficulty_agreement(
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        np.full(5, 2.0),
        [True, True, False, True, True],
        minimum_spearman=0.99,
        rmse_quantile=0.95,
        require_all_converged=True,
    )

    assert not result.passed
    assert not result.gate_checks["difficulty_spearman"]
    assert not result.gate_checks["all_bootstrap_local_fits_converged"]
