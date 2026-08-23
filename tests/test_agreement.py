from __future__ import annotations

import numpy as np

from irt_rank.irt.agreement import evaluate_difficulty_agreement


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
