from __future__ import annotations

from irt_rank.irt.recovery import RecoveryGate, RecoveryMetrics

OBSERVED_V1_METRICS = RecoveryMetrics(
    difficulty_rmse=0.21386469130940508,
    theta_spearman=0.9917795028913772,
    converged=True,
)


def test_original_gate_remains_rejected() -> None:
    gate = RecoveryGate.from_config(
        {
            "difficulty_rmse_less_than": 0.15,
            "theta_spearman_greater_than": 0.98,
        }
    )

    passed, checks = gate.evaluate(OBSERVED_V1_METRICS)

    assert not passed
    assert checks == {"theta_spearman": True, "difficulty_rmse": False}


def test_rank_recovery_gate_uses_only_preregistered_required_checks() -> None:
    gate = RecoveryGate.from_config(
        {
            "required": {
                "theta_spearman_greater_than": 0.98,
                "converged": True,
            },
            "diagnostic_only": ["difficulty_rmse"],
        }
    )

    passed, checks = gate.evaluate(OBSERVED_V1_METRICS)

    assert passed
    assert checks == {"theta_spearman": True, "converged": True}


def test_rank_recovery_gate_rejects_nonconvergence() -> None:
    gate = RecoveryGate(theta_spearman_greater_than=0.98, require_convergence=True)
    metrics = RecoveryMetrics(
        difficulty_rmse=0.10,
        theta_spearman=0.99,
        converged=False,
    )

    passed, checks = gate.evaluate(metrics)

    assert not passed
    assert checks["theta_spearman"]
    assert not checks["converged"]
