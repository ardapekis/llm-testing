from __future__ import annotations

import numpy as np
import pytest

from irt_rank.replay import ReplayPolicy, ResponseOracle, inversion_rate, run_replay


def test_response_oracle_rejects_duplicate_revelation() -> None:
    oracle = ResponseOracle(np.asarray([[0, 1], [1, 0]], dtype=np.uint8))
    assert oracle.reveal(np.asarray([0]), np.asarray([1])).tolist() == [1]
    with pytest.raises(ValueError, match="twice"):
        oracle.reveal(np.asarray([0]), np.asarray([1]))


def test_inversion_rate_ignores_truth_ties() -> None:
    assert inversion_rate([0.0, 1.0, 1.0], [1.0, 0.0, 0.5]) == 1.0


@pytest.mark.parametrize("policy", list(ReplayPolicy))
def test_replay_is_deterministic_and_reaches_each_budget(policy: ReplayPolicy) -> None:
    rng = np.random.default_rng(4)
    responses = rng.integers(0, 2, size=(8, 40), dtype=np.uint8)
    first = run_replay(responses, policy, seed=9, budget_fractions=(0.1, 0.2))
    second = run_replay(responses, policy, seed=9, budget_fractions=(0.1, 0.2))
    assert first == second
    assert [point.observations for point in first] == [32, 64]
    assert all(-1 <= point.kendall_tau <= 1 for point in first)
    assert all(0 <= point.inversion_rate <= 1 for point in first)
