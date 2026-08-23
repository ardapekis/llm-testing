from __future__ import annotations

import numpy as np
import pytest

from irt_rank.irt.model import ItemParameters, fisher_information, probability


def test_three_pl_probability_has_guessing_floor() -> None:
    parameters = ItemParameters(
        np.asarray([1.5]),
        np.asarray([0.25]),
        np.asarray([0.20]),
    )

    low, midpoint, high = probability(np.asarray([-100.0, 0.25, 100.0]), parameters)[:, 0]

    assert low == pytest.approx(0.20)
    assert midpoint == pytest.approx(0.60)
    assert high == pytest.approx(1.0)


def test_three_pl_fisher_information_matches_numerical_derivative() -> None:
    parameters = ItemParameters(
        np.asarray([1.7, 0.8]),
        np.asarray([-0.2, 0.7]),
        np.asarray([0.15, 0.0]),
    )
    theta = 0.35
    step = 1e-6
    derivative = (
        probability(theta + step, parameters) - probability(theta - step, parameters)
    ) / (2.0 * step)
    fitted = probability(theta, parameters)
    numerical = np.square(derivative) / (fitted * (1.0 - fitted))

    assert fisher_information(theta, parameters) == pytest.approx(numerical, rel=1e-7)
    assert fisher_information(theta, parameters)[0] != pytest.approx(
        np.square(parameters.discrimination[0]) * fitted[0] * (1.0 - fitted[0])
    )


def test_item_parameters_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        ItemParameters(np.asarray([0.0]), np.asarray([0.0]), np.asarray([0.0]))
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        ItemParameters(np.asarray([1.0]), np.asarray([0.0]), np.asarray([1.0]))

