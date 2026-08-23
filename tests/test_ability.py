from __future__ import annotations

import numpy as np

from irt_rank.irt.ability import estimate_eap, estimate_map
from irt_rank.irt.model import ItemParameters


def test_map_and_eap_order_clear_response_patterns() -> None:
    parameters = ItemParameters(
        np.ones(9),
        np.linspace(-2.0, 2.0, 9),
        np.zeros(9),
    )
    responses = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )

    eap = estimate_eap(responses, parameters)
    map_estimate = estimate_map(responses, parameters)

    assert np.all(np.diff(eap.mean) > 0)
    assert np.all(np.diff(map_estimate.mean) > 0)
    assert np.all(eap.standard_deviation > 0)
    assert np.all(map_estimate.standard_deviation > 0)


def test_ability_estimators_ignore_explicit_missing_values() -> None:
    parameters = ItemParameters(np.ones(3), np.zeros(3), np.zeros(3))
    complete = estimate_eap(np.asarray([[1.0, 0.0, np.nan]]), parameters)
    reduced_parameters = ItemParameters(np.ones(2), np.zeros(2), np.zeros(2))
    reduced = estimate_eap(np.asarray([[1.0, 0.0]]), reduced_parameters)

    assert complete.mean == reduced.mean
    assert complete.standard_deviation == reduced.standard_deviation

