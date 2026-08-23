"""Synthetic IRT response generation for recovery experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from irt_rank.irt.model import ItemParameters, probability

FloatArray = npt.NDArray[np.float64]
UIntArray = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class SyntheticData:
    """Known latent state and sampled binary responses."""

    abilities: FloatArray
    parameters: ItemParameters
    responses: UIntArray


def generate_responses(
    abilities: npt.ArrayLike,
    parameters: ItemParameters,
    *,
    seed: int,
) -> UIntArray:
    """Sample independent Bernoulli responses from an IRT model."""

    rng = np.random.default_rng(seed)
    probs = probability(abilities, parameters)
    return np.asarray(rng.binomial(1, probs), dtype=np.uint8)


def recovery_dataset(
    *,
    models: int = 50,
    items: int = 1_000,
    discrimination: float = 2.5,
    seed: int = 20260823,
) -> SyntheticData:
    """Generate the preregistered M1 1PL recovery dataset.

    The original M1-A run used this exact regime. Its difficulty-RMSE threshold
    was not statistically attainable with 50 response rows, so revised gates
    must treat the observed item difficulty error as a diagnostic rather than
    silently changing this generator after observing the result.
    """

    rng = np.random.default_rng(seed)
    abilities = np.sort(rng.normal(0.0, 1.0, size=models))
    difficulties = rng.uniform(-1.25, 1.25, size=items)
    parameters = ItemParameters(
        np.full(items, discrimination, dtype=np.float64),
        np.asarray(difficulties, dtype=np.float64),
        np.zeros(items, dtype=np.float64),
    )
    responses = generate_responses(abilities, parameters, seed=seed + 1)
    return SyntheticData(np.asarray(abilities), parameters, responses)
