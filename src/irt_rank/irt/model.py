"""Item-response curves and information for 1PL, 2PL, and 3PL models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from scipy.special import expit

FloatArray = npt.NDArray[np.float64]


class IRTModel(StrEnum):
    """Supported unidimensional dichotomous IRT families."""

    ONE_PL = "1pl"
    TWO_PL = "2pl"
    THREE_PL = "3pl"


@dataclass(frozen=True, slots=True)
class ItemParameters:
    """Per-item parameters on the identified theta ~ N(0, 1) scale."""

    discrimination: FloatArray
    difficulty: FloatArray
    guessing: FloatArray

    def __post_init__(self) -> None:
        discrimination = np.asarray(self.discrimination, dtype=np.float64)
        difficulty = np.asarray(self.difficulty, dtype=np.float64)
        guessing = np.asarray(self.guessing, dtype=np.float64)
        if discrimination.ndim != difficulty.ndim or discrimination.ndim != guessing.ndim:
            raise ValueError("item parameter arrays must have the same rank")
        if discrimination.ndim != 1:
            raise ValueError("item parameter arrays must be one-dimensional")
        if not (discrimination.shape == difficulty.shape == guessing.shape):
            raise ValueError("item parameter arrays must have the same shape")
        if not bool(np.isfinite(discrimination).all()):
            raise ValueError("discrimination must be finite")
        if not bool(np.isfinite(difficulty).all()):
            raise ValueError("difficulty must be finite")
        if not bool(np.isfinite(guessing).all()):
            raise ValueError("guessing must be finite")
        if bool(np.any(discrimination <= 0)):
            raise ValueError("discrimination must be positive")
        if bool(np.any((guessing < 0) | (guessing >= 1))):
            raise ValueError("guessing must be in [0, 1)")
        object.__setattr__(self, "discrimination", discrimination)
        object.__setattr__(self, "difficulty", difficulty)
        object.__setattr__(self, "guessing", guessing)

    @property
    def items(self) -> int:
        """Number of calibrated items."""

        return int(self.difficulty.size)


def probability(theta: npt.ArrayLike, parameters: ItemParameters) -> FloatArray:
    """Return 3PL response probabilities with theta on the leading axis.

    A scalar theta returns one probability per item. A vector theta returns a
    `theta x item` matrix.
    """

    abilities = np.asarray(theta, dtype=np.float64)
    logits = np.multiply.outer(abilities, parameters.discrimination) - (
        parameters.discrimination * parameters.difficulty
    )
    logistic = expit(logits)
    return np.asarray(
        parameters.guessing + (1.0 - parameters.guessing) * logistic,
        dtype=np.float64,
    )


def fisher_information(theta: npt.ArrayLike, parameters: ItemParameters) -> FloatArray:
    """Return Bernoulli Fisher information for the full 3PL response curve.

    For c > 0 this is not `a^2 * P * (1-P)`. The derivative of the 3PL
    probability is used explicitly.
    """

    abilities = np.asarray(theta, dtype=np.float64)
    logits = np.multiply.outer(abilities, parameters.discrimination) - (
        parameters.discrimination * parameters.difficulty
    )
    logistic = expit(logits)
    response_probability = parameters.guessing + (1.0 - parameters.guessing) * logistic
    derivative = (
        (1.0 - parameters.guessing)
        * parameters.discrimination
        * logistic
        * (1.0 - logistic)
    )
    denominator = np.clip(response_probability * (1.0 - response_probability), 1e-15, None)
    return np.asarray(np.square(derivative) / denominator, dtype=np.float64)

