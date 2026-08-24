"""Non-IRT, design-based tools for inexpensive model evaluation.

The response surrogate is allowed to be misspecified. Final score estimates use
logged inclusion probabilities to correct its errors, so prediction quality
affects variance rather than the estimand.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]


def _binary_matrix(values: npt.ArrayLike, *, name: str) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not bool(np.isin(matrix, (0.0, 1.0)).all()):
        raise ValueError(f"{name} must be a binary model-by-item matrix")
    return matrix


@dataclass(frozen=True, slots=True)
class SVDResponseSurrogate:
    """Low-rank response predictor learned from completed historical models."""

    item_mean: FloatArray
    basis: FloatArray
    residual_variance: FloatArray
    ridge: float

    @classmethod
    def fit(
        cls,
        historical_responses: npt.ArrayLike,
        *,
        rank: int = 16,
        ridge: float = 0.25,
    ) -> SVDResponseSurrogate:
        history = _binary_matrix(historical_responses, name="historical_responses")
        if rank < 1:
            raise ValueError("rank must be positive")
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        usable_rank = min(rank, min(history.shape))
        # Beta(1, 1) smoothing avoids degenerate predictions on unanimous items.
        item_mean = (history.sum(axis=0) + 1.0) / (history.shape[0] + 2.0)
        centered = history - item_mean
        _left, _singular, right = np.linalg.svd(centered, full_matrices=False)
        basis = right[:usable_rank].copy()
        reconstructed = item_mean + (centered @ basis.T) @ basis
        residual = history - np.clip(reconstructed, 0.0, 1.0)
        residual_variance = np.mean(np.square(residual), axis=0) + 1e-6
        return cls(item_mean, basis, residual_variance, float(ridge))

    @property
    def item_count(self) -> int:
        return int(self.item_mean.size)

    def predict(self, observed: npt.ArrayLike) -> FloatArray:
        """Predict every item after fitting a new model on its revealed sentinels."""

        values = np.asarray(observed, dtype=np.float64)
        if values.shape != self.item_mean.shape:
            raise ValueError("observed responses must align with the item bank")
        mask = np.isfinite(values)
        if bool(np.any(~np.isin(values[mask], (0.0, 1.0)))):
            raise ValueError("observed responses must be binary or NaN")
        if not bool(np.any(mask)):
            return self.item_mean.copy()

        observed_basis = self.basis[:, mask]
        gram = observed_basis @ observed_basis.T
        gram += self.ridge * np.eye(self.basis.shape[0])
        target = values[mask] - self.item_mean[mask]
        coefficients = np.linalg.solve(gram, observed_basis @ target)
        return np.clip(self.item_mean + coefficients @ self.basis, 1e-6, 1.0 - 1e-6)

    def acquisition_scores(self, prediction: npt.ArrayLike) -> FloatArray:
        """Return non-negative error-risk scores without consulting hidden outcomes."""

        predicted = np.asarray(prediction, dtype=np.float64)
        if predicted.shape != self.item_mean.shape or not bool(np.isfinite(predicted).all()):
            raise ValueError("prediction must be a finite item-aligned vector")
        leverage = np.sum(np.square(self.basis), axis=0)
        return np.asarray(
            self.residual_variance + predicted * (1.0 - predicted) * leverage,
            dtype=np.float64,
        )


def infer_item_strata(item_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Infer stable subject/repository strata from the bundled benchmark IDs."""

    strata: list[str] = []
    for item_id in item_ids:
        if ":" in item_id:
            strata.append(item_id.split(":", maxsplit=1)[0])
        elif "__" in item_id and "-" in item_id:
            strata.append(item_id.rsplit("-", maxsplit=1)[0])
        else:
            strata.append("all")
    return tuple(strata)


def stratified_sentinel_indices(
    strata: npt.ArrayLike,
    size: int,
    *,
    seed: int,
) -> IntArray:
    """Choose a reproducible round-robin sentinel set across content strata."""

    labels = np.asarray(strata, dtype=str)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("strata must be a non-empty vector")
    if size < 0 or size > labels.size:
        raise ValueError("sentinel size must be between zero and the item count")
    if size == 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(labels == label)) for label in np.unique(labels)]
    group_order = rng.permutation(len(groups))
    positions = np.zeros(len(groups), dtype=np.int64)
    selected: list[int] = []
    while len(selected) < size:
        made_progress = False
        for group_index in group_order:
            position = int(positions[group_index])
            group = groups[group_index]
            if position < group.size:
                selected.append(int(group[position]))
                positions[group_index] += 1
                made_progress = True
                if len(selected) == size:
                    break
        if not made_progress:
            raise RuntimeError("unable to allocate sentinel set")
    return np.asarray(selected, dtype=np.int64)


def _bounded_probabilities(weights: FloatArray, expected_count: float) -> FloatArray:
    """Scale positive weights to Bernoulli probabilities with a fixed expected count."""

    if expected_count < 0 or expected_count > weights.size:
        raise ValueError("expected count must fit the eligible population")
    if weights.size == 0:
        return np.empty(0, dtype=np.float64)
    if expected_count == weights.size:
        return np.ones(weights.size, dtype=np.float64)
    probabilities = np.zeros(weights.size, dtype=np.float64)
    open_mask = np.ones(weights.size, dtype=np.bool_)
    remaining = float(expected_count)
    while bool(np.any(open_mask)) and remaining > 0:
        open_weights = weights[open_mask]
        proposed = remaining * open_weights / open_weights.sum()
        saturated = proposed >= 1.0
        open_indices = np.flatnonzero(open_mask)
        if not bool(np.any(saturated)):
            probabilities[open_indices] = proposed
            break
        saturated_indices = open_indices[saturated]
        probabilities[saturated_indices] = 1.0
        open_mask[saturated_indices] = False
        remaining -= saturated_indices.size
    return probabilities


@dataclass(frozen=True, slots=True)
class SamplingDesign:
    """Realized Bernoulli sample and the probabilities that generated it."""

    selected: BoolArray
    inclusion_probability: FloatArray
    sentinel: BoolArray


def randomized_active_design(
    acquisition_score: npt.ArrayLike,
    strata: npt.ArrayLike,
    sentinel_indices: npt.ArrayLike,
    *,
    expected_additional: float,
    exploration: float = 0.2,
    seed: int,
    costs: npt.ArrayLike | None = None,
) -> SamplingDesign:
    """Mix active and stratified-random sampling, then draw independent items."""

    scores = np.asarray(acquisition_score, dtype=np.float64)
    labels = np.asarray(strata, dtype=str)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("scores and strata must be aligned vectors")
    if not bool(np.isfinite(scores).all()) or bool(np.any(scores < 0)):
        raise ValueError("acquisition scores must be finite and non-negative")
    if exploration < 0 or exploration > 1:
        raise ValueError("exploration must be in [0, 1]")
    item_cost = np.ones(scores.size) if costs is None else np.asarray(costs, dtype=np.float64)
    if item_cost.shape != scores.shape or not bool(np.isfinite(item_cost).all()):
        raise ValueError("costs must be a finite item-aligned vector")
    if bool(np.any(item_cost <= 0)):
        raise ValueError("costs must be positive")

    sentinel_index = np.asarray(sentinel_indices, dtype=np.int64)
    if sentinel_index.ndim != 1 or bool(
        np.any((sentinel_index < 0) | (sentinel_index >= scores.size))
    ):
        raise ValueError("sentinel indices are out of bounds")
    if np.unique(sentinel_index).size != sentinel_index.size:
        raise ValueError("sentinel indices must be unique")
    sentinel = np.zeros(scores.size, dtype=np.bool_)
    sentinel[sentinel_index] = True
    eligible = ~sentinel
    eligible_count = int(eligible.sum())
    if expected_additional < 0 or expected_additional > eligible_count:
        raise ValueError("expected additional sample exceeds eligible items")

    active_weight = np.where(eligible, scores / item_cost, 0.0)
    if active_weight.sum() == 0:
        active_weight = eligible.astype(np.float64)
    active_weight /= active_weight.sum()

    random_weight = np.zeros(scores.size, dtype=np.float64)
    eligible_labels = np.unique(labels[eligible])
    for label in eligible_labels:
        group = eligible & (labels == label)
        random_weight[group] = 1.0 / (eligible_labels.size * int(group.sum()))
    mixed_weight = (1.0 - exploration) * active_weight + exploration * random_weight
    probabilities = np.ones(scores.size, dtype=np.float64)
    probabilities[eligible] = _bounded_probabilities(
        mixed_weight[eligible], float(expected_additional)
    )
    rng = np.random.default_rng(seed)
    selected = sentinel | (rng.random(scores.size) < probabilities)
    return SamplingDesign(selected, probabilities, sentinel)


@dataclass(frozen=True, slots=True)
class CorrectedEstimate:
    """Prediction-corrected finite-population estimate and normal interval."""

    mean: float
    standard_error: float
    lower: float
    upper: float
    observations: int


def _corrected_estimate(
    observed: npt.ArrayLike,
    prediction: npt.ArrayLike,
    inclusion_probability: npt.ArrayLike,
    *,
    confidence: float,
    bounded: bool,
) -> CorrectedEstimate:
    outcomes = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    probability = np.asarray(inclusion_probability, dtype=np.float64)
    if (
        outcomes.ndim != 1
        or predicted.shape != outcomes.shape
        or probability.shape != outcomes.shape
    ):
        raise ValueError("observed, prediction, and probabilities must be aligned vectors")
    selected = np.isfinite(outcomes)
    if not bool(np.any(selected)):
        raise ValueError("at least one outcome must be observed")
    if not bool(np.isfinite(predicted).all()):
        raise ValueError("predictions must be finite")
    if not bool(np.isfinite(probability).all()) or bool(
        np.any((probability <= 0) | (probability > 1))
    ):
        raise ValueError("inclusion probabilities must be in (0, 1]")
    residual = outcomes[selected] - predicted[selected]
    correction = np.sum(residual / probability[selected]) / outcomes.size
    mean = float(np.mean(predicted) + correction)
    variance = np.sum(
        (1.0 - probability[selected])
        * np.square(residual)
        / np.square(probability[selected])
    ) / outcomes.size**2
    standard_error = float(np.sqrt(variance))
    if confidence <= 0 or confidence >= 1:
        raise ValueError("confidence must be in (0, 1)")
    critical = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    lower = mean - critical * standard_error
    upper = mean + critical * standard_error
    if bounded:
        lower, upper = max(0.0, lower), min(1.0, upper)
    return CorrectedEstimate(mean, standard_error, lower, upper, int(selected.sum()))


def prediction_corrected_mean(
    observed: npt.ArrayLike,
    prediction: npt.ArrayLike,
    inclusion_probability: npt.ArrayLike,
    *,
    confidence: float = 0.95,
) -> CorrectedEstimate:
    """Estimate mean accuracy using a Horvitz-Thompson residual correction."""

    return _corrected_estimate(
        observed,
        prediction,
        inclusion_probability,
        confidence=confidence,
        bounded=True,
    )


def prediction_corrected_gap(
    observed_left: npt.ArrayLike,
    observed_right: npt.ArrayLike,
    prediction_left: npt.ArrayLike,
    prediction_right: npt.ArrayLike,
    inclusion_probability: npt.ArrayLike,
    *,
    confidence: float = 0.95,
) -> CorrectedEstimate:
    """Estimate a paired model gap when both models use the same sampled items."""

    left = np.asarray(observed_left, dtype=np.float64)
    right = np.asarray(observed_right, dtype=np.float64)
    if left.shape != right.shape or not bool(np.array_equal(np.isfinite(left), np.isfinite(right))):
        raise ValueError("paired outcomes must reveal the same items")
    return _corrected_estimate(
        left - right,
        np.asarray(prediction_left, dtype=np.float64)
        - np.asarray(prediction_right, dtype=np.float64),
        inclusion_probability,
        confidence=confidence,
        bounded=False,
    )
