"""Lightweight fixed-budget offline replay for exploratory ranking comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from itertools import pairwise
from math import gcd

import numpy as np
import numpy.typing as npt
from scipy.stats import kendalltau

UIntArray = npt.NDArray[np.uint8]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class ReplayPolicy(StrEnum):
    """Policies included in the reduced fixed-budget comparison."""

    RANDOM = "random"
    BALANCED_RANDOM = "balanced_random"
    CAT_SE = "cat_se"
    RANK_AWARE = "rank_aware"
    RANK_AWARE_ABLATION = "rank_aware_no_contest"


@dataclass(frozen=True, slots=True)
class ReplayPoint:
    policy: str
    seed: int
    budget_fraction: float
    observations: int
    normalized_cost: float
    kendall_tau: float
    inversion_rate: float


class ResponseOracle:
    """Reveal responses only for explicitly selected, previously unseen pairs."""

    def __init__(self, responses: npt.ArrayLike) -> None:
        matrix = np.asarray(responses, dtype=np.uint8)
        if matrix.ndim != 2 or not bool(np.isin(matrix, (0, 1)).all()):
            raise ValueError("responses must be a binary model-by-item matrix")
        self._responses = matrix
        self._revealed: set[int] = set()

    @property
    def shape(self) -> tuple[int, int]:
        return self._responses.shape

    def reveal(self, models: IntArray, items: IntArray) -> UIntArray:
        if models.shape != items.shape:
            raise ValueError("model and item selections must align")
        encoded = models * self.shape[1] + items
        if len(set(encoded.tolist())) != encoded.size:
            raise ValueError("a batch cannot contain duplicate pairs")
        if any(int(pair) in self._revealed for pair in encoded):
            raise ValueError("a response pair cannot be revealed twice")
        self._revealed.update(int(pair) for pair in encoded)
        return self._responses[models, items].copy()


def inversion_rate(truth: npt.ArrayLike, estimate: npt.ArrayLike) -> float:
    """Fraction of truth-comparable pairs ordered in the wrong direction."""

    actual = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(estimate, dtype=np.float64)
    if actual.ndim != 1 or predicted.shape != actual.shape:
        raise ValueError("rank vectors must be aligned and one-dimensional")
    left, right = np.triu_indices(actual.size, k=1)
    truth_difference = actual[left] - actual[right]
    comparable = truth_difference != 0
    if not bool(np.any(comparable)):
        return 0.0
    estimate_difference = predicted[left] - predicted[right]
    return float(np.mean(truth_difference[comparable] * estimate_difference[comparable] < 0))


def _coprime_steps(rng: np.random.Generator, models: int, items: int) -> IntArray:
    steps = np.empty(models, dtype=np.int64)
    for model in range(models):
        candidate = int(rng.integers(1, items))
        while gcd(candidate, items) != 1:
            candidate = candidate % (items - 1) + 1
        steps[model] = candidate
    return steps


def _posterior(counts: IntArray, correct: IntArray) -> tuple[FloatArray, FloatArray]:
    mean = (correct + 1.0) / (counts + 2.0)
    variance = mean * (1.0 - mean) / (counts + 3.0)
    return mean, np.sqrt(variance)


def _weights(policy: ReplayPolicy, counts: IntArray, correct: IntArray) -> FloatArray:
    models = counts.size
    if policy in {ReplayPolicy.RANDOM, ReplayPolicy.BALANCED_RANDOM}:
        return np.ones(models)
    mean, standard_error = _posterior(counts, correct)
    if policy in {ReplayPolicy.CAT_SE, ReplayPolicy.RANK_AWARE_ABLATION}:
        return standard_error + 1e-12
    order = np.argsort(mean, kind="stable")
    contest = np.zeros(models, dtype=np.float64)
    for left, right in pairwise(order):
        gap = mean[right] - mean[left]
        overlap = max(0.0, standard_error[left] + standard_error[right] - gap)
        contest[left] = max(contest[left], overlap)
        contest[right] = max(contest[right], overlap)
    scale = float(np.max(contest))
    if scale > 0:
        contest /= scale
    return standard_error * (1.0 + 3.0 * contest) + 1e-12


def _target_counts(
    current: IntArray,
    target_total: int,
    item_count: int,
    weights: FloatArray,
    rng: np.random.Generator,
    *,
    randomized: bool,
) -> IntArray:
    result = current.copy()
    remaining = target_total - int(result.sum())
    while remaining > 0:
        available = result < item_count
        probabilities = np.where(available, weights, 0.0)
        probabilities /= probabilities.sum()
        if randomized:
            addition = rng.multinomial(remaining, probabilities).astype(np.int64)
        else:
            raw = remaining * probabilities
            addition = np.floor(raw).astype(np.int64)
            leftover = remaining - int(addition.sum())
            if leftover:
                order = np.argsort(-(raw - addition), kind="stable")
                addition[order[:leftover]] += 1
        addition = np.minimum(addition, item_count - result)
        if not bool(np.any(addition)):
            raise RuntimeError("unable to allocate replay budget")
        result += addition
        remaining = target_total - int(result.sum())
    return result


def run_replay(
    responses: npt.ArrayLike,
    policy: ReplayPolicy | str,
    *,
    seed: int,
    budget_fractions: tuple[float, ...],
) -> list[ReplayPoint]:
    """Run one policy without exposing unrevealed outcomes to its allocation rule."""

    selected_policy = ReplayPolicy(policy)
    oracle = ResponseOracle(responses)
    models, items = oracle.shape
    total_pairs = models * items
    if not budget_fractions or any(
        fraction <= 0 or fraction > 1 for fraction in budget_fractions
    ):
        raise ValueError("budget fractions must be in (0, 1]")
    if tuple(sorted(budget_fractions)) != budget_fractions:
        raise ValueError("budget fractions must be sorted")
    matrix = np.asarray(responses, dtype=np.uint8)
    truth = matrix.mean(axis=1)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, items, size=models, dtype=np.int64)
    steps = _coprime_steps(rng, models, items)
    counts = np.zeros(models, dtype=np.int64)
    correct = np.zeros(models, dtype=np.int64)
    points: list[ReplayPoint] = []

    for fraction in budget_fractions:
        target = max(models, round(total_pairs * fraction))
        target = min(target, total_pairs)
        weights = _weights(selected_policy, counts, correct)
        next_counts = _target_counts(
            counts,
            target,
            items,
            weights,
            rng,
            randomized=selected_policy is ReplayPolicy.RANDOM,
        )
        for model in range(models):
            added = int(next_counts[model] - counts[model])
            if added == 0:
                continue
            offsets = np.arange(counts[model], next_counts[model], dtype=np.int64)
            item_indices = (starts[model] + steps[model] * offsets) % items
            model_indices = np.full(added, model, dtype=np.int64)
            outcomes = oracle.reveal(model_indices, item_indices)
            correct[model] += int(outcomes.sum())
        counts = next_counts
        estimate, _standard_error = _posterior(counts, correct)
        tau = float(kendalltau(truth, estimate).statistic)
        points.append(
            ReplayPoint(
                policy=selected_policy.value,
                seed=seed,
                budget_fraction=fraction,
                observations=int(counts.sum()),
                normalized_cost=float(counts.sum()),
                kendall_tau=tau,
                inversion_rate=inversion_rate(truth, estimate),
            )
        )
    return points


def points_as_dicts(points: list[ReplayPoint]) -> list[dict[str, str | int | float]]:
    return [asdict(point) for point in points]
