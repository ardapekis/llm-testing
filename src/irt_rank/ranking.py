"""Anchored IRT model ranking with pairwise uncertainty and confidence tiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from irt_rank.irt.ability import AbilityPosterior, estimate_eap_posterior
from irt_rank.irt.agreement import estimable_item_mask
from irt_rank.irt.mml import MMLConfig, MMLResult, fit_mml
from irt_rank.irt.model import IRTModel, ItemParameters

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class RankingConfig:
    """Decision rule for turning posterior abilities into a partial ranking."""

    epsilon: float = 0.0
    superiority_probability: float = 0.95
    quadrature_points: int = 61

    def __post_init__(self) -> None:
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if not 0.5 < self.superiority_probability < 1.0:
            raise ValueError("superiority_probability must be in (0.5, 1)")
        if self.quadrature_points < 7:
            raise ValueError("quadrature_points must be at least 7")


@dataclass(frozen=True, slots=True)
class RankedModel:
    """One model's display order and uncertainty-aware ranking summary."""

    model_id: str
    display_rank: int
    confidence_tier: int
    posterior_mean: float
    posterior_standard_deviation: float
    expected_epsilon_rank: float


@dataclass(frozen=True, slots=True)
class RankingResult:
    """Total display order plus the authoritative pairwise partial order."""

    models: tuple[RankedModel, ...]
    model_ids: tuple[str, ...]
    pairwise_superiority: FloatArray
    confidently_better: BoolArray
    epsilon: float
    superiority_probability: float

    def probability_better(self, first: str, second: str) -> float:
        index = {model_id: position for position, model_id in enumerate(self.model_ids)}
        return float(self.pairwise_superiority[index[first], index[second]])

    @property
    def tiers(self) -> tuple[tuple[str, ...], ...]:
        grouped: dict[int, list[str]] = {}
        for model in self.models:
            grouped.setdefault(model.confidence_tier, []).append(model.model_id)
        return tuple(tuple(grouped[tier]) for tier in sorted(grouped))


@dataclass(frozen=True, slots=True)
class ItemBank:
    """Calibrated, reusable item parameters defining a common latent scale."""

    item_ids: tuple[str, ...]
    parameters: ItemParameters
    model: IRTModel

    def __post_init__(self) -> None:
        if len(self.item_ids) != self.parameters.items:
            raise ValueError("item_ids must align with item parameters")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("item_ids must be unique")

    def write_json(self, path: Path) -> None:
        payload = {
            "schema_version": 1,
            "model": self.model.value,
            "item_ids": self.item_ids,
            "discrimination": self.parameters.discrimination.tolist(),
            "difficulty": self.parameters.difficulty.tolist(),
            "guessing": self.parameters.guessing.tolist(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def read_json(cls, path: Path) -> ItemBank:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported item-bank schema")
        parameters = ItemParameters(
            np.asarray(payload["discrimination"], dtype=np.float64),
            np.asarray(payload["difficulty"], dtype=np.float64),
            np.asarray(payload["guessing"], dtype=np.float64),
        )
        return cls(tuple(payload["item_ids"]), parameters, IRTModel(payload["model"]))


def calibrate_item_bank(
    responses: npt.ArrayLike,
    item_ids: tuple[str, ...],
    *,
    model: IRTModel | str = IRTModel.TWO_PL,
    config: MMLConfig | None = None,
) -> tuple[ItemBank, MMLResult]:
    """Calibrate estimable items and return an anchor bank on the N(0,1) scale."""

    matrix = np.asarray(responses)
    if matrix.ndim != 2 or matrix.shape[1] != len(item_ids):
        raise ValueError("responses must align with item_ids")
    mask = estimable_item_mask(matrix)
    if int(mask.sum()) < 2:
        raise ValueError("at least two items with both response classes are required")
    selected_model = IRTModel(model)
    fitted = fit_mml(matrix[:, mask], selected_model, config=config)
    bank = ItemBank(
        tuple(item for item, keep in zip(item_ids, mask, strict=True) if keep),
        fitted.parameters,
        selected_model,
    )
    return bank, fitted


def align_responses_to_item_bank(
    responses: npt.ArrayLike,
    item_ids: tuple[str, ...],
    bank: ItemBank,
) -> FloatArray:
    """Select and order response columns exactly as required by an anchor bank."""

    matrix = np.asarray(responses, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(item_ids):
        raise ValueError("responses must align with item_ids")
    positions = {item_id: index for index, item_id in enumerate(item_ids)}
    missing = [item_id for item_id in bank.item_ids if item_id not in positions]
    if missing:
        raise ValueError(f"response matrix is missing {len(missing)} anchored items")
    columns = [positions[item_id] for item_id in bank.item_ids]
    return matrix[:, columns]


def pairwise_superiority_probabilities(
    posterior: AbilityPosterior,
    *,
    epsilon: float,
) -> FloatArray:
    """Compute P(theta_i > theta_j + epsilon) from quadrature posterior mass."""

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    nodes = posterior.nodes
    mass = posterior.mass
    if mass.ndim != 2 or mass.shape[1] != nodes.size:
        raise ValueError("posterior mass must align with quadrature nodes")
    cumulative = np.concatenate(
        [np.zeros((mass.shape[0], 1)), np.cumsum(mass, axis=1)], axis=1
    )
    thresholds = nodes - epsilon
    lower = np.searchsorted(nodes, thresholds, side="left")
    upper = np.searchsorted(nodes, thresholds, side="right")
    cdf_at_threshold = cumulative[:, lower]
    exact = upper > lower
    if bool(np.any(exact)):
        cdf_at_threshold[:, exact] += 0.5 * mass[:, lower[exact]]
    result = np.empty((mass.shape[0], mass.shape[0]), dtype=np.float64)
    for model in range(mass.shape[0]):
        result[model] = cdf_at_threshold @ mass[model]
    if epsilon == 0:
        np.fill_diagonal(result, 0.5)
    return result


def _confidence_tiers(confidently_better: BoolArray, means: FloatArray) -> npt.NDArray[np.int64]:
    remaining = list(np.argsort(-means, kind="stable"))
    tiers = np.zeros(means.size, dtype=np.int64)
    tier = 1
    while remaining:
        submatrix = confidently_better[np.ix_(remaining, remaining)]
        nondominated = [
            model
            for position, model in enumerate(remaining)
            if not bool(submatrix[:, position].any())
        ]
        if not nondominated:
            nondominated = [remaining[0]]
        for model in nondominated:
            tiers[model] = tier
        removed = set(nondominated)
        remaining = [model for model in remaining if model not in removed]
        tier += 1
    return tiers


def rank_models(
    responses: npt.ArrayLike,
    model_ids: tuple[str, ...],
    bank: ItemBank,
    *,
    config: RankingConfig | None = None,
) -> RankingResult:
    """Rank models against a fixed item bank and retain unresolved comparisons."""

    selected_config = config or RankingConfig()
    matrix = np.asarray(responses, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (len(model_ids), bank.parameters.items):
        raise ValueError("responses must have one aligned row per model and item-bank column")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model_ids must be unique")
    posterior = estimate_eap_posterior(
        matrix,
        bank.parameters,
        quadrature_points=selected_config.quadrature_points,
    )
    pairwise = pairwise_superiority_probabilities(
        posterior,
        epsilon=selected_config.epsilon,
    )
    confident = pairwise >= selected_config.superiority_probability
    np.fill_diagonal(confident, False)
    means = posterior.estimate.mean
    tiers = _confidence_tiers(confident, means)
    expected_ranks = 1.0 + pairwise.sum(axis=0) - np.diag(pairwise)
    order = np.argsort(-means, kind="stable")
    rows = tuple(
        RankedModel(
            model_id=model_ids[index],
            display_rank=position,
            confidence_tier=int(tiers[index]),
            posterior_mean=float(means[index]),
            posterior_standard_deviation=float(
                posterior.estimate.standard_deviation[index]
            ),
            expected_epsilon_rank=float(expected_ranks[index]),
        )
        for position, index in enumerate(order, start=1)
    )
    return RankingResult(
        models=rows,
        model_ids=model_ids,
        pairwise_superiority=pairwise,
        confidently_better=confident,
        epsilon=selected_config.epsilon,
        superiority_probability=selected_config.superiority_probability,
    )
