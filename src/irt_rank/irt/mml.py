"""Bock-Aitkin marginal maximum-likelihood calibration by EM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit, logsumexp

from irt_rank.irt.model import IRTModel, ItemParameters, probability
from irt_rank.irt.quadrature import normal_quadrature

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class MMLConfig:
    """Numerical controls fixed before an MML calibration run."""

    quadrature_points: int = 41
    max_iterations: int = 80
    tolerance: float = 1e-4
    fixed_discrimination: float = 1.0
    minimum_discrimination: float = 0.20
    maximum_discrimination: float = 5.0
    maximum_guessing: float = 0.35
    parameter_bound: float = 6.0

    def __post_init__(self) -> None:
        if self.quadrature_points < 7:
            raise ValueError("quadrature_points must be at least 7")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if self.fixed_discrimination <= 0:
            raise ValueError("fixed_discrimination must be positive")
        if not 0 < self.minimum_discrimination < self.maximum_discrimination:
            raise ValueError("invalid discrimination bounds")
        if not 0 < self.maximum_guessing < 1:
            raise ValueError("maximum_guessing must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class MMLResult:
    """Calibrated items and EM convergence evidence."""

    parameters: ItemParameters
    iterations: int
    converged: bool
    marginal_log_likelihood: float
    log_likelihood_history: tuple[float, ...]


def _validated_responses(responses: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    matrix = np.asarray(responses, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("responses must be a model-by-item matrix")
    observed = np.isfinite(matrix)
    if not bool(np.isin(matrix[observed], (0.0, 1.0)).all()):
        raise ValueError("observed responses must be binary")
    if bool(np.any(observed.sum(axis=0) == 0)):
        raise ValueError("every item needs at least one observed response")
    if bool(np.any(observed.sum(axis=1) == 0)):
        raise ValueError("every model needs at least one observed response")
    return np.where(observed, matrix, 0.0), observed.astype(np.float64)


def _initial_parameters(
    responses: FloatArray,
    observed: FloatArray,
    model: IRTModel,
    config: MMLConfig,
) -> ItemParameters:
    proportions = (responses.sum(axis=0) + 0.5) / (observed.sum(axis=0) + 1.0)
    discrimination = np.full(
        responses.shape[1],
        config.fixed_discrimination if model is IRTModel.ONE_PL else 1.0,
        dtype=np.float64,
    )
    guessing = np.zeros(responses.shape[1], dtype=np.float64)
    if model is IRTModel.THREE_PL:
        guessing.fill(min(0.10, 0.5 * config.maximum_guessing))
        adjusted = np.clip((proportions - guessing) / (1.0 - guessing), 0.01, 0.99)
    else:
        adjusted = proportions
    difficulty = np.clip(
        -logit(adjusted) / discrimination,
        -config.parameter_bound,
        config.parameter_bound,
    )
    return ItemParameters(discrimination, difficulty, guessing)


def _expectation(
    responses: FloatArray,
    observed: FloatArray,
    parameters: ItemParameters,
    nodes: FloatArray,
    weights: FloatArray,
) -> tuple[FloatArray, float]:
    probs = np.clip(probability(nodes, parameters), 1e-12, 1.0 - 1e-12)
    log_likelihood = responses @ np.log(probs).T + (observed - responses) @ np.log1p(
        -probs
    ).T
    log_joint = log_likelihood + np.log(weights)[None, :]
    log_normalizer = logsumexp(log_joint, axis=1, keepdims=True)
    return np.exp(log_joint - log_normalizer), float(log_normalizer.sum())


def _fit_rasch_item(
    expected_correct: FloatArray,
    expected_total: FloatArray,
    nodes: FloatArray,
    discrimination: float,
    bound: float,
) -> float:
    def score(difficulty: float) -> float:
        fitted = expit(discrimination * (nodes - difficulty))
        return float(np.sum(expected_total * fitted - expected_correct))

    lower_score = score(-bound)
    upper_score = score(bound)
    if lower_score <= 0:
        return -bound
    if upper_score >= 0:
        return bound
    return float(brentq(score, -bound, bound, xtol=1e-10))


def _fit_item(
    expected_correct: FloatArray,
    expected_total: FloatArray,
    nodes: FloatArray,
    initial: tuple[float, float, float],
    model: IRTModel,
    config: MMLConfig,
) -> tuple[float, float, float]:
    initial_a, initial_b, initial_c = initial
    include_guessing = model is IRTModel.THREE_PL

    if include_guessing:
        scaled_c = np.clip(initial_c / config.maximum_guessing, 1e-5, 1.0 - 1e-5)
        start = np.asarray([np.log(initial_a), initial_b, logit(scaled_c)])
        bounds = [
            (np.log(config.minimum_discrimination), np.log(config.maximum_discrimination)),
            (-config.parameter_bound, config.parameter_bound),
            (-12.0, 12.0),
        ]
    else:
        start = np.asarray([np.log(initial_a), initial_b])
        bounds = [
            (np.log(config.minimum_discrimination), np.log(config.maximum_discrimination)),
            (-config.parameter_bound, config.parameter_bound),
        ]

    def objective(raw: FloatArray) -> tuple[float, FloatArray]:
        discrimination = float(np.exp(raw[0]))
        difficulty = float(raw[1])
        logistic = expit(discrimination * (nodes - difficulty))
        if include_guessing:
            unit_guessing = float(expit(raw[2]))
            guessing = config.maximum_guessing * unit_guessing
        else:
            unit_guessing = 0.0
            guessing = 0.0
        fitted = np.clip(guessing + (1.0 - guessing) * logistic, 1e-12, 1.0 - 1e-12)
        negative_log_likelihood = -np.sum(
            expected_correct * np.log(fitted)
            + (expected_total - expected_correct) * np.log1p(-fitted)
        )
        derivative_probability = (expected_total * fitted - expected_correct) / (
            fitted * (1.0 - fitted)
        )
        derivative_logit = (1.0 - guessing) * logistic * (1.0 - logistic)
        z = discrimination * (nodes - difficulty)
        gradient = [
            np.sum(derivative_probability * derivative_logit * z),
            np.sum(derivative_probability * derivative_logit * -discrimination),
        ]
        if include_guessing:
            derivative_guessing = (
                config.maximum_guessing * unit_guessing * (1.0 - unit_guessing)
            )
            gradient.append(
                np.sum(derivative_probability * (1.0 - logistic) * derivative_guessing)
            )
        return float(negative_log_likelihood), np.asarray(gradient, dtype=np.float64)

    result = minimize(objective, start, method="L-BFGS-B", jac=True, bounds=bounds)
    if not result.success:
        raise RuntimeError(f"item M-step failed: {result.message}")
    discrimination = float(np.exp(result.x[0]))
    difficulty = float(result.x[1])
    guessing = (
        config.maximum_guessing * float(expit(result.x[2])) if include_guessing else 0.0
    )
    return discrimination, difficulty, guessing


def fit_mml(
    responses: npt.ArrayLike,
    model: IRTModel | str = IRTModel.TWO_PL,
    *,
    config: MMLConfig | None = None,
) -> MMLResult:
    """Calibrate items by marginal-ML EM under theta ~ N(0, 1)."""

    selected_model = IRTModel(model)
    selected_config = config or MMLConfig()
    matrix, observed = _validated_responses(responses)
    nodes, weights = normal_quadrature(selected_config.quadrature_points)
    parameters = _initial_parameters(matrix, observed, selected_model, selected_config)
    history: list[float] = []
    converged = False

    for _iteration in range(1, selected_config.max_iterations + 1):
        posterior, marginal_log_likelihood = _expectation(
            matrix, observed, parameters, nodes, weights
        )
        history.append(marginal_log_likelihood)
        expected_correct = matrix.T @ posterior
        expected_total = observed.T @ posterior
        new_a = parameters.discrimination.copy()
        new_b = parameters.difficulty.copy()
        new_c = parameters.guessing.copy()

        for item in range(matrix.shape[1]):
            if selected_model is IRTModel.ONE_PL:
                new_b[item] = _fit_rasch_item(
                    expected_correct[item],
                    expected_total[item],
                    nodes,
                    selected_config.fixed_discrimination,
                    selected_config.parameter_bound,
                )
            else:
                new_a[item], new_b[item], new_c[item] = _fit_item(
                    expected_correct[item],
                    expected_total[item],
                    nodes,
                    (
                        parameters.discrimination[item],
                        parameters.difficulty[item],
                        parameters.guessing[item],
                    ),
                    selected_model,
                    selected_config,
                )

        if selected_model is IRTModel.ONE_PL:
            new_a.fill(selected_config.fixed_discrimination)
            new_c.fill(0.0)
        elif selected_model is IRTModel.TWO_PL:
            new_c.fill(0.0)

        updated = ItemParameters(new_a, new_b, new_c)
        maximum_change = max(
            float(np.max(np.abs(updated.discrimination - parameters.discrimination))),
            float(np.max(np.abs(updated.difficulty - parameters.difficulty))),
            float(np.max(np.abs(updated.guessing - parameters.guessing))),
        )
        parameters = updated
        if maximum_change < selected_config.tolerance:
            converged = True
            break

    _, final_log_likelihood = _expectation(matrix, observed, parameters, nodes, weights)
    history.append(final_log_likelihood)
    return MMLResult(
        parameters=parameters,
        iterations=len(history) - 1,
        converged=converged,
        marginal_log_likelihood=final_log_likelihood,
        log_likelihood_history=tuple(history),
    )

