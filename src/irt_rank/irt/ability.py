"""MAP and EAP ability estimation with a standard-normal prior."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from irt_rank.irt.model import ItemParameters, probability
from irt_rank.irt.quadrature import normal_quadrature

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AbilityEstimate:
    """Posterior ability location and uncertainty for each response row."""

    mean: FloatArray
    standard_deviation: FloatArray


def _validated_responses(responses: npt.ArrayLike, items: int) -> tuple[FloatArray, FloatArray]:
    matrix = np.asarray(responses, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != items:
        raise ValueError(f"responses must have shape (models, {items})")
    observed = np.isfinite(matrix)
    if not bool(np.isin(matrix[observed], (0.0, 1.0)).all()):
        raise ValueError("observed responses must be binary")
    return np.where(observed, matrix, 0.0), observed.astype(np.float64)


def _log_likelihood_grid(
    responses: FloatArray,
    observed: FloatArray,
    parameters: ItemParameters,
    nodes: FloatArray,
) -> FloatArray:
    probabilities = np.clip(probability(nodes, parameters), 1e-12, 1.0 - 1e-12)
    return responses @ np.log(probabilities).T + (observed - responses) @ np.log1p(
        -probabilities
    ).T


def _map_objective(
    theta: float,
    response: FloatArray,
    observed: FloatArray,
    parameters: ItemParameters,
) -> float:
    probabilities = np.clip(probability(theta, parameters), 1e-12, 1.0 - 1e-12)
    log_likelihood = np.sum(
        response * np.log(probabilities)
        + (observed - response) * np.log1p(-probabilities)
    )
    return float(0.5 * theta * theta - log_likelihood)


def estimate_eap(
    responses: npt.ArrayLike,
    parameters: ItemParameters,
    *,
    quadrature_points: int = 61,
) -> AbilityEstimate:
    """Estimate posterior means and standard deviations by quadrature."""

    response_matrix, observed = _validated_responses(responses, parameters.items)
    nodes, weights = normal_quadrature(quadrature_points)
    log_posterior = _log_likelihood_grid(response_matrix, observed, parameters, nodes)
    log_posterior += np.log(weights)[None, :]
    log_posterior -= logsumexp(log_posterior, axis=1, keepdims=True)
    posterior = np.exp(log_posterior)
    means = posterior @ nodes
    variances = posterior @ np.square(nodes) - np.square(means)
    return AbilityEstimate(means, np.sqrt(np.maximum(variances, 0.0)))


def estimate_map(
    responses: npt.ArrayLike,
    parameters: ItemParameters,
    *,
    bounds: tuple[float, float] = (-6.0, 6.0),
) -> AbilityEstimate:
    """Estimate posterior modes under an N(0, 1) prior.

    The returned standard deviation is a Laplace approximation from the observed
    posterior curvature.
    """

    response_matrix, observed = _validated_responses(responses, parameters.items)
    means = np.empty(response_matrix.shape[0], dtype=np.float64)
    standard_deviations = np.empty_like(means)

    for row, (response, mask) in enumerate(zip(response_matrix, observed, strict=True)):
        objective = partial(
            _map_objective,
            response=response,
            observed=mask,
            parameters=parameters,
        )

        result = minimize_scalar(objective, bounds=bounds, method="bounded")
        if not result.success:
            raise RuntimeError(f"MAP optimization failed for row {row}: {result.message}")
        theta_hat = float(result.x)
        step = 1e-4 * max(1.0, abs(theta_hat))
        curvature = (
            objective(theta_hat + step)
            - 2.0 * objective(theta_hat)
            + objective(theta_hat - step)
        ) / (step * step)
        means[row] = theta_hat
        standard_deviations[row] = np.sqrt(1.0 / max(curvature, 1e-12))

    return AbilityEstimate(means, standard_deviations)
