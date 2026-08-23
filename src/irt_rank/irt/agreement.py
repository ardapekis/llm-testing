"""Uncertainty-aware agreement metrics for independent IRT implementations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.stats import spearmanr

from irt_rank.irt.model import ItemParameters, probability

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ScaleLink:
    """Reference item parameters transformed onto the local latent scale."""

    slope: float
    intercept: float
    parameters: ItemParameters
    tcc_rmse: float


@dataclass(frozen=True, slots=True)
class ParameterAgreementMetrics:
    """Linked 2PL parameter and response-function disagreement."""

    difficulty_rmse: float
    log_discrimination_rmse: float
    difficulty_spearman: float
    discrimination_spearman: float
    icc_rmse: float


def estimable_item_mask(responses: npt.ArrayLike) -> BoolArray:
    """Identify items with both response classes and therefore finite 2PL parameters."""

    matrix = np.asarray(responses)
    if matrix.ndim != 2:
        raise ValueError("responses must be a model-by-item matrix")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError("responses must be finite")
    if not bool(np.isin(matrix, (0, 1)).all()):
        raise ValueError("responses must be binary")
    return np.asarray((matrix.min(axis=0) == 0) & (matrix.max(axis=0) == 1))


def sample_estimable_item_indices(
    item_ids: tuple[str, ...],
    estimable_mask: npt.ArrayLike,
    *,
    sample_size: int,
    salt: str,
) -> npt.NDArray[np.int64]:
    """Select a stable hash-ranked sample without inspecting response values."""

    mask = np.asarray(estimable_mask, dtype=np.bool_)
    if mask.ndim != 1 or mask.size != len(item_ids):
        raise ValueError("estimable_mask must align with item_ids")
    eligible = np.flatnonzero(mask)
    if sample_size < 2 or sample_size > eligible.size:
        raise ValueError("sample_size must be between 2 and the estimable item count")
    ranked = sorted(
        eligible.tolist(),
        key=lambda index: hashlib.sha256(
            f"{salt}\0{item_ids[index]}".encode()
        ).digest(),
    )
    return np.asarray(sorted(ranked[:sample_size]), dtype=np.int64)


def _linked_parameters(
    reference: ItemParameters,
    slope: float,
    intercept: float,
) -> ItemParameters:
    return ItemParameters(
        discrimination=reference.discrimination / slope,
        difficulty=slope * reference.difficulty + intercept,
        guessing=reference.guessing,
    )


def link_reference_scale(
    local: ItemParameters,
    reference: ItemParameters,
    *,
    theta_grid: npt.ArrayLike,
) -> ScaleLink:
    """Link reference parameters to the local scale by Stocking-Lord TCC matching."""

    if local.items != reference.items:
        raise ValueError("local and reference parameters must contain the same items")
    grid = np.asarray(theta_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 3 or not bool(np.isfinite(grid).all()):
        raise ValueError("theta_grid must be a finite one-dimensional grid")
    local_tcc = probability(grid, local).sum(axis=1)

    def objective(raw: FloatArray) -> float:
        slope = float(np.exp(raw[0]))
        intercept = float(raw[1])
        linked = _linked_parameters(reference, slope, intercept)
        difference = probability(grid, linked).sum(axis=1) - local_tcc
        return float(np.mean(np.square(difference)))

    result = minimize(
        objective,
        np.zeros(2, dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(np.log(0.1), np.log(10.0)), (-6.0, 6.0)],
    )
    if not result.success:
        raise RuntimeError(f"Stocking-Lord linking failed: {result.message}")
    slope = float(np.exp(result.x[0]))
    intercept = float(result.x[1])
    return ScaleLink(
        slope=slope,
        intercept=intercept,
        parameters=_linked_parameters(reference, slope, intercept),
        tcc_rmse=float(np.sqrt(result.fun)),
    )


def parameter_agreement_metrics(
    local: ItemParameters,
    linked_reference: ItemParameters,
    *,
    theta_grid: npt.ArrayLike,
) -> ParameterAgreementMetrics:
    """Measure linked parameter and item-characteristic-curve disagreement."""

    if local.items != linked_reference.items:
        raise ValueError("local and linked reference parameters must contain the same items")
    grid = np.asarray(theta_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 3 or not bool(np.isfinite(grid).all()):
        raise ValueError("theta_grid must be a finite one-dimensional grid")
    if bool(
        np.any(local.discrimination <= 0)
        or np.any(linked_reference.discrimination <= 0)
    ):
        raise ValueError("discriminations must be positive")

    difficulty_difference = local.difficulty - linked_reference.difficulty
    log_discrimination_difference = np.log(local.discrimination) - np.log(
        linked_reference.discrimination
    )
    probability_difference = probability(grid, local) - probability(grid, linked_reference)
    return ParameterAgreementMetrics(
        difficulty_rmse=float(np.sqrt(np.mean(np.square(difficulty_difference)))),
        log_discrimination_rmse=float(
            np.sqrt(np.mean(np.square(log_discrimination_difference)))
        ),
        difficulty_spearman=float(
            spearmanr(local.difficulty, linked_reference.difficulty).statistic
        ),
        discrimination_spearman=float(
            spearmanr(local.discrimination, linked_reference.discrimination).statistic
        ),
        icc_rmse=float(np.sqrt(np.mean(np.square(probability_difference)))),
    )


@dataclass(frozen=True, slots=True)
class AgreementResult:
    """Observed agreement, Monte Carlo tolerance, and named gate checks."""

    difficulty_rmse: float
    difficulty_spearman: float
    bootstrap_rmse_quantile: float
    bootstrap_replicates: int
    gate_checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.gate_checks.values())


def evaluate_difficulty_agreement(
    local_difficulty: npt.ArrayLike,
    reference_difficulty: npt.ArrayLike,
    bootstrap_rmse: npt.ArrayLike,
    bootstrap_converged: npt.ArrayLike,
    *,
    minimum_spearman: float,
    rmse_quantile: float,
    require_all_converged: bool,
) -> AgreementResult:
    """Compare real-data disagreement with a paired Monte Carlo reference distribution."""

    local = np.asarray(local_difficulty, dtype=np.float64)
    reference = np.asarray(reference_difficulty, dtype=np.float64)
    bootstrap = np.asarray(bootstrap_rmse, dtype=np.float64)
    converged = np.asarray(bootstrap_converged, dtype=np.bool_)
    if local.ndim != 1 or reference.shape != local.shape or local.size < 2:
        raise ValueError("difficulty vectors must be one-dimensional and shape-aligned")
    if bootstrap.ndim != 1 or bootstrap.size < 1 or converged.shape != bootstrap.shape:
        raise ValueError("bootstrap RMSE and convergence arrays must be non-empty and aligned")
    if not bool(np.isfinite(local).all() and np.isfinite(reference).all()):
        raise ValueError("difficulty vectors must be finite")
    if not bool(np.isfinite(bootstrap).all()):
        raise ValueError("bootstrap RMSE values must be finite")
    if not -1.0 <= minimum_spearman < 1.0:
        raise ValueError("minimum_spearman must be in [-1, 1)")
    if not 0.0 < rmse_quantile < 1.0:
        raise ValueError("rmse_quantile must be in (0, 1)")

    difficulty_rmse = float(np.sqrt(np.mean(np.square(local - reference))))
    difficulty_spearman = float(spearmanr(local, reference).statistic)
    bootstrap_threshold = float(np.quantile(bootstrap, rmse_quantile, method="higher"))
    checks = {
        "difficulty_spearman": difficulty_spearman > minimum_spearman,
        "rmse_within_monte_carlo": difficulty_rmse <= bootstrap_threshold,
    }
    if require_all_converged:
        checks["all_bootstrap_local_fits_converged"] = bool(converged.all())
    return AgreementResult(
        difficulty_rmse=difficulty_rmse,
        difficulty_spearman=difficulty_spearman,
        bootstrap_rmse_quantile=bootstrap_threshold,
        bootstrap_replicates=int(bootstrap.size),
        gate_checks=checks,
    )
