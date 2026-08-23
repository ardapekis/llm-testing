"""Uncertainty-aware agreement metrics for independent IRT implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr

FloatArray = npt.NDArray[np.float64]


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
