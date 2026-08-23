"""Explicit acceptance rules for synthetic IRT recovery experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RecoveryMetrics:
    """Decision-relevant metrics produced by a recovery run."""

    difficulty_rmse: float
    theta_spearman: float
    converged: bool


@dataclass(frozen=True, slots=True)
class RecoveryGate:
    """Versionable recovery gate with optional diagnostic-only difficulty error."""

    theta_spearman_greater_than: float
    difficulty_rmse_less_than: float | None = None
    require_convergence: bool = False

    def __post_init__(self) -> None:
        if not -1.0 <= self.theta_spearman_greater_than < 1.0:
            raise ValueError("theta_spearman_greater_than must be in [-1, 1)")
        if self.difficulty_rmse_less_than is not None and self.difficulty_rmse_less_than <= 0:
            raise ValueError("difficulty_rmse_less_than must be positive")

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> RecoveryGate:
        """Load either the original flat gate or the v2 required/diagnostic form."""

        raw_required = config.get("required", config)
        if not isinstance(raw_required, Mapping):
            raise ValueError("gate.required must be a mapping")
        theta_threshold = raw_required.get("theta_spearman_greater_than")
        if not isinstance(theta_threshold, int | float):
            raise ValueError("gate requires theta_spearman_greater_than")
        difficulty_threshold = raw_required.get("difficulty_rmse_less_than")
        if difficulty_threshold is not None and not isinstance(
            difficulty_threshold, int | float
        ):
            raise ValueError("difficulty_rmse_less_than must be numeric")
        require_convergence = raw_required.get("converged", False)
        if not isinstance(require_convergence, bool):
            raise ValueError("converged must be Boolean")
        return cls(
            theta_spearman_greater_than=float(theta_threshold),
            difficulty_rmse_less_than=(
                float(difficulty_threshold) if difficulty_threshold is not None else None
            ),
            require_convergence=require_convergence,
        )

    def evaluate(self, metrics: RecoveryMetrics) -> tuple[bool, dict[str, bool]]:
        """Return the overall decision and named hard-gate checks."""

        checks = {
            "theta_spearman": metrics.theta_spearman > self.theta_spearman_greater_than,
        }
        if self.difficulty_rmse_less_than is not None:
            checks["difficulty_rmse"] = (
                metrics.difficulty_rmse < self.difficulty_rmse_less_than
            )
        if self.require_convergence:
            checks["converged"] = metrics.converged
        return all(checks.values()), checks
