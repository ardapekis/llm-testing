#!/usr/bin/env python3
"""Run the fixed M1 synthetic parameter-recovery gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import spearmanr

from irt_rank.irt.ability import estimate_eap
from irt_rank.irt.mml import MMLConfig, fit_mml
from irt_rank.irt.model import IRTModel
from irt_rank.irt.recovery import RecoveryGate, RecoveryMetrics
from irt_rank.irt.synthetic import recovery_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/m1_recovery.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".agent/ml/adaptive-irt-ranking/artifacts/m1-recovery-result.json"),
    )
    return parser.parse_args()


def git_provenance() -> dict[str, str | bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"revision": revision, "dirty": dirty}


def _oracle_score(
    difficulty: float,
    abilities: np.ndarray[tuple[int], np.dtype[np.float64]],
    item_responses: np.ndarray[tuple[int], np.dtype[np.uint8]],
    discrimination: float,
) -> float:
    fitted = expit(discrimination * (abilities - difficulty))
    return float(np.sum(fitted - item_responses))


def oracle_difficulty_diagnostic(
    responses: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
    abilities: np.ndarray[tuple[int], np.dtype[np.float64]],
    true_difficulties: np.ndarray[tuple[int], np.dtype[np.float64]],
    discrimination: float,
    *,
    bound: float = 6.0,
) -> tuple[float, float]:
    """Return oracle-theta RMSE and the itemwise Cramer-Rao RMS scale."""

    estimates = np.empty(responses.shape[1], dtype=np.float64)
    for item in range(responses.shape[1]):
        item_responses = responses[:, item]
        score = partial(
            _oracle_score,
            abilities=abilities,
            item_responses=item_responses,
            discrimination=discrimination,
        )

        if score(-bound) <= 0:
            estimates[item] = -bound
        elif score(bound) >= 0:
            estimates[item] = bound
        else:
            estimates[item] = brentq(score, -bound, bound)

    oracle_rmse = float(np.sqrt(np.mean(np.square(estimates - true_difficulties))))
    true_probs = expit(
        discrimination * (abilities[:, None] - true_difficulties[None, :])
    )
    item_information = np.sum(
        discrimination * discrimination * true_probs * (1.0 - true_probs),
        axis=0,
    )
    cramer_rao_rms = float(np.sqrt(np.mean(1.0 / item_information)))
    return oracle_rmse, cramer_rao_rms


def main() -> int:
    args = parse_args()
    config_bytes = args.config.read_bytes()
    config = cast(dict[str, Any], json.loads(config_bytes))
    data_config = cast(dict[str, Any], config["data"])
    estimator_config = cast(dict[str, Any], config["estimator"])
    gate_config = cast(dict[str, object], config["gate"])
    gate = RecoveryGate.from_config(gate_config)

    data = recovery_dataset(
        models=int(data_config["models"]),
        items=int(data_config["items"]),
        discrimination=float(data_config["discrimination"]),
        seed=int(data_config["seed"]),
    )
    mml_config = MMLConfig(
        quadrature_points=int(estimator_config["quadrature_points"]),
        max_iterations=int(estimator_config["max_iterations"]),
        tolerance=float(estimator_config["tolerance"]),
        fixed_discrimination=float(estimator_config["fixed_discrimination"]),
    )

    started = time.perf_counter()
    calibration = fit_mml(data.responses, IRTModel(estimator_config["model"]), config=mml_config)
    abilities = estimate_eap(
        data.responses,
        calibration.parameters,
        quadrature_points=mml_config.quadrature_points,
    )
    elapsed = time.perf_counter() - started
    difficulty_rmse = float(
        np.sqrt(np.mean(np.square(calibration.parameters.difficulty - data.parameters.difficulty)))
    )
    theta_spearman = float(spearmanr(abilities.mean, data.abilities).statistic)
    oracle_rmse, cramer_rao_rms = oracle_difficulty_diagnostic(
        data.responses,
        data.abilities,
        data.parameters.difficulty,
        float(data_config["discrimination"]),
    )
    passed, gate_checks = gate.evaluate(
        RecoveryMetrics(
            difficulty_rmse=difficulty_rmse,
            theta_spearman=theta_spearman,
            converged=calibration.converged,
        )
    )

    result = {
        "schema_version": 1,
        "experiment": config.get("experiment", "m1_synthetic_recovery"),
        "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "provenance": git_provenance(),
        "metrics": {
            "difficulty_rmse": difficulty_rmse,
            "oracle_theta_difficulty_rmse": oracle_rmse,
            "cramer_rao_rms_scale": cramer_rao_rms,
            "theta_spearman": theta_spearman,
            "converged": calibration.converged,
            "iterations": calibration.iterations,
            "marginal_log_likelihood": calibration.marginal_log_likelihood,
            "elapsed_seconds": elapsed,
        },
        "gate_checks": gate_checks,
        "gate_pass": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
