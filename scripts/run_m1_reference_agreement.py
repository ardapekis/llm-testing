#!/usr/bin/env python3
"""Run the preregistered M1-B real-matrix reference agreement gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np

from irt_rank.data.m0 import sha256_file
from irt_rank.data.matrix import DenseResponseMatrix, load_long_response_matrix
from irt_rank.irt.ability import estimate_eap
from irt_rank.irt.agreement import AgreementResult, evaluate_difficulty_agreement
from irt_rank.irt.mml import MMLConfig, MMLResult, fit_mml
from irt_rank.irt.model import IRTModel
from irt_rank.irt.reference import fit_girth_rasch_difficulty
from irt_rank.irt.synthetic import generate_responses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/m1_reference_agreement.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            ".agent/ml/adaptive-irt-ranking/artifacts/m1-reference-agreement-result.json"
        ),
    )
    return parser.parse_args()


def git_provenance() -> dict[str, str | bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return {"revision": revision, "dirty": dirty}


def _local_config(config: dict[str, Any], discrimination: float) -> MMLConfig:
    return MMLConfig(
        quadrature_points=int(config["quadrature_points"]),
        max_iterations=int(config["max_iterations"]),
        tolerance=float(config["tolerance"]),
        fixed_discrimination=discrimination,
        parameter_bound=float(config["parameter_bound"]),
    )


def _validate_matrix(
    matrix: DenseResponseMatrix, config: dict[str, Any], path: Path
) -> None:
    if sha256_file(path) != config["sha256"]:
        raise ValueError("real response matrix checksum does not match preregistration")
    expected_shape = (int(config["expected_models"]), int(config["expected_items"]))
    if matrix.responses.shape != expected_shape:
        raise ValueError(
            f"real response matrix shape {matrix.responses.shape} != {expected_shape}"
        )


def _fit_local_and_reference(
    responses: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
    local_config: MMLConfig,
    reference_config: dict[str, Any],
    discrimination: float,
) -> tuple[MMLResult, np.ndarray[tuple[int], np.dtype[np.float64]]]:
    local = fit_mml(responses, IRTModel.ONE_PL, config=local_config)
    reference = fit_girth_rasch_difficulty(
        responses,
        discrimination,
        quadrature_bounds=tuple(reference_config["quadrature_bounds"]),
        quadrature_points=int(reference_config["quadrature_points"]),
    )
    return local, reference


def _agreement_payload(result: AgreementResult) -> dict[str, object]:
    return {
        "difficulty_rmse": result.difficulty_rmse,
        "difficulty_spearman": result.difficulty_spearman,
        "bootstrap_rmse_quantile": result.bootstrap_rmse_quantile,
        "bootstrap_replicates": result.bootstrap_replicates,
        "gate_checks": result.gate_checks,
    }


def _write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_bytes = args.config.read_bytes()
    config = cast(dict[str, Any], json.loads(config_bytes))
    matrix_config = cast(dict[str, Any], config["matrix"])
    model_config = cast(dict[str, Any], config["model"])
    local_config_raw = cast(dict[str, Any], config["local_estimator"])
    reference_config = cast(dict[str, Any], config["reference_estimator"])
    monte_carlo = cast(dict[str, Any], config["monte_carlo"])
    gate = cast(dict[str, Any], config["gate"])

    if version("girth") != reference_config["version"]:
        raise RuntimeError("installed girth version does not match preregistration")
    matrix_path = Path(matrix_config["path"])
    matrix = load_long_response_matrix(matrix_path)
    _validate_matrix(matrix, matrix_config, matrix_path)
    discrimination = float(model_config["fixed_discrimination"])
    local_config = _local_config(local_config_raw, discrimination)

    provenance = git_provenance()
    started = time.perf_counter()
    local_real, reference_real = _fit_local_and_reference(
        matrix.responses, local_config, reference_config, discrimination
    )
    if not local_real.converged:
        invalid_result: dict[str, object] = {
            "schema_version": 1,
            "experiment": config["experiment"],
            "config": config,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "provenance": provenance,
            "matrix": {
                "models": len(matrix.model_ids),
                "items": len(matrix.item_ids),
                "sha256": matrix_config["sha256"],
            },
            "reference": {
                "package": "girth",
                "version": version("girth"),
                "function": reference_config["function"],
            },
            "validity": "invalid",
            "invalid_reason": "local estimator did not converge on the real matrix",
            "metrics": {
                "local_real_converged": False,
                "local_real_iterations": local_real.iterations,
                "elapsed_seconds": time.perf_counter() - started,
            },
            "gate_pass": None,
        }
        _write_result(args.output, invalid_result)
        print(json.dumps(invalid_result["metrics"], sort_keys=True))
        return 3
    real_abilities = estimate_eap(
        matrix.responses,
        local_real.parameters,
        quadrature_points=local_config.quadrature_points,
    )

    replicates = int(monte_carlo["replicates"])
    seed = int(monte_carlo["seed"])
    bootstrap_rmse = np.empty(replicates, dtype=np.float64)
    bootstrap_converged = np.empty(replicates, dtype=np.bool_)
    for replicate in range(replicates):
        simulated = generate_responses(
            real_abilities.mean,
            local_real.parameters,
            seed=seed + replicate,
        )
        local_bootstrap, reference_bootstrap = _fit_local_and_reference(
            simulated, local_config, reference_config, discrimination
        )
        bootstrap_converged[replicate] = local_bootstrap.converged
        bootstrap_rmse[replicate] = np.sqrt(
            np.mean(np.square(local_bootstrap.parameters.difficulty - reference_bootstrap))
        )

    agreement = evaluate_difficulty_agreement(
        local_real.parameters.difficulty,
        reference_real,
        bootstrap_rmse,
        bootstrap_converged,
        minimum_spearman=float(gate["difficulty_spearman_greater_than"]),
        rmse_quantile=float(gate["observed_rmse_at_most_bootstrap_quantile"]),
        require_all_converged=bool(gate["require_all_bootstrap_local_fits_converged"]),
    )
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "provenance": provenance,
        "matrix": {
            "models": len(matrix.model_ids),
            "items": len(matrix.item_ids),
            "sha256": matrix_config["sha256"],
        },
        "reference": {
            "package": "girth",
            "version": version("girth"),
            "function": reference_config["function"],
        },
        "metrics": {
            **_agreement_payload(agreement),
            "local_real_converged": local_real.converged,
            "local_real_iterations": local_real.iterations,
            "bootstrap_converged": int(bootstrap_converged.sum()),
            "bootstrap_rmse": bootstrap_rmse.tolist(),
            "elapsed_seconds": elapsed,
        },
        "validity": "valid",
        "gate_pass": agreement.passed,
    }
    _write_result(args.output, result)
    print(json.dumps({**_agreement_payload(agreement), "elapsed_seconds": elapsed}, sort_keys=True))
    return 0 if agreement.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
