#!/usr/bin/env python3
"""Run the scale-linked, like-for-like M1-B-v2 reference agreement gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from irt_rank.data.m0 import sha256_file
from irt_rank.data.matrix import DenseResponseMatrix, load_long_response_matrix
from irt_rank.irt.agreement import (
    ParameterAgreementMetrics,
    ScaleLink,
    estimable_item_mask,
    link_reference_scale,
    parameter_agreement_metrics,
    sample_estimable_item_indices,
)
from irt_rank.irt.mml import MMLConfig, MMLResult, fit_mml
from irt_rank.irt.model import IRTModel, ItemParameters
from irt_rank.irt.reference import fit_girth_2pl_parameters
from irt_rank.irt.synthetic import generate_responses

FloatArray = npt.NDArray[np.float64]
UIntArray = npt.NDArray[np.uint8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/m1_reference_agreement_v2.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            ".agent/ml/adaptive-irt-ranking/artifacts/m1-reference-agreement-v2-result.json"
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


def _write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _theta_grid(config: dict[str, Any]) -> FloatArray:
    return np.linspace(
        float(config["theta_grid_min"]),
        float(config["theta_grid_max"]),
        int(config["theta_grid_points"]),
    )


def _local_config(config: dict[str, Any]) -> MMLConfig:
    return MMLConfig(
        quadrature_points=int(config["quadrature_points"]),
        max_iterations=int(config["max_iterations"]),
        tolerance=float(config["tolerance"]),
        minimum_discrimination=float(config["minimum_discrimination"]),
        maximum_discrimination=float(config["maximum_discrimination"]),
        parameter_bound=float(config["parameter_bound"]),
    )


def _fit_pair(
    responses: UIntArray,
    local_config: MMLConfig,
    reference_config: dict[str, Any],
    theta_grid: FloatArray,
) -> tuple[MMLResult, ScaleLink, ParameterAgreementMetrics]:
    local = fit_mml(responses, IRTModel.TWO_PL, config=local_config)
    reference = fit_girth_2pl_parameters(
        responses,
        quadrature_bounds=tuple(reference_config["quadrature_bounds"]),
        quadrature_points=int(reference_config["quadrature_points"]),
        max_iterations=int(reference_config["max_iterations"]),
    )
    linked = link_reference_scale(local.parameters, reference, theta_grid=theta_grid)
    metrics = parameter_agreement_metrics(
        local.parameters,
        linked.parameters,
        theta_grid=theta_grid,
    )
    return local, linked, metrics


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


def _subset_parameters(parameters: ItemParameters, mask: npt.NDArray[np.bool_]) -> ItemParameters:
    return ItemParameters(
        parameters.discrimination[mask],
        parameters.difficulty[mask],
        parameters.guessing[mask],
    )


def _parameter_payload(parameters: ItemParameters) -> dict[str, list[float]]:
    return {
        "discrimination": parameters.discrimination.tolist(),
        "difficulty": parameters.difficulty.tolist(),
    }


def _invalid_result(
    *,
    config: dict[str, Any],
    config_bytes: bytes,
    provenance: dict[str, str | bool],
    matrix: DenseResponseMatrix,
    reason: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "experiment": config["experiment"],
        "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "provenance": provenance,
        "matrix": {"models": len(matrix.model_ids), "items": len(matrix.item_ids)},
        "validity": "invalid",
        "invalid_reason": reason,
        "metrics": metrics,
        "gate_pass": None,
    }


def main() -> int:
    args = parse_args()
    config_bytes = args.config.read_bytes()
    config = cast(dict[str, Any], json.loads(config_bytes))
    matrix_config = cast(dict[str, Any], config["matrix"])
    sample_config = cast(dict[str, Any], config["item_sample"])
    local_config_raw = cast(dict[str, Any], config["local_estimator"])
    reference_config = cast(dict[str, Any], config["reference_estimator"])
    linking_config = cast(dict[str, Any], config["linking"])
    monte_carlo = cast(dict[str, Any], config["monte_carlo"])
    gate = cast(dict[str, Any], config["gate"])
    provenance = git_provenance()

    if version("girth") != reference_config["version"]:
        raise RuntimeError("installed girth version does not match preregistration")
    matrix_path = Path(matrix_config["path"])
    matrix = load_long_response_matrix(matrix_path)
    _validate_matrix(matrix, matrix_config, matrix_path)
    real_mask = estimable_item_mask(matrix.responses)
    estimable_fraction = float(real_mask.mean())
    if sample_config["method"] != "sha256_smallest":
        raise ValueError("unsupported item sampling method")
    sampled_indices = sample_estimable_item_indices(
        matrix.item_ids,
        real_mask,
        sample_size=int(sample_config["items"]),
        salt=str(sample_config["salt"]),
    )
    real_responses = matrix.responses[:, sampled_indices]
    item_ids = [matrix.item_ids[index] for index in sampled_indices]
    local_config = _local_config(local_config_raw)
    theta_grid = _theta_grid(linking_config)
    started = time.perf_counter()

    if estimable_fraction < float(gate["minimum_estimable_item_fraction"]):
        result = _invalid_result(
            config=config,
            config_bytes=config_bytes,
            provenance=provenance,
            matrix=matrix,
            reason="too few items contain both response classes",
            metrics={"estimable_item_fraction": estimable_fraction},
        )
        _write_result(args.output, result)
        return 3

    local_real, linked_real, observed = _fit_pair(
        real_responses, local_config, reference_config, theta_grid
    )
    print(
        json.dumps(
            {
                "stage": "real_fit",
                "sampled_items": len(item_ids),
                "local_iterations": local_real.iterations,
                "local_converged": local_real.converged,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not local_real.converged:
        result = _invalid_result(
            config=config,
            config_bytes=config_bytes,
            provenance=provenance,
            matrix=matrix,
            reason="local 2PL estimator did not converge on the real matrix",
            metrics={
                "estimable_item_fraction": estimable_fraction,
                "local_real_iterations": local_real.iterations,
            },
        )
        _write_result(args.output, result)
        return 3

    replicates = int(monte_carlo["replicates"])
    seed = int(monte_carlo["seed"])
    bootstrap: list[dict[str, float]] = []
    failed_replicates: list[int] = []
    for replicate in range(replicates):
        rng = np.random.default_rng(seed + replicate)
        abilities = rng.normal(0.0, 1.0, size=int(monte_carlo["models_per_replicate"]))
        simulated = generate_responses(
            abilities,
            local_real.parameters,
            seed=seed + replicates + replicate,
        )
        replicate_mask = estimable_item_mask(simulated)
        try:
            local_bootstrap, _linked_bootstrap, metrics = _fit_pair(
                simulated[:, replicate_mask],
                local_config,
                reference_config,
                theta_grid,
            )
        except (RuntimeError, ValueError):
            failed_replicates.append(replicate)
            continue
        if not local_bootstrap.converged:
            failed_replicates.append(replicate)
            continue
        metric_payload = asdict(metrics)
        metric_payload["estimable_items"] = int(replicate_mask.sum())
        bootstrap.append(metric_payload)
        print(
            json.dumps(
                {
                    "stage": "monte_carlo",
                    "replicate": replicate + 1,
                    "replicates": replicates,
                    "valid_replicates": len(bootstrap),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    minimum_valid = int(monte_carlo["minimum_valid_replicates"])
    if len(bootstrap) < minimum_valid:
        result = _invalid_result(
            config=config,
            config_bytes=config_bytes,
            provenance=provenance,
            matrix=matrix,
            reason="too few valid Monte Carlo reference fits",
            metrics={
                "valid_replicates": len(bootstrap),
                "failed_replicates": failed_replicates,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        _write_result(args.output, result)
        return 3

    quantile = float(gate["observed_metrics_at_most_bootstrap_quantile"])
    metric_names = cast(list[str], gate["monte_carlo_metrics"])
    thresholds = {
        name: float(
            np.quantile(
                np.asarray([row[name] for row in bootstrap]),
                quantile,
                method="higher",
            )
        )
        for name in metric_names
    }
    observed_payload = asdict(observed)
    checks = {
        "estimable_item_fraction": estimable_fraction
        >= float(gate["minimum_estimable_item_fraction"]),
        "difficulty_spearman": observed.difficulty_spearman
        > float(gate["difficulty_spearman_greater_than"]),
        "discrimination_spearman": observed.discrimination_spearman
        > float(gate["discrimination_spearman_greater_than"]),
        **{
            f"{name}_within_monte_carlo": observed_payload[name] <= thresholds[name]
            for name in metric_names
        },
    }
    elapsed = time.perf_counter() - started
    result = {
        "schema_version": 2,
        "experiment": config["experiment"],
        "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "provenance": provenance,
        "matrix": {
            "models": len(matrix.model_ids),
            "items": len(matrix.item_ids),
            "estimable_items": int(real_mask.sum()),
            "estimable_item_fraction": estimable_fraction,
            "sampled_items": len(item_ids),
            "item_sample_method": sample_config["method"],
            "sha256": matrix_config["sha256"],
        },
        "reference": {
            "package": "girth",
            "version": version("girth"),
            "function": reference_config["function"],
        },
        "observed": {
            "metrics": observed_payload,
            "scale_link": {
                "slope": linked_real.slope,
                "intercept": linked_real.intercept,
                "tcc_rmse": linked_real.tcc_rmse,
            },
            "item_ids": item_ids,
            "local_parameters": _parameter_payload(local_real.parameters),
            "linked_reference_parameters": _parameter_payload(linked_real.parameters),
            "local_iterations": local_real.iterations,
        },
        "monte_carlo": {
            "valid_replicates": len(bootstrap),
            "failed_replicates": failed_replicates,
            "thresholds": thresholds,
            "replicate_metrics": bootstrap,
        },
        "gate_checks": checks,
        "elapsed_seconds": elapsed,
        "validity": "valid",
        "gate_pass": all(checks.values()),
    }
    _write_result(args.output, result)
    print(
        json.dumps(
            {
                "observed": observed_payload,
                "thresholds": thresholds,
                "valid_replicates": len(bootstrap),
                "gate_checks": checks,
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        )
    )
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
