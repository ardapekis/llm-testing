#!/usr/bin/env python3
"""Replay non-IRT, prediction-corrected ranking on held-out models."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.stats import kendalltau

from irt_rank.data.matrix import DenseResponseMatrix, load_long_response_matrix
from irt_rank.efficient import (
    SVDResponseSurrogate,
    infer_item_strata,
    prediction_corrected_mean,
    randomized_active_batch,
    randomized_active_design,
    stratified_sentinel_indices,
)
from irt_rank.replay import inversion_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/non_irt_efficiency.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/non-irt-efficiency.json"),
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


def _creator_group(model_id: str) -> str:
    return model_id.split("__", maxsplit=1)[0]


def split_models(
    matrix: DenseResponseMatrix,
    strategy: str,
    holdout_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    model_count = len(matrix.model_ids)
    target_count = max(2, round(model_count * holdout_fraction))
    if strategy == "chronological_suffix":
        target = np.arange(model_count - target_count, model_count, dtype=np.int64)
    elif strategy == "creator_group_hash":
        groups = {_creator_group(model_id) for model_id in matrix.model_ids}
        ordered_groups = sorted(
            groups,
            key=lambda value: hashlib.sha256(value.encode()).digest(),
        )
        selected_groups: set[str] = set()
        selected_count = 0
        for group in ordered_groups:
            selected_groups.add(group)
            selected_count += sum(
                _creator_group(model_id) == group for model_id in matrix.model_ids
            )
            if selected_count >= target_count:
                break
        target = np.asarray(
            [
                index
                for index, model_id in enumerate(matrix.model_ids)
                if _creator_group(model_id) in selected_groups
            ],
            dtype=np.int64,
        )
    else:
        raise ValueError(f"unknown split strategy: {strategy}")
    target_set = set(target.tolist())
    history = np.asarray(
        [index for index in range(model_count) if index not in target_set],
        dtype=np.int64,
    )
    if history.size < 2 or target.size < 2:
        raise ValueError("split must contain at least two historical and target models")
    return history, target


def _summarize_estimates(
    truth: np.ndarray,
    estimates: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
) -> dict[str, float | None]:
    tau = float(kendalltau(truth, estimates).statistic)
    coverage = None
    if lower is not None and upper is not None:
        coverage = float(np.mean((lower <= truth) & (truth <= upper)))
    return {
        "kendall_tau": tau,
        "inversion_rate": inversion_rate(truth, estimates),
        "score_mae": float(np.mean(np.abs(truth - estimates))),
        "score_interval_coverage": coverage,
    }


def _corrected_vector(
    responses: np.ndarray,
    predictions: np.ndarray,
    selected: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    for response, prediction in zip(responses, predictions, strict=True):
        observed = np.full(response.size, np.nan)
        observed[selected] = response[selected]
        estimate = prediction_corrected_mean(observed, prediction, probabilities)
        means.append(estimate.mean)
        lowers.append(estimate.lower)
        uppers.append(estimate.upper)
    return np.asarray(means), np.asarray(lowers), np.asarray(uppers)


def _adjacent_pair_diagnostics(
    responses: np.ndarray,
    predictions: np.ndarray,
    selected: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    truth = responses.mean(axis=1)
    order = np.argsort(-truth, kind="stable")
    covered = 0
    resolved = 0
    correct = 0
    comparisons = 0
    for left, right in pairwise(order):
        observed_gap = responses[left, selected] - responses[right, selected]
        predicted_gap = predictions[left] - predictions[right]
        residual = observed_gap - predicted_gap[selected]
        mean = float(
            predicted_gap.mean()
            + np.sum(residual / probabilities[selected]) / responses.shape[1]
        )
        variance = np.sum(
            (1.0 - probabilities[selected])
            * np.square(residual)
            / np.square(probabilities[selected])
        ) / responses.shape[1] ** 2
        radius = 1.959963984540054 * float(np.sqrt(variance))
        true_gap = float(truth[left] - truth[right])
        covered += mean - radius <= true_gap <= mean + radius
        is_resolved = mean - radius > 0 or mean + radius < 0
        resolved += is_resolved
        correct += is_resolved and np.sign(mean) == np.sign(true_gap)
        comparisons += 1
    return {
        "adjacent_gap_coverage": covered / comparisons,
        "adjacent_resolved_fraction": resolved / comparisons,
        "adjacent_resolved_accuracy": correct / resolved if resolved else 1.0,
    }


def run_matrix(
    matrix: DenseResponseMatrix,
    matrix_config: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, object]:
    history_indices, target_indices = split_models(
        matrix,
        str(matrix_config["split"]),
        float(config["holdout_fraction"]),
    )
    surrogate = SVDResponseSurrogate.fit(
        matrix.responses[history_indices],
        rank=int(matrix_config["surrogate_rank"]),
    )
    target = matrix.responses[target_indices].astype(np.float64)
    truth = target.mean(axis=1)
    strata = np.asarray(infer_item_strata(matrix.item_ids))
    seed_config = cast(dict[str, Any], config["seeds"])
    seeds = range(
        int(seed_config["start"]),
        int(seed_config["start"]) + int(seed_config["count"]),
    )
    replicates: list[dict[str, object]] = []

    for budget_fraction in cast(list[float], config["budget_fractions"]):
        budget = max(2, round(target.shape[1] * float(budget_fraction)))
        sentinel_size = max(1, round(budget * float(config["sentinel_share_of_budget"])))
        sentinel_size = min(sentinel_size, budget - 1)
        for seed in seeds:
            sentinel_indices = stratified_sentinel_indices(strata, sentinel_size, seed=seed)
            predictions: list[np.ndarray] = []
            scores: list[np.ndarray] = []
            for response in target:
                observed = np.full(target.shape[1], np.nan)
                observed[sentinel_indices] = response[sentinel_indices]
                prediction = surrogate.predict(observed)
                predictions.append(prediction)
                scores.append(surrogate.acquisition_scores(prediction))
            prediction_matrix = np.stack(predictions)
            aggregate_score = np.mean(np.stack(scores), axis=0)

            active = randomized_active_design(
                aggregate_score,
                strata,
                sentinel_indices,
                expected_additional=budget - sentinel_size,
                exploration=float(config["active_exploration"]),
                seed=seed + 1_000_000,
            )
            random = randomized_active_design(
                np.ones(target.shape[1]),
                strata,
                sentinel_indices,
                expected_additional=budget - sentinel_size,
                exploration=1.0,
                seed=seed + 2_000_000,
            )

            sequential_sentinel_size = max(
                1,
                round(
                    budget * float(config["sequential_sentinel_share_of_budget"])
                ),
            )
            sequential_training_size = max(
                1,
                round(
                    budget * float(config["sequential_training_share_of_budget"])
                ),
            )
            if sequential_sentinel_size + sequential_training_size >= budget:
                sequential_training_size = budget - sequential_sentinel_size - 1
            sequential_sentinel = stratified_sentinel_indices(
                strata,
                sequential_sentinel_size,
                seed=seed + 3_000_000,
            )
            first_predictions: list[np.ndarray] = []
            first_scores: list[np.ndarray] = []
            for response in target:
                observed = np.full(target.shape[1], np.nan)
                observed[sequential_sentinel] = response[sequential_sentinel]
                prediction = surrogate.predict(observed)
                first_predictions.append(prediction)
                first_scores.append(surrogate.acquisition_scores(prediction))
            training_batch = randomized_active_batch(
                np.mean(np.stack(first_scores), axis=0),
                strata,
                sequential_sentinel,
                size=sequential_training_size,
                exploration=float(config["active_exploration"]),
                seed=seed + 4_000_000,
            )
            training_indices = np.concatenate((sequential_sentinel, training_batch))
            sequential_predictions: list[np.ndarray] = []
            sequential_scores: list[np.ndarray] = []
            for response in target:
                observed = np.full(target.shape[1], np.nan)
                observed[training_indices] = response[training_indices]
                prediction = surrogate.predict(observed)
                sequential_predictions.append(prediction)
                sequential_scores.append(surrogate.acquisition_scores(prediction))
            sequential_prediction_matrix = np.stack(sequential_predictions)
            sequential_audit = randomized_active_design(
                np.mean(np.stack(sequential_scores), axis=0),
                strata,
                training_indices,
                expected_additional=budget - training_indices.size,
                exploration=float(config["sequential_audit_exploration"]),
                seed=seed + 5_000_000,
            )

            active_mean, active_lower, active_upper = _corrected_vector(
                target,
                prediction_matrix,
                active.selected,
                active.inclusion_probability,
            )
            random_mean, random_lower, random_upper = _corrected_vector(
                target,
                prediction_matrix,
                random.selected,
                random.inclusion_probability,
            )
            sequential_mean, sequential_lower, sequential_upper = _corrected_vector(
                target,
                sequential_prediction_matrix,
                sequential_audit.selected,
                sequential_audit.inclusion_probability,
            )
            method_values: list[
                tuple[str, np.ndarray, np.ndarray | None, np.ndarray | None, int]
            ] = [
                (
                    "active_corrected",
                    active_mean,
                    active_lower,
                    active_upper,
                    int(active.selected.sum()),
                ),
                (
                    "stratified_random_corrected",
                    random_mean,
                    random_lower,
                    random_upper,
                    int(random.selected.sum()),
                ),
                (
                    "sequential_active_corrected",
                    sequential_mean,
                    sequential_lower,
                    sequential_upper,
                    int(sequential_audit.selected.sum()),
                ),
                (
                    "stratified_sample_mean",
                    target[:, random.selected].mean(axis=1),
                    None,
                    None,
                    int(random.selected.sum()),
                ),
                (
                    "surrogate_only",
                    prediction_matrix.mean(axis=1),
                    None,
                    None,
                    sentinel_size,
                ),
            ]
            for method, estimate, lower, upper, observations in method_values:
                row: dict[str, object] = {
                    "method": method,
                    "seed": seed,
                    "budget_fraction": float(budget_fraction),
                    "target_observations_per_model": budget,
                    "realized_observations_per_model": observations,
                    **_summarize_estimates(truth, estimate, lower, upper),
                }
                if method == "active_corrected":
                    row.update(
                        _adjacent_pair_diagnostics(
                            target,
                            prediction_matrix,
                            active.selected,
                            active.inclusion_probability,
                        )
                    )
                elif method == "stratified_random_corrected":
                    row.update(
                        _adjacent_pair_diagnostics(
                            target,
                            prediction_matrix,
                            random.selected,
                            random.inclusion_probability,
                        )
                    )
                elif method == "sequential_active_corrected":
                    row.update(
                        _adjacent_pair_diagnostics(
                            target,
                            sequential_prediction_matrix,
                            sequential_audit.selected,
                            sequential_audit.inclusion_probability,
                        )
                    )
                replicates.append(row)
        print(
            json.dumps(
                {
                    "matrix": matrix_config["id"],
                    "budget_fraction": budget_fraction,
                }
            ),
            flush=True,
        )

    groups: defaultdict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in replicates:
        groups[
            (
                str(row["method"]),
                float(cast(float, row["budget_fraction"])),
            )
        ].append(row)
    summary: list[dict[str, object]] = []
    numeric_metrics = (
        "realized_observations_per_model",
        "kendall_tau",
        "inversion_rate",
        "score_mae",
        "score_interval_coverage",
        "adjacent_gap_coverage",
        "adjacent_resolved_fraction",
        "adjacent_resolved_accuracy",
    )
    for (method, budget_fraction), rows in sorted(groups.items()):
        aggregate: dict[str, object] = {
            "method": method,
            "budget_fraction": budget_fraction,
            "replicates": len(rows),
        }
        for metric in numeric_metrics:
            values = [
                float(cast(float | int, row[metric]))
                for row in rows
                if row.get(metric) is not None
            ]
            if values:
                aggregate[f"{metric}_median"] = float(np.median(values))
                aggregate[f"{metric}_mean"] = float(np.mean(values))
        summary.append(aggregate)

    return {
        "historical_models": int(history_indices.size),
        "held_out_models": int(target_indices.size),
        "items": len(matrix.item_ids),
        "strata": int(np.unique(strata).size),
        "split": matrix_config["split"],
        "surrogate_rank": int(surrogate.basis.shape[0]),
        "summary": summary,
        "replicates": replicates,
    }


def main() -> int:
    args = parse_args()
    config_bytes = args.config.read_bytes()
    config = cast(dict[str, Any], json.loads(config_bytes))
    started = time.perf_counter()
    matrices: dict[str, object] = {}
    for matrix_config in cast(list[dict[str, Any]], config["matrices"]):
        matrix = load_long_response_matrix(Path(matrix_config["path"]))
        matrices[str(matrix_config["id"])] = run_matrix(matrix, matrix_config, config)
    result = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "scope": config["scope"],
        "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "provenance": git_provenance(),
        "matrices": matrices,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "elapsed_seconds": result["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
