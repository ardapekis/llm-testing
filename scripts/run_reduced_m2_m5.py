#!/usr/bin/env python3
"""Run the user-authorized reduced M2-M5 offline prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from irt_rank.data.matrix import load_long_response_matrix
from irt_rank.live import CachedCappedEvaluator, FakeProvider, RequestLimitExceeded
from irt_rank.replay import ReplayPoint, ReplayPolicy, points_as_dicts, run_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/reduced_m2_m5.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            ".agent/ml/adaptive-irt-ranking/artifacts/reduced-m2-m5-result.json"
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


def aggregate(points: list[ReplayPoint]) -> list[dict[str, str | float | int]]:
    groups: defaultdict[tuple[str, float], list[ReplayPoint]] = defaultdict(list)
    for point in points:
        groups[(point.policy, point.budget_fraction)].append(point)
    rows: list[dict[str, str | float | int]] = []
    for (policy, budget), values in sorted(groups.items()):
        tau = np.asarray([point.kendall_tau for point in values])
        inversion = np.asarray([point.inversion_rate for point in values])
        rows.append(
            {
                "policy": policy,
                "budget_fraction": budget,
                "seeds": len(values),
                "observations": values[0].observations,
                "kendall_tau_median": float(np.median(tau)),
                "kendall_tau_q25": float(np.quantile(tau, 0.25)),
                "kendall_tau_q75": float(np.quantile(tau, 0.75)),
                "inversion_rate_median": float(np.median(inversion)),
                "inversion_rate_q25": float(np.quantile(inversion, 0.25)),
                "inversion_rate_q75": float(np.quantile(inversion, 0.75)),
            }
        )
    return rows


def fake_provider_evidence(config: dict[str, Any]) -> dict[str, object]:
    models = [f"fake-model-{index}" for index in range(int(config["fake_models"]))]
    items = [f"fake-item-{index}" for index in range(int(config["fake_items"]))]
    outcomes = {
        (model, item): (model_index + item_index) % 2 == 0
        for model_index, model in enumerate(models)
        for item_index, item in enumerate(items)
    }
    provider = FakeProvider(outcomes)
    evaluator = CachedCappedEvaluator(provider, request_limit=int(config["request_limit"]))
    all_keys = list(outcomes)
    selected = [(model, items[0]) for model in models]
    selected.extend(
        key
        for key in all_keys
        if key not in selected
    )
    selected = selected[: int(config["request_limit"])]
    first_results = [evaluator.evaluate(*key) for key in selected]
    cached_results = [evaluator.evaluate(*key) for key in selected]
    cap_enforced = False
    try:
        evaluator.evaluate(*list(outcomes)[int(config["request_limit"])])
    except RequestLimitExceeded:
        cap_enforced = True
    return {
        "models": len(models),
        "items": len(items),
        "request_limit": int(config["request_limit"]),
        "provider_calls": provider.calls,
        "queried_models": len({model for model, _item in selected}),
        "cache_consistent": first_results == cached_results,
        "cap_enforced_before_provider_call": cap_enforced,
        "pass": first_results == cached_results
        and cap_enforced
        and provider.calls == int(config["request_limit"]),
    }


def main() -> int:
    args = parse_args()
    config_bytes = args.config.read_bytes()
    config = cast(dict[str, Any], json.loads(config_bytes))
    seed_config = cast(dict[str, Any], config["seeds"])
    seeds = range(int(seed_config["start"]), int(seed_config["start"]) + int(seed_config["count"]))
    budgets = tuple(float(value) for value in cast(list[float], config["budget_fractions"]))
    policies = [ReplayPolicy(value) for value in cast(list[str], config["policies"])]
    started = time.perf_counter()
    matrix_results: dict[str, object] = {}
    for matrix_config_raw in cast(list[dict[str, Any]], config["matrices"]):
        matrix_id = str(matrix_config_raw["id"])
        matrix = load_long_response_matrix(Path(matrix_config_raw["path"]))
        points: list[ReplayPoint] = []
        for policy in policies:
            for seed in seeds:
                points.extend(
                    run_replay(
                        matrix.responses,
                        policy,
                        seed=seed,
                        budget_fractions=budgets,
                    )
                )
            print(json.dumps({"matrix": matrix_id, "policy": policy.value}), flush=True)
        matrix_results[matrix_id] = {
            "models": len(matrix.model_ids),
            "items": len(matrix.item_ids),
            "curves": aggregate(points),
            "replicates": points_as_dicts(points),
        }
    m4 = fake_provider_evidence(cast(dict[str, Any], config["m4"]))
    result = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "scope": config["scope_revision"],
        "config": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "provenance": git_provenance(),
        "matrices": matrix_results,
        "m4_fake_provider": m4,
        "elapsed_seconds": time.perf_counter() - started,
        "validity": "valid" if bool(m4["pass"]) else "invalid",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"elapsed_seconds": result["elapsed_seconds"], "m4": m4}))
    return 0 if bool(m4["pass"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
