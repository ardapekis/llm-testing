#!/usr/bin/env python3
"""Calibrate or reuse an anchored IRT item bank and rank response-matrix entities."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from irt_rank.data.matrix import load_long_response_matrix
from irt_rank.irt.mml import MMLConfig
from irt_rank.irt.model import IRTModel
from irt_rank.ranking import (
    ItemBank,
    RankingConfig,
    align_responses_to_item_bank,
    calibrate_item_bank,
    rank_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--item-bank", type=Path)
    source.add_argument("--calibration-matrix", type=Path)
    parser.add_argument("--save-item-bank", type=Path)
    parser.add_argument("--model", choices=[model.value for model in IRTModel], default="2pl")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--quadrature-points", type=int, default=41)
    parser.add_argument("--max-iterations", type=int, default=160)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--entity-semantics",
        default="model or evaluated system represented by one response-matrix row",
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


def main() -> int:
    args = parse_args()
    target = load_long_response_matrix(args.matrix)
    calibration_payload: dict[str, object]
    if args.item_bank is not None:
        bank = ItemBank.read_json(args.item_bank)
        calibration_payload = {
            "method": "anchored_item_bank",
            "path": str(args.item_bank),
        }
    else:
        calibration = (
            load_long_response_matrix(args.calibration_matrix)
            if args.calibration_matrix is not None
            else target
        )
        mml_config = MMLConfig(
            quadrature_points=args.quadrature_points,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
        )
        bank, fitted = calibrate_item_bank(
            calibration.responses,
            calibration.item_ids,
            model=IRTModel(args.model),
            config=mml_config,
        )
        if not fitted.converged:
            raise RuntimeError(
                f"item calibration did not converge in {fitted.iterations} iterations"
            )
        calibration_payload = {
            "method": (
                "independent_anchor_matrix"
                if args.calibration_matrix is not None
                else "joint_descriptive_calibration"
            ),
            "matrix": str(args.calibration_matrix or args.matrix),
            "model": bank.model.value,
            "estimable_items": bank.parameters.items,
            "iterations": fitted.iterations,
            "marginal_log_likelihood": fitted.marginal_log_likelihood,
        }
    if args.save_item_bank is not None:
        bank.write_json(args.save_item_bank)
    aligned = align_responses_to_item_bank(target.responses, target.item_ids, bank)
    ranking_config = RankingConfig(
        epsilon=args.epsilon,
        superiority_probability=args.confidence,
        quadrature_points=args.quadrature_points,
    )
    result = rank_models(aligned, target.model_ids, bank, config=ranking_config)
    rows = [asdict(model) for model in result.models]
    adjacent: list[dict[str, object]] = []
    for higher, lower in zip(result.models[:-1], result.models[1:], strict=True):
        probability = result.probability_better(higher.model_id, lower.model_id)
        adjacent.append(
            {
                "higher": higher.model_id,
                "lower": lower.model_id,
                "probability_higher_exceeds_lower_by_epsilon": probability,
                "resolved": probability >= args.confidence,
            }
        )
    output = {
        "schema_version": 1,
        "matrix": str(args.matrix),
        "entity_semantics": args.entity_semantics,
        "models": len(target.model_ids),
        "item_bank_items": bank.parameters.items,
        "calibration": calibration_payload,
        "decision_rule": {
            "epsilon": args.epsilon,
            "superiority_probability": args.confidence,
            "display_order": "descending EAP posterior mean",
            "authoritative_order": "pairwise partial order and confidence tiers",
        },
        "provenance": git_provenance(),
        "ranking": rows,
        "confidence_tiers": result.tiers,
        "adjacent_comparisons": adjacent,
        "limitations": [
            "Item-parameter uncertainty is conditional on the fixed calibrated bank.",
            "Joint descriptive calibration is not an independent test of new models.",
            "A total display order does not resolve models in the same confidence "
            "tier.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
