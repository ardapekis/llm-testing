from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from irt_rank.irt.ability import estimate_eap_posterior
from irt_rank.irt.mml import MMLConfig
from irt_rank.irt.model import IRTModel, ItemParameters
from irt_rank.irt.synthetic import generate_responses
from irt_rank.ranking import (
    ItemBank,
    RankingConfig,
    align_responses_to_item_bank,
    calibrate_item_bank,
    pairwise_superiority_probabilities,
    rank_models,
)


def _bank() -> ItemBank:
    items = 16
    return ItemBank(
        tuple(f"item-{index}" for index in range(items)),
        ItemParameters(
            discrimination=np.full(items, 1.5),
            difficulty=np.linspace(-1.5, 1.5, items),
            guessing=np.zeros(items),
        ),
        IRTModel.TWO_PL,
    )


def test_pairwise_probabilities_are_complementary_without_epsilon() -> None:
    bank = _bank()
    responses = np.asarray(
        [np.ones(16), np.r_[np.ones(8), np.zeros(8)], np.zeros(16)]
    )
    posterior = estimate_eap_posterior(responses, bank.parameters, quadrature_points=41)
    probabilities = pairwise_superiority_probabilities(posterior, epsilon=0.0)

    assert probabilities + probabilities.T == pytest.approx(np.ones((3, 3)))
    assert np.diag(probabilities) == pytest.approx(np.full(3, 0.5))
    assert probabilities[0, 2] > 0.99


def test_rank_models_reports_display_order_and_confidence_tiers() -> None:
    bank = _bank()
    responses = np.asarray(
        [np.ones(16), np.r_[np.ones(8), np.zeros(8)], np.zeros(16)]
    )
    result = rank_models(
        responses,
        ("strong", "middle", "weak"),
        bank,
        config=RankingConfig(superiority_probability=0.95, quadrature_points=41),
    )

    assert [model.model_id for model in result.models] == ["strong", "middle", "weak"]
    assert result.probability_better("strong", "weak") > 0.99
    assert result.confidently_better[0, 2]
    assert result.models[0].confidence_tier < result.models[-1].confidence_tier


def test_positive_epsilon_leaves_more_probability_unresolved() -> None:
    bank = _bank()
    responses = np.asarray([np.ones(16), np.r_[np.ones(12), np.zeros(4)]])
    posterior = estimate_eap_posterior(responses, bank.parameters, quadrature_points=41)
    zero = pairwise_superiority_probabilities(posterior, epsilon=0.0)
    practical = pairwise_superiority_probabilities(posterior, epsilon=0.5)
    assert practical[0, 1] < zero[0, 1]
    assert practical[0, 1] + practical[1, 0] < 1.0


def test_item_bank_round_trip_and_response_alignment(tmp_path: Path) -> None:
    bank = _bank()
    path = tmp_path / "bank.json"
    bank.write_json(path)
    loaded = ItemBank.read_json(path)
    assert loaded.item_ids == bank.item_ids
    assert loaded.parameters.difficulty == pytest.approx(bank.parameters.difficulty)

    reversed_ids = tuple(reversed(bank.item_ids))
    responses = np.arange(32, dtype=np.float64).reshape(2, 16)
    aligned = align_responses_to_item_bank(responses, reversed_ids, bank)
    assert aligned[:, 0].tolist() == responses[:, -1].tolist()


def test_calibration_excludes_constant_items_and_converges() -> None:
    rng = np.random.default_rng(31)
    abilities = rng.normal(size=120)
    source = _bank()
    responses = generate_responses(abilities, source.parameters, seed=32)
    responses = np.column_stack([np.zeros(120, dtype=np.uint8), responses])
    item_ids = ("constant", *source.item_ids)

    bank, fitted = calibrate_item_bank(
        responses,
        item_ids,
        model=IRTModel.ONE_PL,
        config=MMLConfig(max_iterations=100, tolerance=0.01),
    )

    assert fitted.converged
    assert "constant" not in bank.item_ids
    assert bank.parameters.items == source.parameters.items
