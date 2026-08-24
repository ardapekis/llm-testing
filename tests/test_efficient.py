import numpy as np
import pytest

from irt_rank.efficient import (
    SVDResponseSurrogate,
    infer_item_strata,
    prediction_corrected_gap,
    prediction_corrected_mean,
    randomized_active_batch,
    randomized_active_design,
    stratified_sentinel_indices,
)


def test_surrogate_predicts_from_partial_binary_responses() -> None:
    history = np.asarray(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    surrogate = SVDResponseSurrogate.fit(history, rank=2)
    observed = np.asarray([1.0, np.nan, 1.0, np.nan, 0.0, np.nan])

    prediction = surrogate.predict(observed)

    assert prediction.shape == (6,)
    assert np.all((prediction > 0) & (prediction < 1))
    assert np.all(surrogate.acquisition_scores(prediction) > 0)


def test_strata_inference_and_sentinel_cover_groups() -> None:
    item_ids = (
        "harness_math:0001",
        "harness_math:0002",
        "harness_history:0001",
        "astropy__astropy-123",
    )
    strata = infer_item_strata(item_ids)
    selected = stratified_sentinel_indices(strata, 3, seed=4)

    assert strata == ("harness_math", "harness_math", "harness_history", "astropy__astropy")
    assert {strata[index] for index in selected} == {
        "harness_math",
        "harness_history",
        "astropy__astropy",
    }


def test_active_design_logs_probabilities_and_preserves_exploration() -> None:
    scores = np.asarray([100.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    strata = np.asarray(["a", "a", "a", "b", "b", "b"])
    design = randomized_active_design(
        scores,
        strata,
        np.asarray([5]),
        expected_additional=2.5,
        exploration=0.2,
        seed=8,
    )

    assert design.sentinel[5]
    assert design.selected[5]
    assert design.inclusion_probability[5] == 1.0
    assert design.inclusion_probability.sum() == pytest.approx(3.5)
    assert np.all(design.inclusion_probability > 0)


def test_active_training_batch_has_exact_size_and_excludes_seen_items() -> None:
    selected = randomized_active_batch(
        [100.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ["a", "a", "a", "b", "b", "b"],
        [0, 5],
        size=3,
        exploration=0.2,
        seed=12,
    )

    assert selected.size == 3
    assert np.unique(selected).size == 3
    assert not {0, 5} & set(selected.tolist())


def test_prediction_correction_is_exact_at_full_census() -> None:
    truth = np.asarray([1.0, 0.0, 1.0, 1.0])
    wrong_prediction = np.asarray([0.1, 0.9, 0.1, 0.1])
    estimate = prediction_corrected_mean(truth, wrong_prediction, np.ones(4))

    assert estimate.mean == pytest.approx(0.75)
    assert estimate.standard_error == 0.0
    assert estimate.observations == 4


def test_prediction_correction_is_unbiased_with_wrong_surrogate() -> None:
    truth = np.asarray(([1.0] * 37) + ([0.0] * 63))
    prediction = np.full(100, 0.9)
    probability = np.full(100, 0.25)
    estimates: list[float] = []
    for seed in range(2_000):
        selected = np.random.default_rng(seed).random(100) < probability
        observed = np.full(100, np.nan)
        observed[selected] = truth[selected]
        estimates.append(prediction_corrected_mean(observed, prediction, probability).mean)

    assert np.mean(estimates) == pytest.approx(truth.mean(), abs=0.01)


def test_paired_gap_uses_shared_item_outcomes() -> None:
    left = np.asarray([1.0, 1.0, 0.0, 1.0])
    right = np.asarray([0.0, 1.0, 0.0, 0.0])
    estimate = prediction_corrected_gap(
        left,
        right,
        np.zeros(4),
        np.zeros(4),
        np.ones(4),
    )

    assert estimate.mean == pytest.approx(0.5)
    assert estimate.lower == pytest.approx(0.5)
    assert estimate.upper == pytest.approx(0.5)


def test_paired_gap_rejects_different_revealed_items() -> None:
    with pytest.raises(ValueError, match="same items"):
        prediction_corrected_gap(
            [1.0, np.nan],
            [np.nan, 0.0],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        )
