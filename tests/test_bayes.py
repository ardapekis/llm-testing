from __future__ import annotations

import numpy as np

from irt_rank.irt.bayes import fit_full_bayes
from irt_rank.irt.model import IRTModel


def test_full_bayes_flag_runs_on_tiny_matrix() -> None:
    responses = np.asarray(
        [
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0, 1, 0],
        ]
    )

    result = fit_full_bayes(
        responses,
        IRTModel.ONE_PL,
        seed=3,
        warmup=5,
        samples=5,
        progress_bar=False,
    )

    assert result.samples["theta"].shape == (5, 4)
    assert result.samples["difficulty"].shape == (5, 3)
