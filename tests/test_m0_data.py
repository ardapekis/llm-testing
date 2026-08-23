from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from irt_rank.data.m0 import audit_dense_arrays, sha256_file


def test_audit_dense_arrays_accepts_binary_dense_matrix() -> None:
    responses = np.asarray([[0, 1, 0], [1, 1, 0]], dtype=np.uint8)

    stats = audit_dense_arrays(["a", "b"], ["x", "y", "z"], responses)

    assert stats.models == 2
    assert stats.items == 3
    assert stats.observed == 6
    assert stats.density == 1.0
    assert stats.correct == 3
    assert stats.incorrect == 3
    assert not stats.passes_m0()


@pytest.mark.parametrize(
    ("models", "items", "responses", "message"),
    [
        (["a", "a"], ["x"], np.asarray([[0], [1]]), "model identifiers"),
        (["a"], ["x", "x"], np.asarray([[0, 1]]), "item identifiers"),
        (["a"], ["x"], np.asarray([[2]]), "binary"),
        (["a"], ["x"], np.asarray([[np.nan]]), "missing or non-finite"),
        (["a"], ["x"], np.asarray([0]), "two-dimensional"),
        (["a"], ["x"], np.asarray([[0, 1]]), "does not match"),
    ],
)
def test_audit_dense_arrays_rejects_invalid_input(
    models: list[str],
    items: list[str],
    responses: np.ndarray[tuple[int, ...], np.dtype[np.generic]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        audit_dense_arrays(models, items, responses)


def test_sha256_file(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"abc")

    assert sha256_file(artifact) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
