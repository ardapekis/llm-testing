#!/usr/bin/env python3
"""Build the two pinned M0 replay matrices from local source checkouts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pickle
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, ClassVar, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from irt_rank.data.m0 import MatrixStatistics, audit_dense_arrays, require_keys, sha256_file

TINYBENCH_COMMIT = "e9a8b1031b0340571beb6c9ca3a27891be09a8fd"
TINYBENCH_PICKLE_SHA256 = "34f44d6a819512ef74d00a95288d252fa679288a10ca167cd97fdbc3aae66437"
SWEBENCH_COMMIT = "1faa91cade0562ba62b66c1c99e71f7b72d96f13"
SWE_CANONICAL_SUBMISSION = "20240402_rag_gpt4"
MMLU_PREFIX = "harness_hendrycksTest_"
_NUMPY_FROMBUFFER = cast(
    Callable[..., object],
    importlib.import_module("numpy._core.numeric").__dict__["_frombuffer"],
)


class RestrictedNumpyUnpickler(pickle.Unpickler):
    """Load the audited NumPy-only tinyBenchmarks pickle.

    The pinned pickle contains exactly two globals. Refusing every other global prevents a changed
    source artifact from turning acquisition into arbitrary code execution.
    """

    _ALLOWED: ClassVar[dict[tuple[str, str], Callable[..., object]]] = {
        ("numpy.core.numeric", "_frombuffer"): _NUMPY_FROMBUFFER,
        ("numpy", "dtype"): np.dtype,
    }

    def find_class(self, module: str, name: str) -> Callable[..., object]:
        try:
            return self._ALLOWED[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}") from error


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_revision(root: Path, expected: str) -> None:
    actual = _git_revision(root)
    if actual != expected:
        raise ValueError(f"source checkout {root} is at {actual}, expected {expected}")


def sha256_tree(root: Path, paths: list[Path]) -> str:
    """Hash relative paths and contents in a stable order."""

    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _load_restricted_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as raw:
        unpickler = RestrictedNumpyUnpickler(cast(BinaryIO, raw))
        value = unpickler.load()
    if not isinstance(value, dict):
        raise TypeError(f"expected dictionary in {path}, got {type(value).__name__}")
    return cast(dict[str, Any], value)


def load_mmlu(root: Path) -> tuple[list[str], list[str], npt.NDArray[np.uint8]]:
    """Load MMLU as a model-by-item matrix from the pinned tinyBenchmarks artifact."""

    _require_revision(root, TINYBENCH_COMMIT)
    source = root / "tutorials" / "data" / "lb.pickle"
    actual_hash = sha256_file(source)
    if actual_hash != TINYBENCH_PICKLE_SHA256:
        raise ValueError(f"unexpected tinyBenchmarks pickle hash: {actual_hash}")

    payload = _load_restricted_pickle(source)
    require_keys(payload, {"data", "models"}, context="tinyBenchmarks payload")
    models = cast(list[str], payload["models"])
    data = cast(dict[str, dict[str, Any]], payload["data"])

    item_ids: list[str] = []
    blocks: list[npt.NDArray[np.generic]] = []
    for scenario in sorted(name for name in data if name.startswith(MMLU_PREFIX)):
        block_payload = data[scenario]
        require_keys(block_payload, {"correctness"}, context=scenario)
        block = np.asarray(block_payload["correctness"])
        if block.ndim != 2 or block.shape[1] != len(models):
            raise ValueError(f"unexpected correctness shape for {scenario}: {block.shape}")
        blocks.append(block)
        item_ids.extend(f"{scenario}:{index:05d}" for index in range(block.shape[0]))

    if not blocks:
        raise ValueError("no MMLU scenarios found")
    item_by_model = np.vstack(blocks)
    model_by_item = item_by_model.T.astype(np.uint8, copy=False)
    audit_dense_arrays(models, item_ids, model_by_item)
    return models, item_ids, model_by_item


def load_swebench(root: Path) -> tuple[list[str], list[str], npt.NDArray[np.uint8]]:
    """Load SWE-bench Verified submissions, treating non-resolved instances as incorrect."""

    _require_revision(root, SWEBENCH_COMMIT)
    verified = root / "evaluation" / "verified"
    canonical_path = verified / SWE_CANONICAL_SUBMISSION / "results" / "results.json"
    canonical_payload = cast(dict[str, Any], json.loads(canonical_path.read_text(encoding="utf-8")))
    require_keys(canonical_payload, {"generated"}, context=str(canonical_path))
    item_ids = sorted(cast(list[str], canonical_payload["generated"]))
    if len(item_ids) != 500 or len(set(item_ids)) != 500:
        raise ValueError("SWE canonical submission must contain exactly 500 unique generated IDs")

    result_paths = sorted(verified.glob("*/results/results.json"))
    model_ids: list[str] = []
    correctness = np.zeros((len(result_paths), len(item_ids)), dtype=np.uint8)
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}

    for row, result_path in enumerate(result_paths):
        submission_id = result_path.parents[1].name
        payload = cast(dict[str, Any], json.loads(result_path.read_text(encoding="utf-8")))
        require_keys(payload, {"resolved"}, context=str(result_path))
        resolved = cast(list[str], payload["resolved"])
        if len(resolved) != len(set(resolved)):
            raise ValueError(f"duplicate resolved IDs in {result_path}")
        outside = set(resolved).difference(item_index)
        if outside:
            raise ValueError(
                f"resolved IDs outside canonical set in {result_path}: {sorted(outside)}"
            )
        model_ids.append(submission_id)
        for item_id in resolved:
            correctness[row, item_index[item_id]] = 1

    audit_dense_arrays(model_ids, item_ids, correctness)
    return model_ids, item_ids, correctness


def write_long_parquet(
    path: Path,
    model_ids: list[str],
    item_ids: list[str],
    correctness: npt.NDArray[np.uint8],
    *,
    models_per_row_group: int = 16,
) -> None:
    """Write `(model_id, item_id, correct)` rows without one giant string allocation."""

    schema = pa.schema(
        [
            pa.field("model_id", pa.string(), nullable=False),
            pa.field("item_id", pa.string(), nullable=False),
            pa.field("correct", pa.bool_(), nullable=False),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(
        path,
        schema,
        compression="zstd",
        use_dictionary=["model_id", "item_id"],
        write_statistics=True,
    ) as writer:
        for start in range(0, len(model_ids), models_per_row_group):
            stop = min(start + models_per_row_group, len(model_ids))
            batch_models = model_ids[start:stop]
            table = pa.table(
                {
                    "model_id": np.repeat(np.asarray(batch_models), len(item_ids)),
                    "item_id": np.tile(np.asarray(item_ids), len(batch_models)),
                    "correct": correctness[start:stop].reshape(-1).astype(bool, copy=False),
                },
                schema=schema,
            )
            writer.write_table(table)


def write_manifest(
    path: Path,
    *,
    matrix_id: str,
    benchmark: str,
    entity_semantics: str,
    source_url: str,
    source_commit: str,
    source_artifacts: list[dict[str, str]],
    licence: dict[str, str],
    parquet_path: Path,
    statistics: MatrixStatistics,
    transform_notes: list[str],
) -> None:
    """Write a self-contained, deterministic-provenance manifest."""

    manifest = {
        "schema_version": 1,
        "matrix_id": matrix_id,
        "benchmark": benchmark,
        "entity_semantics": entity_semantics,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "url": source_url,
            "commit": source_commit,
            "artifacts": source_artifacts,
        },
        "licence": licence,
        "artifact": {
            "format": "parquet",
            "path": parquet_path.name,
            "sha256": sha256_file(parquet_path),
            "columns": ["model_id", "item_id", "correct"],
        },
        "statistics": statistics.as_dict(),
        "m0_gate_pass": statistics.passes_m0(),
        "transform": {"notes": transform_notes},
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_matrix(
    output_root: Path,
    *,
    matrix_id: str,
    benchmark: str,
    entity_semantics: str,
    source_url: str,
    source_commit: str,
    source_artifacts: list[dict[str, str]],
    licence: dict[str, str],
    transform_notes: list[str],
    loaded: tuple[list[str], list[str], npt.NDArray[np.uint8]],
) -> MatrixStatistics:
    """Validate, serialize, and manifest one matrix."""

    model_ids, item_ids, correctness = loaded
    statistics = audit_dense_arrays(model_ids, item_ids, correctness)
    matrix_dir = output_root / matrix_id
    parquet_path = matrix_dir / "responses.parquet"
    write_long_parquet(parquet_path, model_ids, item_ids, correctness)
    write_manifest(
        matrix_dir / "manifest.json",
        matrix_id=matrix_id,
        benchmark=benchmark,
        entity_semantics=entity_semantics,
        source_url=source_url,
        source_commit=source_commit,
        source_artifacts=source_artifacts,
        licence=licence,
        parquet_path=parquet_path,
        statistics=statistics,
        transform_notes=transform_notes,
    )
    return statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinybench-root", type=Path, required=True)
    parser.add_argument("--swebench-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    swe_result_paths = sorted(
        (args.swebench_root / "evaluation" / "verified").glob("*/results/results.json")
    )
    swe_source_hash = sha256_tree(args.swebench_root, swe_result_paths)
    mmlu = build_matrix(
        args.output_root,
        matrix_id="mmlu_openllm",
        benchmark="MMLU 5-shot",
        entity_semantics="language model evaluation run",
        source_url="https://github.com/felipemaiapolo/tinyBenchmarks",
        source_commit=TINYBENCH_COMMIT,
        source_artifacts=[
            {
                "path": "tutorials/data/lb.pickle",
                "sha256": TINYBENCH_PICKLE_SHA256,
            }
        ],
        licence={
            "expression": "MIT",
            "scope": "tinyBenchmarks repository and checked-in derived response artifact",
            "source": "https://github.com/felipemaiapolo/tinyBenchmarks/blob/main/LICENSE",
        },
        transform_notes=[
            "Selected all harness_hendrycksTest_* 5-shot scenarios.",
            "Transposed upstream item-by-model arrays to model-by-item before long conversion.",
        ],
        loaded=load_mmlu(args.tinybench_root),
    )
    swe = build_matrix(
        args.output_root,
        matrix_id="swebench_verified",
        benchmark="SWE-bench Verified",
        entity_semantics="model-plus-agent leaderboard submission",
        source_url="https://github.com/SWE-bench/experiments",
        source_commit=SWEBENCH_COMMIT,
        source_artifacts=[
            {
                "path": "evaluation/verified/*/results/results.json",
                "sha256_tree": swe_source_hash,
                "file_count": str(len(swe_result_paths)),
            },
        ],
        licence={
            "expression": "NOASSERTION",
            "scope": "leaderboard submission result artifacts",
            "source": "No root licence file at the pinned experiments commit",
            "upstream_benchmark_code": "MIT",
        },
        transform_notes=[
            f"Canonical 500 IDs taken from {SWE_CANONICAL_SUBMISSION}.generated.",
            "Resolved IDs map to 1; every other canonical ID maps to 0.",
            "Rows rank submitted systems, not isolated base models.",
        ],
        loaded=load_swebench(args.swebench_root),
    )
    print(json.dumps({"mmlu_openllm": mmlu.as_dict(), "swebench_verified": swe.as_dict()}))
    return 0 if mmlu.passes_m0() and swe.passes_m0() else 2


if __name__ == "__main__":
    raise SystemExit(main())
