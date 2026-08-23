# Response matrices

Each matrix directory contains `responses.parquet` in sparse-compatible long format:
`(model_id, item_id, correct)`, plus `manifest.json` with pinned provenance, licence status,
checksums, transformation semantics, and structural statistics.

`mmlu_openllm` is sourced from the tinyBenchmarks artifact at a fixed commit and carries that
repository's MIT licence declaration.

`swebench_verified` is derived from public SWE-bench leaderboard result artifacts. The pinned
`SWE-bench/experiments` repository has no root licence declaration, so the artifact manifest records
`NOASSERTION`; it does not infer a licence from the upstream MIT-licensed SWE-bench code. Its rows
are model-plus-agent submissions, not isolated base models.

Rebuild from pinned checkouts:

```bash
uv run python scripts/build_m0_data.py \
  --tinybench-root data/raw/tinyBenchmarks \
  --swebench-root data/raw/swebench-experiments
uv run python scripts/verify_m0_data.py
```
