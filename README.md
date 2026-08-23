# llm-testing

`irt-rank` is a research package for ranking language models with Item Response Theory while
minimizing evaluation cost. The primary objective is a ranking guarantee, not independent
per-model standard-error minimization.

The project is milestone-gated. M0 builds two pinned offline-replay matrices:

- MMLU per-item correctness from the tinyBenchmarks Open LLM Leaderboard artifact.
- SWE-bench Verified per-instance resolution outcomes from public leaderboard submissions.

SWE-bench rows represent model-plus-agent systems. They must not be interpreted as isolated base
model measurements.

## Current result

M0 passed. The original M1-A synthetic gate failed on difficulty RMSE and remains recorded. The
user subsequently revised M1-A around ranking recovery: convergence and theta Spearman are hard
checks, while difficulty RMSE is diagnostic. M1-A-v2 passed. The corrected, lightweight M1-B-v2
test passed from a clean revision in 52.44 seconds; M2 is now unblocked but remains unrun.
See [REPORT.md](REPORT.md) for exact metrics, provenance, and current gate status.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy src scripts tests
```

Tests do not access the network. Public source acquisition and transformation are explicit scripts;
processed matrices include manifests with exact commits and checksums.
