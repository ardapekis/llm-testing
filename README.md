# llm-testing

`irt-rank` is a research package for ranking language models with Item Response Theory while
minimizing evaluation cost. The primary objective is a ranking guarantee, not independent
per-model standard-error minimization.

The project is milestone-gated. M0 builds two pinned offline-replay matrices:

- MMLU per-item correctness from the tinyBenchmarks Open LLM Leaderboard artifact.
- SWE-bench Verified per-instance resolution outcomes from public leaderboard submissions.

SWE-bench rows represent model-plus-agent systems. They must not be interpreted as isolated base
model measurements.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy src scripts tests
```

Tests do not access the network. Public source acquisition and transformation are explicit scripts;
processed matrices include manifests with exact commits and checksums.

