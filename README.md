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
test passed from a clean revision in 52.44 seconds.

The original full 468-item, 20-replicate M1-B-v2 run later completed from its clean preregistration
revision and also passed all six checks. The lightweight test remains the fast regression gate.

A user-authorized reduced M2–M5 prototype is also complete: 20-seed fixed-budget unit-cost
replay on both matrices, five policies including a rank-aware ablation, and a cached/capped fake
provider. It is exploratory and does not satisfy the original fixed-confidence, 200-seed,
live-provider, dollar-cost, or 5x gates.
See [REPORT.md](REPORT.md) for exact metrics, provenance, and current gate status.

## Ranking models

Use the anchored IRT ranking command for model-level results. It ranks by EAP posterior mean for
display, but treats pairwise posterior probabilities and confidence tiers as authoritative. Models
in the same tier are not claimed to be distinguishable.

```bash
# Preferred: calibrate on an independent historical matrix with matching item IDs.
uv run python scripts/rank_models.py \
  --matrix data/processed/swebench_verified/responses.parquet \
  --calibration-matrix path/to/historical_responses.parquet \
  --save-item-bank artifacts/swebench-item-bank.json \
  --output artifacts/swebench-ranking.json \
  --epsilon 0.1 --confidence 0.95

# Rank later models on exactly the same latent scale.
uv run python scripts/rank_models.py \
  --matrix path/to/new_responses.parquet \
  --item-bank artifacts/swebench-item-bank.json \
  --output artifacts/new-ranking.json \
  --epsilon 0.1 --confidence 0.95
```

Omitting both calibration options performs a joint descriptive calibration on the target matrix.
That is useful for describing a complete historical matrix, but a saved independent item bank is
required for fair longitudinal comparisons. SWE-bench rows must be described as model-plus-agent
systems. See [RANKING.md](docs/RANKING.md) for interpretation details.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy src scripts tests
```

Tests do not access the network. Public source acquisition and transformation are explicit scripts;
processed matrices include manifests with exact commits and checksums.
