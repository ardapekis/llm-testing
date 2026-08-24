# llm-testing

`irt-rank` is a research package for ranking language models while minimizing evaluation cost.
It now contains two distinct paths: anchored IRT ranking and a non-IRT, prediction-corrected
randomized evaluation design. The primary objective is a defensible ranking decision, not
independent per-model standard-error minimization.

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
For a presentation-ready overview, open [REPORT.html](REPORT.html) in a browser.

## Non-IRT cheap evaluation

The non-IRT prototype learns a low-rank response surrogate from historical models, adapts it with a
small stratified sentinel set, samples additional items with a randomized exploration floor, and
uses logged inclusion probabilities to correct prediction error. The correction—not the
surrogate—makes the score target valid under model misspecification.

In 10-seed held-out replay, active corrected evaluation reached median Kendall tau 0.8085 on MMLU
at 1% of items and 0.9404 at 10%. On the harder chronological SWE-bench holdout, it reached 0.6955
at roughly 46 of 500 items; active selection did not consistently beat the raw stratified baseline,
so no universal superiority claim is made. See
[NON_IRT_EVALUATION.md](docs/NON_IRT_EVALUATION.md) for the design, exact results, and limitations.

```bash
uv run python scripts/run_non_irt_efficiency.py
```

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
