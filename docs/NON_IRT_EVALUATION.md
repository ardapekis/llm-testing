# Cheap model evaluation without IRT

## Decision

The recommended non-IRT path is a **prediction-assisted randomized evaluation**:

1. Learn a low-rank response surrogate from completed historical models.
2. Give each new model a small, content-stratified sentinel set.
3. Use the sentinel responses to predict the remaining item outcomes.
4. Sample additional items with a mixture of predicted error risk and mandatory random exploration.
5. Correct the prediction with the logged item-inclusion probabilities.
6. Rank by corrected scores, but resolve a model pair only when its corrected gap interval excludes
   the practical indifference region.

IRT is no longer part of either the predictor or the estimator. The surrogate can be replaced by
matrix completion, k-nearest models, embeddings, a learned acquisition policy, or an ensemble
without changing the estimand.

## Why the correction matters

Active selection alone changes the item distribution and generally biases a raw sample average.
For item outcomes `y_i`, surrogate predictions `q_i`, and logged inclusion probabilities `pi_i`,
the implemented estimator is:

```text
score_hat = mean(q_i) + (1 / N) * sum over sampled i of (y_i - q_i) / pi_i
```

This is a finite-population, Horvitz–Thompson residual correction. Conditional on the sentinel
responses and sampling design, every remaining item retains positive probability of selection.
The prediction can therefore be wrong without changing the target score; a better prediction only
reduces correction variance. Pairwise model gaps use the same construction on `y_Ai - y_Bi` and
benefit from evaluating models on a shared item set.

The implementation lives in [`src/irt_rank/efficient.py`](../src/irt_rank/efficient.py). It includes:

- a ridge-fitted SVD response surrogate;
- content-stratified sentinel selection;
- randomized active acquisition with an exploration floor and optional item costs;
- logged Bernoulli inclusion probabilities;
- prediction-corrected score and paired-gap estimates with approximate intervals.

## Offline holdout diagnostic

The checked-in runner trains only on historical rows and then replays evaluation on held-out rows.
MMLU holds out complete creator groups; SWE-bench holds out the chronologically latest quarter of
systems. Ten seeds are evaluated at 1%, 2%, 5%, and 10% item budgets. All candidate models share
the realized item set for direct paired comparison.

Median results:

| Matrix | Budget | Items/model | Active corrected tau | Random corrected tau | Raw stratified tau | Active score MAE |
|---|---:|---:|---:|---:|---:|---:|
| MMLU | 1% | 144 | 0.8085 | 0.7205 | **0.8289** | 0.0296 |
| MMLU | 5% | 707 | **0.9157** | 0.8714 | 0.9109 | 0.0119 |
| MMLU | 10% | 1,397 | **0.9404** | 0.9153 | 0.9359 | 0.0083 |
| SWE-bench Verified | 1% | 4 | **0.4654** | 0.3814 | 0.3779 | 0.1301 |
| SWE-bench Verified | 5% | 24 | 0.6169 | 0.5737 | **0.6646** | 0.0600 |
| SWE-bench Verified | 10% | 46 | 0.6955 | 0.6127 | **0.7156** | 0.0481 |

The result supports the architecture, not a universal policy win. Active correction clearly beats
corrected random sampling in this replay and provides a design-based route to uncertainty. It does
not always beat the raw stratified mean on point-ranking metrics, especially on the small and
heterogeneous SWE-bench matrix. At very small SWE budgets, approximate interval coverage is also
unstable. A production gate must therefore tune acquisition on training data only and validate
coverage on substantially more held-out seeds.

The complete result is
[`non-irt-efficiency-result.json`](../.agent/ml/adaptive-irt-ranking/artifacts/non-irt-efficiency-result.json).

## Correct ranking semantics

A cheap evaluation should not force a total order when the observed sample cannot resolve one.
For a practical gap `epsilon`, declare model A above model B only when the lower confidence bound
for `score_A - score_B` exceeds `epsilon`. If the interval intersects `[-epsilon, epsilon]`, place
the models in an unresolved tier and buy more shared observations only when that decision matters.

This creates a decision-focused stopping rule:

- stop when every important adjacent pair is resolved;
- stop early when a model is clearly outside the promotion or deployment boundary;
- continue only on strata/items with high residual risk for unresolved pairs;
- always preserve a non-zero randomized exploration component and log propensities.

The current normal intervals are diagnostics, not anytime-valid guarantees. Production deployment
still needs a sealed calibration study (at least 200 seeds), propensity and coverage audits,
family/time-shift holdouts, and a preregistered epsilon/cost target.

## Reproduce

```bash
uv run python scripts/run_non_irt_efficiency.py \
  --output .agent/ml/adaptive-irt-ranking/artifacts/non-irt-efficiency-result.json
uv run pytest tests/test_efficient.py
```

## Research basis

This design follows the bias-correction argument in
[Active Testing](https://proceedings.mlr.press/v139/kossen21a.html), while allowing modern learned
selection policies such as [Active Evaluation Acquisition](https://arxiv.org/abs/2410.05952).
The anchor/subset evidence in [SubLIME](https://aclanthology.org/2025.acl-long.1477/) and the
item-centric cold-start results in [Scales++](https://arxiv.org/abs/2510.26384) motivate learning
cross-model response structure, but neither is treated as proof for these two local matrices.
