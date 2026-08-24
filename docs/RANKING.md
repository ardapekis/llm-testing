# Uncertainty-aware model ranking

## Ranking target

The implementation ranks one response-matrix row per evaluated entity on a fixed IRT latent scale.
MMLU rows are models. SWE-bench Verified rows are model-plus-agent systems and must not be described
as isolated base models.

The preferred workflow calibrates a 2PL item bank on an independent historical cohort, freezes that
bank, and estimates EAP ability posteriors for new entities. Reusing the bank anchors every run to
the same scale. A joint calibration on the target matrix is supported for descriptive historical
analysis but is not an independent evaluation of new models.

## Output semantics

`scripts/rank_models.py` emits:

- `posterior_mean` and `posterior_standard_deviation` for every entity;
- a deterministic display order by descending posterior mean;
- `P(theta_i > theta_j + epsilon)` for adjacent displayed entities;
- confidence tiers derived from the full pairwise partial order;
- an expected epsilon-rank that accounts for posterior uncertainty.

The total display order is convenient, not authoritative. Two entities are resolved only when the
configured pairwise probability reaches the confidence threshold. Entities in the same confidence
tier must be reported as unresolved even if their display ranks differ.

`epsilon` is the smallest latent-ability difference considered practically meaningful. With
`epsilon=0.1` and `confidence=0.95`, the statement “A ranks above B” means the fitted model assigns
at least 95% posterior probability that A's ability exceeds B's by more than 0.1.

## Information and uncertainty boundaries

- Item parameters are estimated only during calibration and remain fixed during anchored ranking.
- Missing evaluation responses are supported as `NaN` by the library API.
- Pairwise probabilities are integrated directly over the quadrature posterior rather than inferred
  from point estimates.
- Current intervals are conditional on the calibrated item bank; item-parameter uncertainty is not
  integrated into the ranking posterior.
- A single unidimensional ranking can hide subject-specific tradeoffs. Important MMLU uses should
  also publish domain-level rankings or a preregistered domain-weighted aggregate.
- Rankings apply only to the pinned benchmark distribution and evaluation harness.

## Library API

```python
from irt_rank.ranking import (
    RankingConfig,
    align_responses_to_item_bank,
    calibrate_item_bank,
    rank_models,
)

bank, calibration = calibrate_item_bank(
    historical_responses,
    historical_item_ids,
    model="2pl",
)
assert calibration.converged

aligned = align_responses_to_item_bank(new_responses, new_item_ids, bank)
result = rank_models(
    aligned,
    new_model_ids,
    bank,
    config=RankingConfig(epsilon=0.1, superiority_probability=0.95),
)

for tier_number, model_ids in enumerate(result.tiers, start=1):
    print(tier_number, model_ids)
```
