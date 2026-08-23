# Cost-Minimal Model Ranking via Adaptive IRT

## Status

**STOP-BEST at M1-B.** M0 and the user-revised M1-A-v2 gate passed. M1-B produced near-perfect ordinal agreement but failed its absolute-scale Monte Carlo tolerance and bootstrap-convergence checks. The original M1-A-v1 failure also remains recorded. M2–M5 were not run.

The ordered milestone contract still applies. The revised gate changes only the decision rule and preserves the original response data, estimator, seed, and failed result. The repository currently contains no replay cost curves, adaptive-policy comparison, live run, or claimed cost reduction.

## Milestone outcomes

| Milestone | Status | Evidence |
|---|---|---|
| M0 — response matrices | PASS | MMLU: 395 models × 14,042 items, 100% dense. SWE-bench Verified: 134 systems × 500 items, 100% dense. Both artifacts are Boolean, pair-unique, checksummed, and carry source manifests. |
| M1-A-v1 — original synthetic recovery | **FAIL** | Difficulty RMSE 0.2138646913 (required <0.15); theta Spearman 0.9917795029 (required >0.98). |
| M1-A-v2 — ranking recovery | **PASS** | Clean revision `496012a`: converged; theta Spearman 0.9917795029 >0.98. Difficulty RMSE remains diagnostic. |
| M1-B — reference agreement | **FAIL** | b Spearman 0.999944 passes; b RMSE 0.66304 exceeds bootstrap q95 0.60888; 99/100 local bootstrap fits converged. |
| M2 — replay baselines | NOT RUN | Blocked by the M1-B failure. |
| M3 — adaptive ranking | NOT RUN | Blocked until M2 passes. |
| M4 — live adapter | NOT RUN | Blocked by milestone order; no paid API use is authorized. |
| M5 — complete report and ablations | NOT RUN | Required upstream evidence does not exist. |

## M0 data evidence

The checked-in long-format matrices use the exact schema `(model_id: string, item_id: string, correct: bool)`.

| Matrix | Entities | Items | Observations | Density | Derived SHA-256 |
|---|---:|---:|---:|---:|---|
| MMLU / Open LLM Leaderboard | 395 models | 14,042 | 5,546,590 | 1.0 | `eebe5be98848d595f005e76f35bab9eb795cddac9b74e6feec3dfd1df99283e4` |
| SWE-bench Verified | 134 submitted systems | 500 | 67,000 | 1.0 | `a1cd623aad7f7e1405db19ee4878bfe532466e71fc29e24908e018b5dadb1d80` |

The MMLU source is tinyBenchmarks commit `e9a8b1031b0340571beb6c9ca3a27891be09a8fd`. The SWE-bench source is the experiments repository at commit `1faa91cade0562ba62b66c1c99e71f7b72d96f13`. SWE rows are model-plus-agent systems, not isolated base models. The pinned experiments checkout has no root license assertion; its derived manifest records `NOASSERTION` rather than inferring one.

## M1 implementation and failed gate

Revision `121cec42581122016ec17343a55a0ee82917de29` implements:

- 1PL, 2PL, and 3PL response probabilities and item information;
- marginal maximum-likelihood EM item calibration;
- MAP and EAP ability estimation under a standard-normal prior;
- an optional NumPyro/NUTS full-Bayes path;
- deterministic synthetic generation and a frozen recovery runner.

The acceptance run began with a clean worktree. Its configuration hash is `39039a9b150257058ea86373adc469aa45d26d2e0ba867030534edfb919c027a`: 50 models, 1,000 items, fixed discrimination 2.5, seed 20260823, 41 quadrature points, and no post-observation tuning.

| Metric | Threshold | Observed | Outcome |
|---|---:|---:|---|
| Difficulty RMSE | <0.15 | 0.2138646913 | FAIL |
| Theta Spearman | >0.98 | 0.9917795029 | PASS |
| EM convergence | required | 10 iterations | PASS |

Two same-data diagnostics bound the interpretation:

- estimating difficulty with the true simulated abilities still gives RMSE 0.1757007915;
- the Cramér–Rao RMS scale is 0.1668800354.

Both exceed 0.15. This supports the narrow conclusion that the frozen 50-response-per-item experiment does not establish the requested difficulty recovery. The later M1-B run separately tested reference agreement and failed its absolute-scale criterion. The synthetic result does not justify changing the seed, difficulty distribution, or discrimination after observing the gate.

### M1-B reference comparison

The full 134x500 SWE-bench Verified matrix was calibrated with the local fixed-discrimination 1PL MML-EM implementation and `girth 0.8.0` `rasch_mml`. The real local fit converged at iteration 168. The uncertainty tolerance came from 100 preregistered paired parametric-bootstrap comparisons.

| Check | Required | Observed | Outcome |
|---|---:|---:|---|
| Difficulty Spearman | >0.99 | 0.9999439914 | PASS |
| Difficulty RMSE | <= bootstrap q95 | 0.6630381298 vs 0.6088838866 | FAIL |
| Local bootstrap convergence | 100/100 | 99/100 | FAIL |

Bootstrap RMSE ranged from 0.5187877005 to 0.6299211656, with median 0.5547452898. The ordinal match is strong, but the absolute item parameters did not agree within the preregistered Monte Carlo tolerance.

## Reproducibility

```bash
uv sync --all-groups --all-extras
uv run python scripts/verify_m0_data.py
uv run python scripts/run_m1_recovery.py  # expected exit code: 2
uv run python scripts/run_m1_recovery.py --config configs/m1_recovery_rank_v2.json --output .agent/ml/adaptive-irt-ranking/artifacts/m1-recovery-rank-v2-result.json
uv run python scripts/run_m1_reference_agreement.py  # expected exit code: 2
uv run pytest
uv run ruff check .
uv run mypy src scripts tests
```

The recorded engineering verification passed with 27 tests, Ruff, and strict mypy over 30 source files. Tests make no network calls. M1 result artifacts carry the implementation Git SHA and config hash.

## Cost curves, ablations, and guarantees

No cost curves or ablations are reported. The replay harness, cost models, baseline and rank-aware policies, anytime-valid confidence sequences, item-parameter uncertainty propagation through replay, and live adapter belong to M2–M4 and remain blocked by the M1-B failure. Consequently:

- no cost reduction at matched ranking fidelity has been demonstrated;
- no fixed-confidence epsilon-inversion guarantee has been calibrated;
- no fixed-budget Kendall-tau comparison exists;
- no claim about rank-aware selection outperforming CAT or random is supported.

## Assumptions and invalidation boundaries

The available evidence is additionally bounded by these conditions:

- the full-matrix ability ranking would only be a replay proxy, not ground truth;
- SWE-bench submissions confound model and agent harness effects;
- the SWE experiment artifact's license remains `NOASSERTION` in the manifest;
- the M1 diagnostic is specific to the frozen generator, sample size, and seed;
- near-perfect ordinal item agreement does not establish absolute-scale parameter agreement.

## Stop condition

The user-authorized M1-A-v2 run passed, but the subsequent M1-B gate failed validly at clean revision `2e93db3`. The ordered milestone contract therefore stops before M2. Continuing requires an explicit M1-B contract revision; E-M1-004 must remain preserved as negative evidence.
