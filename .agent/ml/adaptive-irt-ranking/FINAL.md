# Final ML Engineering Report

## Outer decision

`STOP-BEST`

M0 and M1-A-v2 pass. M1-A-v1 remains negative evidence. M1-B fails: item ordering agrees with `girth`, but absolute difficulty RMSE exceeds the paired-bootstrap tolerance and only 99/100 bootstrap local fits converge. The ordered contract stops before M2.

## Objective and locked contract

- Objective: build and validate a cost-minimal adaptive IRT system for language-model benchmark ranking.
- Ordered completion rule: pass M0–M5 in order; stop and report at the first failed gate.
- Original M1-A-v1 rule: on a 50-model x 1,000-item synthetic experiment, difficulty RMSE <0.15 and theta Spearman >0.98.
- Revised M1-A-v2 rule: on the unchanged experiment, the estimator converges and theta Spearman >0.98; difficulty RMSE is diagnostic.
- Authorization: public downloads and local CPU work only; no paid API spend.

## Configuration sources, assumptions, and authorization

- User-confirmed decisions:
- Use `https://github.com/ardapekis/llm-testing` and the attached milestone contract.
- Repository-inferred fields:
- Upstream `main` was effectively empty; package targets Python >=3.11 with `uv`.
- Agent-proposed defaults retained:
- Frozen M1 configuration: seed 20260823, discrimination 2.5, uniform difficulty in [-1.25, 1.25], 41 quadrature points.
- Authorization boundaries:
- No paid API calls. No post-observation gate tuning. Do not proceed after failure.

## Delivered artifacts

- `REPORT.md`: public failed-gate report.
- `data/processed/mmlu_openllm/`: 395 x 14,042 dense MMLU response matrix and manifest.
- `data/processed/swebench_verified/`: 134 x 500 dense SWE-bench Verified system response matrix and manifest.
- `src/irt_rank/irt/`: 1PL/2PL/3PL, MML-EM, MAP/EAP, quadrature, synthetic generator, and optional NumPyro/NUTS path.
- `configs/m1_recovery.json` and `scripts/run_m1_recovery.py`: frozen acceptance runner.
- `artifacts/m1-recovery-result.json` and `artifacts/M1_FAILURE.md`: raw result and stop diagnosis.
- `artifacts/m1-reference-agreement-result.json` and `artifacts/M1B_FAILURE.md`: valid M1-B result and stop diagnosis.

## Baseline and final candidate

| Item | Provenance | Primary metric | Secondary metrics | Runtime / cost |
|---|---|---:|---|---|
| M0 processed matrices | E-M0-005; commit `8c2ba93` | Both structural gates pass | Pair uniqueness, schema, checksum, manifest | Local CPU; no paid cost |
| M1-A 1PL recovery | E-M1-001; clean revision `121cec4` | b RMSE 0.2138646913 (FAIL) | theta Spearman 0.9917795029; 10 iterations | 0.454 seconds; no paid cost |
| M1-B reference agreement | E-M1-004; clean revision `2e93db3` | b RMSE 0.66304 > q95 0.60888 (FAIL) | b Spearman 0.999944; 99/100 bootstrap fits converged | 147.16 seconds; no paid cost |

## Constraint outcomes

| ID | Final status | Evidence | Notes |
|---|---|---|---|
| M0-A/B/C | PASS | E-M0-005 | Two public, dense, binary, manifested matrices |
| M1-A-v1 | FAIL | E-M1-001 | b RMSE exceeds 0.15; preserved negative evidence |
| M1-A-v2 | PASS | E-M1-002 | Converged; theta Spearman 0.99178 |
| M1-B | FAIL | E-M1-004 | Absolute-scale and bootstrap-convergence checks fail |
| M2–M5 | UNRESOLVED | Not run | Blocked by M1-B |
| ENG | PASS | 27 tests; Ruff; strict mypy | No network tests |

## Weakest supported claim

- Claim: the local and `girth` 1PL calibrations agree on item ordering on the full SWE matrix, but not on absolute difficulty within the preregistered Monte Carlo tolerance.
- Tested scope: full pinned 134x500 SWE matrix, fixed a=1, local MML-EM, girth 0.8.0, and 100 paired bootstraps.
- Conditions removed: none from M1-B after the valid run.
- Stronger interpretations not established: absolute-scale reference agreement, adaptive cost savings, confidence calibration, and live integration.
- Counterexamples bounding further weakening: real b RMSE 0.66304 exceeds bootstrap q95 0.60888; one bootstrap local fit did not converge.

## Commands and reproducibility

```bash
uv run python scripts/verify_m0_data.py
uv run python scripts/run_m1_recovery.py
uv run python scripts/run_m1_recovery.py --config configs/m1_recovery_rank_v2.json --output .agent/ml/adaptive-irt-ranking/artifacts/m1-recovery-rank-v2-result.json
uv run python scripts/run_m1_reference_agreement.py
uv run pytest
uv run ruff check .
uv run mypy src scripts tests
```

The original M1-A-v1 and current M1-B commands exit 2 because their respective gates fail. Exact configs and provenance are in their result artifacts.

## Failures, regressions, and rejected paths

- H-M1-001 rejected: b RMSE 0.2138646913 is not below 0.15.
- H-M1-003 rejected: real b RMSE exceeds the paired-bootstrap q95 and one bootstrap local fit does not converge.
- No seed or hyperparameter search was used to replace the failed acceptance result.
- No engineering regressions were observed in the final verification suite.

## Remaining caveats and unresolved constraints

- M1-B fails and all M2–M5 requirements remain unresolved.
- The SWE matrix ranks submitted model-plus-agent systems and carries a `NOASSERTION` experiment-artifact license caveat.
- No empirical cost-reduction or fixed-confidence claim can be made.

## Highest-value next experiment

No experiment is authorized under the failed-gate rule. An explicit M1-B contract revision is required to resume.

Evidence confidence: 99/100 for the valid M1-B failure decision; no confidence is assigned to unrun downstream claims.

## Evidence map

| Evidence ID | Artifact / command | Conclusion |
|---|---|---|
| E-M0-005 | Processed manifests; `scripts/verify_m0_data.py` | M0 passes |
| E-M1-001 | `artifacts/m1-recovery-result.json`; `uv run python scripts/run_m1_recovery.py` | Original M1-A-v1 fails; result remains valid after contract revision |
| E-M1-002 | `artifacts/m1-recovery-rank-v2-result.json`; v2 config | Revised M1-A passes and unblocks M1-B |
| E-M1-003 | `artifacts/m1b-invalid-80-iterations.json` | Initial M1-B attempt invalid; numerical ceiling repaired |
| E-M1-004 | `artifacts/m1-reference-agreement-result.json`; `artifacts/M1B_FAILURE.md` | Valid M1-B failure stops M2 |
