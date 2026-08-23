# Final ML Engineering Report

## Outer decision

`REFINE (resumed)`

M0 and M1-A-v2 pass. M1-A-v1 remains negative evidence. E-M1-005 invalidates the M1-B-v1 acceptance inference while preserving its raw measurements. The user authorized a corrected M1-B-v2 validator; M2 remains blocked pending its result.

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
- `artifacts/m1-reference-agreement-result.json` and `artifacts/M1B_FAILURE.md`: preserved M1-B-v1 raw result and original stop diagnosis, whose acceptance inference E-M1-005 invalidates.

## Baseline and final candidate

| Item | Provenance | Primary metric | Secondary metrics | Runtime / cost |
|---|---|---:|---|---|
| M0 processed matrices | E-M0-005; commit `8c2ba93` | Both structural gates pass | Pair uniqueness, schema, checksum, manifest | Local CPU; no paid cost |
| M1-A 1PL recovery | E-M1-001; clean revision `121cec4` | b RMSE 0.2138646913 (FAIL) | theta Spearman 0.9917795029; 10 iterations | 0.454 seconds; no paid cost |
| M1-B-v1 reference agreement | E-M1-004 raw result; invalidated by E-M1-005 | b RMSE 0.66304 > q95 0.60888 | b Spearman 0.999944; 99/100 bootstrap fits converged | 147.16 seconds; no paid cost |

## Constraint outcomes

| ID | Final status | Evidence | Notes |
|---|---|---|---|
| M0-A/B/C | PASS | E-M0-005 | Two public, dense, binary, manifested matrices |
| M1-A-v1 | FAIL | E-M1-001 | b RMSE exceeds 0.15; preserved negative evidence |
| M1-A-v2 | PASS | E-M1-002 | Converged; theta Spearman 0.99178 |
| M1-B-v1 | INVALID | E-M1-005 | Non-equivalent estimators, unlinked scales, wrong bootstrap population |
| M1-B-v2 | UNRESOLVED | Revised config | Corrected validator pending |
| M2–M5 | UNRESOLVED | Not run | Blocked by M1-B-v2 |
| ENG | PASS | 27 tests; Ruff; strict mypy | No network tests |

## Weakest supported claim

- Claim: M1-B-v1 establishes ordinal similarity only; it cannot decide absolute-scale implementation agreement.
- Tested scope: estimator source, real-fit affine decomposition, bootstrap population, and a model-correct synthetic 2PL smoke comparison.
- Conditions removed: the unsupported assumptions that both v1 estimators solve the same objective, share raw coordinates, and use N(0,1) bootstrap abilities.
- Stronger interpretations not established: corrected real-matrix reference agreement, adaptive cost savings, confidence calibration, and live integration.
- Counterexamples bounding further weakening: raw E-M1-004 remains reproducible; only its acceptance interpretation is invalid.

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

The original M1-A-v1 command exits 2 under its preserved gate. The M1-B-v1 command also exited 2, but E-M1-005 later showed that its gate was invalid. Exact configs and provenance are in their result artifacts.

## Failures, regressions, and rejected paths

- H-M1-001 rejected: b RMSE 0.2138646913 is not below 0.15.
- H-M1-003 superseded: its raw checks failed, but the validator compared non-equivalent estimators on unlinked scales and used the wrong Monte Carlo population.
- No seed or hyperparameter search was used to replace the failed acceptance result.
- No engineering regressions were observed in the final verification suite.

## Remaining caveats and unresolved constraints

- M1-B-v2 is pending, so all M2–M5 requirements remain unresolved.
- The SWE matrix ranks submitted model-plus-agent systems and carries a `NOASSERTION` experiment-artifact license caveat.
- No empirical cost-reduction or fixed-confidence claim can be made.

## Highest-value next experiment

Run the preregistered `configs/m1_reference_agreement_v2.json` comparison from a clean revision.

Evidence confidence: high for the E-M1-005 validity diagnosis; no confidence is assigned to the pending M1-B-v2 or unrun downstream claims.

## Evidence map

| Evidence ID | Artifact / command | Conclusion |
|---|---|---|
| E-M0-005 | Processed manifests; `scripts/verify_m0_data.py` | M0 passes |
| E-M1-001 | `artifacts/m1-recovery-result.json`; `uv run python scripts/run_m1_recovery.py` | Original M1-A-v1 fails; result remains valid after contract revision |
| E-M1-002 | `artifacts/m1-recovery-rank-v2-result.json`; v2 config | Revised M1-A passes and unblocks M1-B |
| E-M1-003 | `artifacts/m1b-invalid-80-iterations.json` | Initial M1-B attempt invalid; numerical ceiling repaired |
| E-M1-004 | `artifacts/m1-reference-agreement-result.json`; `artifacts/M1B_FAILURE.md` | Reproducible M1-B-v1 raw measurements; acceptance inference invalidated by E-M1-005 |
| E-M1-005 | `artifacts/M1B_VALIDATOR_DIAGNOSIS.md` | M1-B-v1 acceptance inference invalid; corrected validator required |
