# Final ML Engineering Report

## Outer decision

`REFINE (resumed)`

M0 passed. M1-A-v1 failed its frozen difficulty-RMSE threshold. The user revised M1-A on 2026-08-23 to gate ranking recovery on convergence and theta Spearman >0.98 while retaining difficulty RMSE as a diagnostic. M1-A-v2 is preregistered but not yet evaluated.

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

## Baseline and final candidate

| Item | Provenance | Primary metric | Secondary metrics | Runtime / cost |
|---|---|---:|---|---|
| M0 processed matrices | E-M0-005; commit `8c2ba93` | Both structural gates pass | Pair uniqueness, schema, checksum, manifest | Local CPU; no paid cost |
| M1-A 1PL recovery | E-M1-001; clean revision `121cec4` | b RMSE 0.2138646913 (FAIL) | theta Spearman 0.9917795029; 10 iterations | 0.454 seconds; no paid cost |

## Constraint outcomes

| ID | Final status | Evidence | Notes |
|---|---|---|---|
| M0-A/B/C | PASS | E-M0-005 | Two public, dense, binary, manifested matrices |
| M1-A-v1 | FAIL | E-M1-001 | b RMSE exceeds 0.15; preserved negative evidence |
| M1-A-v2 | UNRESOLVED | `configs/m1_recovery_rank_v2.json` | User-authorized revised gate |
| M1-B | UNRESOLVED | Not run | Begins only after M1-A-v2 passes |
| M2–M5 | UNRESOLVED | Not run | Blocked by milestone order |
| ENG | PASS | 18 tests; Ruff; strict mypy | No network tests |

## Weakest supported claim

- Claim: the IRT implementation converges and recovers the ability ordering in the frozen synthetic run, but it fails the required difficulty-recovery gate.
- Tested scope: exactly revision `121cec4`, configuration hash `39039a9b...27a`, and seed 20260823.
- Conditions removed: none; the acceptance configuration was not weakened after observation.
- Stronger interpretations not established: reference agreement, adaptive cost savings, confidence calibration, and live integration.
- Counterexamples bounding further weakening: true-theta oracle b RMSE 0.1757007915 and Cramér–Rao RMS scale 0.1668800354 both exceed the 0.15 gate.

## Commands and reproducibility

```bash
uv run python scripts/verify_m0_data.py
uv run python scripts/run_m1_recovery.py
uv run pytest
uv run ruff check .
uv run mypy src scripts tests
```

The M1 command is expected to exit 2 because the gate fails. Exact config and provenance are in `artifacts/m1-recovery-result.json`.

## Failures, regressions, and rejected paths

- H-M1-001 rejected: b RMSE 0.2138646913 is not below 0.15.
- No seed or hyperparameter search was used to replace the failed acceptance result.
- No engineering regressions were observed in the final verification suite.

## Remaining caveats and unresolved constraints

- M1-B and all M2–M5 requirements remain unresolved.
- The SWE matrix ranks submitted model-plus-agent systems and carries a `NOASSERTION` experiment-artifact license caveat.
- No empirical cost-reduction or fixed-confidence claim can be made.

## Highest-value next experiment

Run `configs/m1_recovery_rank_v2.json` from the clean preregistration revision. If it passes, run M1-B against `girth` on one full real matrix.

Evidence confidence: 98/100 for the M1-A failure decision; no confidence is assigned to unrun downstream claims.

## Evidence map

| Evidence ID | Artifact / command | Conclusion |
|---|---|---|
| E-M0-005 | Processed manifests; `scripts/verify_m0_data.py` | M0 passes |
| E-M1-001 | `artifacts/m1-recovery-result.json`; `uv run python scripts/run_m1_recovery.py` | Original M1-A-v1 fails; result remains valid after contract revision |
