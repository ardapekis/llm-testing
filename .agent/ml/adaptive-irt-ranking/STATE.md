# ML Improvement State

## Readiness and objective

- Status: `READY`
- Objective: build and validate a cost-minimal adaptive IRT system for LLM benchmark ranking, executing M0–M5 in order.
- Intake: `INTAKE.md`

## Locked contract

- Primary metric: fixed-confidence epsilon-inversion rate <= delta; fixed-budget Kendall tau and inversion rate.
- Primary empirical target: >=5x median dollar-cost reduction vs random at tau >=0.95, clear win over cat-se, and calibrated epsilon-inversion rate at delta in {0.05, 0.1}.
- Baselines: random, stratified, hard-subset, cat-se, cat-cost.
- Development / sealed evaluation: seeds 0–49 / at least 200 seeds beginning at 10,000.
- Completion: all M0–M5 gates pass in order with reproducible direct evidence.
- External budget: no paid API spend authorized; public downloads and local CPU work only.

## Current provenance

- Repository: `/Users/arda/Documents/ChatGPT/New project`
- Revision: no commits yet on `main`
- Initial state: empty except Git metadata
- Runtime: macOS; system Python 3.14.0; `uv 0.6.3`; package targets Python >=3.11
- Source candidate 1: `felipemaiapolo/tinyBenchmarks` commit `e9a8b1031b0340571beb6c9ca3a27891be09a8fd`, `tutorials/data/lb.pickle`, SHA-256 `34f44d6a819512ef74d00a95288d252fa679288a10ca167cd97fdbc3aae66437`
- Source candidate 2: `SWE-bench/experiments` commit `1faa91cade0562ba62b66c1c99e71f7b72d96f13`, `evaluation/verified/*/results/results.json`

## Verified findings

| Evidence ID | Finding | Scope | Artifact / command |
|---|---|---|---|
| E-M0-001 | tinyBenchmarks artifact contains 395 entities and 62 scenarios; MMLU subtasks are item-by-model dense arrays | Pinned tinyBenchmarks commit | Restricted pickle inspection; source `tutorials/utils.py` |
| E-M0-002 | SWE-bench source contains 134 Verified result files | Pinned experiments commit | `find .../evaluation/verified -path '*/results/results.json'` |
| E-M0-003 | SWE canonical item set has 500 unique IDs; union of resolved IDs is a subset; no result has duplicate resolved IDs | Pinned experiments commit | `jq`, `sort`, `comm` audit in `/tmp` |
| E-M0-004 | SWE-bench result rows are model+agent submissions, so the second matrix ranks systems rather than isolated base models | SWE Verified source | Submission metadata and README |
| E-M0-005 | M0 passes: MMLU is 395x14,042 and SWE Verified is 134x500; both are binary, pair-unique, and 100% dense with verified checksums/manifests | Pinned processed artifacts | `scripts/verify_m0_data.py`; 10 passing tests; strict lint/type checks |

## Best valid candidate

- Candidate: checked-in MMLU and SWE-bench Verified long-format Parquet matrices.
- Evidence: E-M0-005.
- Gate status: M0 passed. MMLU SHA-256 `eebe5be98848d595f005e76f35bab9eb795cddac9b74e6feec3dfd1df99283e4`; SWE SHA-256 `a1cd623aad7f7e1405db19ee4878bfe532466e71fc29e24908e018b5dadb1d80`.

## Hypothesis register

| H-ID | Claim | Commitments | Prediction | Falsifier | Status | Evidence |
|---|---|---|---|---|---|---|
| H-M0-001 | The two pinned sources satisfy M0 without paid evaluation | MMLU rows are 0/1; SWE unresolved complement counts as incorrect | >=15 entities, >=500 items, >=90% dense for both | Invalid values, fewer rows/items, unresolved IDs outside canonical set, or unusable provenance/licence | SUPPORTED | E-M0-005 |

## Weakest currently admissible claim

- Claim: the two pinned processed artifacts satisfy the structural M0 gate.
- Directly tested scope: exact source revisions and generated Parquet files.
- Conditions retained: SWE rows represent submitted systems; source result-artifact licence is `NOASSERTION`.
- Action: begin M1 without broadening the empirical claim.

## Constraint summary

| ID | Status | Evidence | Next action |
|---|---|---|---|
| M0 | PASS | E-M0-005 | Preserve artifacts and provenance |
| M1 | UNRESOLVED | Milestone order | Implement IRT core and run recovery/reference gates |
| M2–M5 | UNRESOLVED | Milestone order | Do not begin until preceding gates pass |

## Validity concerns

- The tinyBenchmarks artifact is a pickle; acquisition restricts loading to the two audited NumPy globals.
- The SWE-bench experiments repository does not include a root licence file in the pinned checkout. The manifest must distinguish the MIT SWE-bench code/dataset licence from the experiments artifact's `NOASSERTION` status.
- SWE submissions confound base model and agent harness. Any later ranking claim must call them systems.

## Next targeted objective

- Constraint: M1 synthetic recovery and real-matrix reference agreement.
- Experiment: implement 1PL/2PL/3PL marginal-ML EM, MAP/EAP ability estimation, and synthetic recovery before touching adaptive replay.
- Positive result: with 50 models x 1000 items, b RMSE <0.15 and theta Spearman >0.98; real calibration agrees with a reference implementation within Monte Carlo error.
- Rejection result: stop at M1 and report the diagnostic rather than proceeding to M2.
