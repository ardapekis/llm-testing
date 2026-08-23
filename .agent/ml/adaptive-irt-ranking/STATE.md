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
- Revision: `121cec42581122016ec17343a55a0ee82917de29` on `codex/adaptive-irt-ranking`
- Initial state: upstream `main` contained only a README and stock Python `.gitignore`
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
| E-M1-001 | M1-A fails: the preregistered 50x1,000 1PL run has difficulty RMSE 0.21386, above 0.15; theta Spearman 0.99178 passes | Revision `121cec4`, clean worktree | `artifacts/m1-recovery-result.json`; exit code 2 |

## Best valid candidate

- Candidate: checked-in MMLU and SWE-bench Verified long-format Parquet matrices.
- Evidence: E-M0-005.
- Gate status: M0 passed. MMLU SHA-256 `eebe5be98848d595f005e76f35bab9eb795cddac9b74e6feec3dfd1df99283e4`; SWE SHA-256 `a1cd623aad7f7e1405db19ee4878bfe532466e71fc29e24908e018b5dadb1d80`.

## Hypothesis register

| H-ID | Claim | Commitments | Prediction | Falsifier | Status | Evidence |
|---|---|---|---|---|---|---|
| H-M0-001 | The two pinned sources satisfy M0 without paid evaluation | MMLU rows are 0/1; SWE unresolved complement counts as incorrect | >=15 entities, >=500 items, >=90% dense for both | Invalid values, fewer rows/items, unresolved IDs outside canonical set, or unusable provenance/licence | SUPPORTED | E-M0-005 |
| H-M1-001 | The preregistered synthetic regime satisfies M1-A | 50 models, 1,000 items, fixed discrimination 2.5, seed 20260823, no post-result tuning | b RMSE <0.15 and theta Spearman >0.98 | Either threshold fails | REJECTED | E-M1-001 |

## Weakest currently admissible claim

- Claim: the implemented 1PL MML-EM estimator converges and recovers ability ordering in the declared synthetic regime, but the required difficulty-recovery threshold is not met.
- Directly tested scope: the frozen M1-A configuration at revision `121cec4` and seed 20260823.
- Conditions retained: difficulty RMSE is 0.21386; the same-data true-theta oracle RMSE is 0.17570 and the Cramér–Rao RMS scale is 0.16688, both above the required threshold.
- Action: stop at M1-A per the milestone contract. Do not run M1-B or begin M2.

## Constraint summary

| ID | Status | Evidence | Next action |
|---|---|---|---|
| M0 | PASS | E-M0-005 | Preserve artifacts and provenance |
| M1-A | FAIL | E-M1-001 | Stop and report; no tuning after acceptance observation |
| M1-B | UNRESOLVED | Not run because M1-A failed | Blocked by milestone order |
| M2–M5 | UNRESOLVED | M1-A failure | Do not begin |

## Validity concerns

- The tinyBenchmarks artifact is a pickle; acquisition restricts loading to the two audited NumPy globals.
- The SWE-bench experiments repository does not include a root licence file in the pinned checkout. The manifest must distinguish the MIT SWE-bench code/dataset licence from the experiments artifact's `NOASSERTION` status.
- SWE submissions confound base model and agent harness. Any later ranking claim must call them systems.

## Stop condition

- Constraint: M1-A synthetic parameter recovery.
- Result: rejected by difficulty RMSE 0.21386 >= 0.15; theta Spearman 0.99178 > 0.98.
- Diagnostic: the true-theta oracle RMSE (0.17570) and Cramér–Rao RMS scale (0.16688) also exceed 0.15 under the frozen 50-response-per-item regime.
- Required action: report the failed gate. A revised statistical contract or explicitly authorized new experiment is required before further milestone work.
