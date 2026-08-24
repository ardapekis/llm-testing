# ML Improvement State

## Readiness and objective

- Status: `READY`
- Objective: build and validate a cost-minimal adaptive IRT system for LLM benchmark ranking, executing M0–M5 in order.
- Intake: `INTAKE.md`

## Objective

Build and validate a Python 3.11+ cost-minimal adaptive IRT ranking system, executing M0–M5 in order and stopping with a diagnostic report at the first failed gate.

## Locked contract

- Primary metric: fixed-confidence epsilon-inversion rate <= delta; fixed-budget Kendall tau and inversion rate.
- Primary empirical target: >=5x median dollar-cost reduction vs random at tau >=0.95, clear win over cat-se, and calibrated epsilon-inversion rate at delta in {0.05, 0.1}.
- Baselines: random, stratified, hard-subset, cat-se, cat-cost.
- Development / sealed evaluation: seeds 0–49 / at least 200 seeds beginning at 10,000.
- Completion: all M0–M5 gates pass in order with reproducible direct evidence.
- Revised M1-A: convergence and theta Spearman >0.98 are hard requirements; difficulty RMSE is diagnostic. The original v1 failure remains valid evidence.
- External budget: no paid API spend authorized; public downloads and local CPU work only.

## Current provenance

- Repository: `/Users/arda/Documents/ChatGPT/New project`
- Revision: M1-B ran cleanly from `2e93db345d0f86e04f05782b8ab3064c1dc50468` on `codex/adaptive-irt-ranking`
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
| E-M1-002 | M1-A-v2 passes: estimator converged and theta Spearman 0.99178 exceeds 0.98; b RMSE 0.21386 remains diagnostic | Revision `496012a`, clean worktree | `artifacts/m1-recovery-rank-v2-result.json`; exit code 0 |
| E-M1-003 | First M1-B attempt is invalid: local real-matrix EM hit its 80-iteration ceiling before agreement metrics; diagnostic ceiling 500 converged at iteration 168 | Revision `9eba82f`, clean worktree | `artifacts/m1b-invalid-80-iterations.json` |
| E-M1-004 | M1-B-v1 raw checks fail: b Spearman 0.999944 passes, but real b RMSE 0.66304 exceeds bootstrap q95 0.60888 and only 99/100 bootstrap local fits converge; E-M1-005 invalidates the acceptance inference | Revision `2e93db3`, clean worktree | `artifacts/m1-reference-agreement-result.json`; exit code 2 |
| E-M1-005 | M1-B-v1 validator is invalid: it compared different 1PL estimating equations on unlinked scales and generated Monte Carlo outcomes from fixed EAP abilities rather than N(0,1) draws | Raw E-M1-004 preserved | `artifacts/M1B_VALIDATOR_DIAGNOSIS.md` |
| E-M1-006 | Lightweight M1-B-v2 passes all six checks on a deterministic 64-item sample; all four Monte Carlo replicates are valid | Revision `ed6ca95`, clean worktree; 52.44 seconds | `artifacts/m1-reference-agreement-v2-result.json`; exit code 0 |
| E-R-001 | Reduced M2–M5 prototype completes on both matrices over 20 seeds; fake adapter passes; rank-aware shows no clear baseline win | Revision `6f7a11d`, clean worktree; 32.29 seconds | `artifacts/reduced-m2-m5-result.json`; exit code 0 |

## Best valid candidate

- Candidate: checked-in response matrices plus the implemented 1PL/2PL/3PL core and revised rank-recovery gate.
- Evidence: E-M0-005 and E-M1-002.
- Gate status: M0, M1-A-v2, and M1-B-v2 pass; M1-B-v1 remains invalid. The reduced M2–M5 prototype is complete; the original M2–M5 gates remain unresolved by explicit scope reduction.

## Hypothesis register

| H-ID | Claim | Commitments | Prediction | Falsifier | Status | Evidence |
|---|---|---|---|---|---|---|
| H-M0-001 | The two pinned sources satisfy M0 without paid evaluation | MMLU rows are 0/1; SWE unresolved complement counts as incorrect | >=15 entities, >=500 items, >=90% dense for both | Invalid values, fewer rows/items, unresolved IDs outside canonical set, or unusable provenance/licence | SUPPORTED | E-M0-005 |
| H-M1-001 | The preregistered synthetic regime satisfies M1-A | 50 models, 1,000 items, fixed discrimination 2.5, seed 20260823, no post-result tuning | b RMSE <0.15 and theta Spearman >0.98 | Either threshold fails | REJECTED | E-M1-001 |
| H-M1-002 | The unchanged synthetic run establishes ranking recovery under revised M1-A | Same data, estimator, and seed as v1; only the user-authorized decision rule changes | Convergence and theta Spearman >0.98 | Nonconvergence or theta Spearman <=0.98 | SUPPORTED | E-M1-002 |
| H-M1-003 | Local and reference 1PL item difficulties agree within Monte Carlo error on the full SWE matrix | Fixed a=1; local Bock-Aitkin EM vs girth marginal-rate Rasch estimator; unlinked raw b; fixed-EAP bootstrap | Raw gate checks | Validator equivalence failure | SUPERSEDED | E-M1-004 raw result; invalidated by E-M1-005 |
| H-M1-004 | Equivalent local and girth joint 2PL MML implementations agree after Stocking-Lord linking | Full artifact validated; deterministic 64-item estimable sample; theta drawn from N(0,1); 4 Monte Carlo replicates | b rho >0.98; a rho >0.95; b/log-a/ICC RMSE each <= its bootstrap q95 | Any valid hard check fails | SUPPORTED | E-M1-006 |

## Weakest currently admissible claim

- Claim: M1-B-v1 shows nearly identical ordinal difficulty but cannot support an absolute-scale agreement conclusion because its validator is confounded.
- Directly tested scope: estimator-equation inspection, real-fit scale decomposition, bootstrap-generator inspection, and a like-for-like synthetic 2PL smoke comparison.
- Conditions retained: E-M1-004 raw measurements remain valid; its acceptance inference is invalid. M1-B-v2 is not yet evaluated on the real matrix.
- Action: preserve the passing M1-B-v2 artifact and implement M2 replay baselines.

## Constraint summary

| ID | Status | Evidence | Next action |
|---|---|---|---|
| M0 | PASS | E-M0-005 | Preserve artifacts and provenance |
| M1-A-v1 | FAIL | E-M1-001 | Preserve as rejected original contract |
| M1-A-v2 | PASS | E-M1-002 | Preserve config and result |
| M1-B-v1 | INVALID | E-M1-005 | Preserve E-M1-004 raw result; do not use its gate decision |
| M1-B-v2 | PASS | E-M1-006 | Preserve artifact and config |
| Reduced M2–M5 | PASS | E-R-001 | Preserve exploratory artifact and limitations |
| Original M2–M5 | UNRESOLVED | Explicitly deferred scope | Do not infer original-gate completion |

## Validity concerns

- The tinyBenchmarks artifact is a pickle; acquisition restricts loading to the two audited NumPy globals.
- The SWE-bench experiments repository does not include a root licence file in the pinned checkout. The manifest must distinguish the MIT SWE-bench code/dataset licence from the experiments artifact's `NOASSERTION` status.
- SWE submissions confound base model and agent harness. Any later ranking claim must call them systems.

## Contract revision

- Constraint: M1-A synthetic parameter recovery.
- Result: rejected by difficulty RMSE 0.21386 >= 0.15; theta Spearman 0.99178 > 0.98.
- Diagnostic: the true-theta oracle RMSE (0.17570) and Cramér–Rao RMS scale (0.16688) also exceed 0.15 under the frozen 50-response-per-item regime.
- Revision: the user authorized ability-rank recovery as the M1-A hard gate on 2026-08-23. Difficulty RMSE remains diagnostic and E-M1-001 remains a valid failed result under v1.

## Next targeted objective

- Constraint: M2 replay baseline curves.
- Experiment: implement the five required policies and cost models behind an information barrier, then run at least 200 sealed seeds on both matrices.
- Exact first action: preregister the replay protocol and implement the common replay engine plus baseline policies.
- Decision map: valid baseline curves permit M3 policy development; a validity failure repairs only the verifier.

## Remaining budget

- Full runs: no numerical limit supplied; the revised M1-A run and subsequent local milestone work are authorized.
- Compute: local CPU available.
- External cost: $0 authorized.
- Recursive rounds: unbounded, subject to milestone gates.
- Replication reserve: untouched downstream sealed seeds; M2 was not reached.
