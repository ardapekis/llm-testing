# Agentic ML Engineering Intake

## Readiness

- Status: `READY`
- Repository: `/Users/arda/Documents/ChatGPT/New project`
- Blocking decisions: none for M0–M3. A paid M4 run requires a configured provider and an explicit dollar cap before calls are issued.

## Configured contract

| Field | Value | Source | Evidence / consequence |
|---|---|---|---|
| Decision and objective | Rank LLMs at minimum evaluation cost, with a stated fixed-confidence error guarantee or a fixed budget | `USER_CONFIRMED` | Attached project specification |
| Deliverable | Typed Python 3.11+ package, reproducible experiments, two public replay matrices, live adapter, and `REPORT.md` | `USER_CONFIRMED` | Mission and engineering constraints |
| Target and access | Empty local Git repository; public data may be downloaded; tests make no network calls | `USER_CONFIRMED` | Repository inspection; project specification |
| Primary verifier | Fixed-confidence epsilon-inversion rate at most delta; fixed-budget Kendall tau / inversion rate; M3 target is at least 5x median cost reduction vs random at tau >= 0.95 and a clear win over cat-se | `USER_CONFIRMED` | M2/M3 gates |
| Milestone stopping rule | Execute M0–M5 in order and stop/report on the first failed gate | `USER_CONFIRMED` | Working style and milestone gates |
| Baselines | random, stratified, hard-subset, cat-se, and cat-cost | `USER_CONFIRMED` | Policy table |
| Development policy | Develop on synthetic seeds 0–49 and replay seeds 0–49; do not inspect sealed replay outcomes for tuning | `AGENT_PROPOSED` | Reversible split that protects the required final experiment |
| Sealed evaluation policy | Final replay uses at least 200 fresh seeds beginning at 10,000, fixed before M3 tuning | `AGENT_PROPOSED` | Required >=200 seeds; no test-set tuning |
| Hard constraints | Correct 3PL Fisher information; anytime-valid stopping; item-parameter uncertainty; information barrier; dimensionality and dependence diagnostics; reproducibility; no network in tests | `USER_CONFIRMED` | Statistical and engineering requirements |
| Non-goals | Training/fine-tuning, leaderboard UI, human-subject testing, contamination remediation, product claims about frontier-model rankings | `USER_CONFIRMED` | Non-goals section |
| Budget and authorization | Public downloads and local CPU work are authorized by the requested deliverable. No paid model/API spend is authorized yet. | `REPO_INFERRED` | M0 requires acquisition; M4 spend deferred |
| Completion rule | Every M0–M5 gate passes with direct artifacts and a requirement-by-requirement audit | `USER_CONFIRMED` | Acceptance criteria and goal completion audit |
| Intended claim scope | Only the exact dataset revisions, seeds, cost configs, and live model/provider versions recorded in manifests | `AGENT_PROPOSED` | Prevents unsupported generalization |

## Authorization boundaries

- Pre-authorized compute: local CPU work and small dependency installation.
- Pre-authorized external services: read-only downloads of public source repositories and documentation.
- Data movement: public benchmark artifacts into this repository; no private data.
- Destructive changes: none.
- Explicit approval required: paid M4 calls, credentials, or cloud compute spend.

## READY deployment brief

- Decision: whether rank-aware adaptive allocation can lower dollar cost at matched ranking fidelity without breaking coverage.
- Deliverable: package, data manifests, experiment configs/artifacts, adapter, and report.
- Baseline: the five required non-rank-aware policies on the same response cache and cost model.
- Evaluation: development seeds are separate from a locked 200-seed final cohort.
- Completion: all milestone gates pass in order; a failed gate is reported rather than tuned away.
- Nonblocking assumptions: public benchmark submissions are replay entities even when a SWE-bench row represents a model+agent system rather than a bare model.
- Exact first action: transform the two pinned public artifacts into validated long-format Parquet matrices and emit provenance/licence manifests.

## Configuration history

- 2026-08-23: Repository inspected and found empty; task marked `READY` from the complete user specification.
- 2026-08-23: Deferred only paid live-evaluation authorization; it does not block M0–M3.
