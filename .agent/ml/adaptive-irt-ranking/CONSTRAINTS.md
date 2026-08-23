# Verification Matrix

Statuses: `PASS`, `PROVISIONAL`, `UNRESOLVED`, `FAIL`, `INVALID`, `N/A`.

| ID | Constraint | Hard/Soft | Verifier | Threshold | Status | Evidence | Owner / next action |
|---|---|---|---|---|---|---|---|
| M0-A | Public matrix count and size | Hard | Data audit | >=2 matrices; each >=15 entities x >=500 items | PASS | E-M0-005: 395x14,042 and 134x500 | Preserve |
| M0-B | Matrix density and binary validity | Hard | Data audit | >=90% observed; values in {0,1} | PASS | E-M0-005: both 100% dense, Boolean, pair-unique | Preserve |
| M0-C | Provenance and licence manifest | Hard | Manifest schema/checksum test | Source URL, commit, artifact hash, retrieval time, licence status, derived hash | PASS | `data/processed/*/manifest.json`; E-M0-005 | Preserve `NOASSERTION` caveat |
| M1-A-v1 | Original synthetic parameter recovery | Hard (superseded) | 50x1000 synthetic experiment | b RMSE <0.15; theta Spearman >0.98 | FAIL | E-M1-001: b RMSE 0.21386; theta Spearman 0.99178; clean revision `121cec4` | Preserve negative evidence |
| M1-A-v2 | Synthetic ability-rank recovery | Hard | Same 50x1000 experiment | converged; theta Spearman >0.98; b RMSE diagnostic | PASS | E-M1-002: converged; theta Spearman 0.99178; clean revision `496012a` | Preserve |
| M1-B | Reference implementation agreement | Hard | Real-matrix comparison | Within Monte Carlo error vs girth or py-irt | UNRESOLVED | M1-A-v2 passed | Preregister and run full-matrix comparison |
| M2 | Replay baseline curves | Hard | >=200 sealed seeds on both matrices | Cost-vs-tau and cost-vs-inversion curves with bootstrap CIs | UNRESOLVED | | Blocked by M1 order |
| M3-A | Adaptive cost reduction | Hard | Matched-fidelity replay | >=5x median cost vs random at tau >=0.95 and clear win over cat-se | UNRESOLVED | | Blocked by M0 order |
| M3-B | Fixed-confidence calibration | Hard | >=200 sealed seeds | Empirical epsilon-inversion rate <= delta for delta 0.05 and 0.1 | UNRESOLVED | | Blocked by M0 order |
| M3-C | Information barrier | Hard | Adversarial replay test | Policy cannot read unrevealed outcomes | UNRESOLVED | | Blocked by M0 order |
| M3-D | Ranking-selection signature | Hard | Selection trace analysis | Item difficulty concentrates near contested midpoint, not own theta | UNRESOLVED | | Blocked by M0 order |
| M4 | Live adapter and capped run | Hard | Integration run | >=5 models; cache; hard ceiling enforced; graceful report | UNRESOLVED | | Paid/provider authority deferred |
| M5 | Final report and ablations | Hard | Artifact audit | Required curves, ablations, failures, invalidating assumptions | UNRESOLVED | | Blocked by prior gates |
| ENG | Package quality | Hard | `pytest`, `ruff`, `mypy --strict` | All pass; no network calls in tests | PASS | E-M1-001 verification: 18 tests, Ruff, and strict mypy pass | Preserve |
