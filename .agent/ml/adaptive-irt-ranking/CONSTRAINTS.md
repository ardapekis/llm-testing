# Verification Matrix

Statuses: `PASS`, `PROVISIONAL`, `UNRESOLVED`, `FAIL`, `INVALID`, `N/A`.

| ID | Constraint | Hard/Soft | Verifier | Threshold | Status | Evidence | Owner / next action |
|---|---|---|---|---|---|---|---|
| M0-A | Public matrix count and size | Hard | Data audit | >=2 matrices; each >=15 entities x >=500 items | PASS | E-M0-005: 395x14,042 and 134x500 | Preserve |
| M0-B | Matrix density and binary validity | Hard | Data audit | >=90% observed; values in {0,1} | PASS | E-M0-005: both 100% dense, Boolean, pair-unique | Preserve |
| M0-C | Provenance and licence manifest | Hard | Manifest schema/checksum test | Source URL, commit, artifact hash, retrieval time, licence status, derived hash | PASS | `data/processed/*/manifest.json`; E-M0-005 | Preserve `NOASSERTION` caveat |
| M1-A-v1 | Original synthetic parameter recovery | Hard (superseded) | 50x1000 synthetic experiment | b RMSE <0.15; theta Spearman >0.98 | FAIL | E-M1-001: b RMSE 0.21386; theta Spearman 0.99178; clean revision `121cec4` | Preserve negative evidence |
| M1-A-v2 | Synthetic ability-rank recovery | Hard | Same 50x1000 experiment | converged; theta Spearman >0.98; b RMSE diagnostic | PASS | E-M1-002: converged; theta Spearman 0.99178; clean revision `496012a` | Preserve |
| M1-B-v1 | Original reference implementation agreement | Hard (superseded) | Local Bock-Aitkin 1PL vs girth marginal-rate Rasch; raw scale; fixed-EAP bootstrap | Original raw checks | INVALID | E-M1-005; E-M1-004 raw metrics preserved | Do not use as acceptance evidence |
| M1-B-v2 | Corrected lightweight reference implementation agreement | Hard | Validate full SWE artifact; salted-hash sample of 64 estimable items; local vs girth joint 2PL MML; Stocking-Lord link; 4 theta~N(0,1) replicates | >=90% estimable; b rho >0.98; a rho >0.95; observed b/log-a/ICC RMSE each <= bootstrap q95 | PASS | E-M1-006; clean `ed6ca95`; all six checks pass | Preserve; M2 unblocked |
| M1-B-v2-full | Original corrected full reference agreement | Hard | All 468 estimable SWE items; 20 theta~N(0,1) replicates | Same six M1-B-v2 checks | PASS | E-M1-007; clean `5d8f926`; 20/20 valid | Preserve as stronger full-run evidence |
| M2 | Replay baseline curves | Hard | >=200 sealed seeds on both matrices | Cost-vs-tau and cost-vs-inversion curves with bootstrap CIs | UNRESOLVED | | Blocked by M1 order |
| M3-A | Adaptive cost reduction | Hard | Matched-fidelity replay | >=5x median cost vs random at tau >=0.95 and clear win over cat-se | UNRESOLVED | | Blocked by M1-B and M2 order |
| M3-B | Fixed-confidence calibration | Hard | >=200 sealed seeds | Empirical epsilon-inversion rate <= delta for delta 0.05 and 0.1 | UNRESOLVED | | Blocked by M1-B and M2 order |
| M3-C | Information barrier | Hard | Adversarial replay test | Policy cannot read unrevealed outcomes | UNRESOLVED | | Blocked by M1-B and M2 order |
| M3-D | Ranking-selection signature | Hard | Selection trace analysis | Item difficulty concentrates near contested midpoint, not own theta | UNRESOLVED | | Blocked by M1-B and M2 order |
| M4 | Live adapter and capped run | Hard | Integration run | >=5 models; cache; hard ceiling enforced; graceful report | UNRESOLVED | | Paid/provider authority deferred |
| M5 | Final report and ablations | Hard | Artifact audit | Required curves, ablations, failures, invalidating assumptions | UNRESOLVED | | Blocked by prior gates |
| R-M2 | Reduced fixed-budget replay | Reduced | 20 seeds; both matrices; five checkpoints; unit cost | Curves and IQR summaries emitted | PASS | E-R-001 | Exploratory only |
| R-M3 | Reduced rank-aware comparison | Reduced | Rank-aware plus no-contest ablation | Compare and report without 5x gate | PASS (negative result) | E-R-001 | No clear win |
| R-M4 | Reduced fake adapter | Reduced | Five fake models; cache; cap | Cache consistent; cap enforced before call | PASS | E-R-001 | No live calls |
| R-M5 | Reduced report | Reduced | Artifact and limitations audit | Results, ablation, commands, caveats present | PASS | E-R-001; `REPORT.md` | Original M2–M5 unresolved |
| ENG | Package quality | Hard | `pytest`, `ruff`, `mypy --strict` | All pass; no network calls in tests | PASS | 44 tests; Ruff; strict mypy | Preserve |
| RANK | Anchored uncertainty-aware ranking | Hard | Fixed item bank; EAP posterior; pairwise epsilon rule | Total display order distinguished from confidence partial order | PASS | Library/CLI tests and 134-system smoke run | Prefer independent 2PL anchor |
