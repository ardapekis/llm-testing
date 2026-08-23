# M1-B reference-agreement failure

M1-B was evaluated from clean revision `2e93db345d0f86e04f05782b8ab3064c1dc50468` on the full 134-system x 500-item SWE-bench Verified matrix.

The local fixed-discrimination 1PL MML-EM calibration was compared with `girth 0.8.0` `rasch_mml`. The Monte Carlo tolerance was preregistered as the 95th percentile of difficulty RMSE across 100 paired parametric-bootstrap matrices.

| Check | Required | Observed | Result |
|---|---:|---:|---|
| Difficulty Spearman | > 0.99 | 0.9999439914 | PASS |
| Difficulty RMSE | <= bootstrap q95 | 0.6630381298 vs 0.6088838866 | FAIL |
| Local bootstrap convergence | 100 / 100 | 99 / 100 | FAIL |

The real local fit converged at iteration 168. Bootstrap RMSE ranged from 0.5187877005 to 0.6299211656, with median 0.5547452898. The result artifact records `validity: valid` and `gate_pass: false`.

The near-perfect rank correlation supports only ordinal agreement. It does not establish that item parameters agree on the absolute latent scale within the preregistered Monte Carlo tolerance. Per the milestone contract, M2 was not started.
