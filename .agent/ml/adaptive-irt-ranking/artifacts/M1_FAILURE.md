# M1 acceptance failure

M1-A was evaluated from clean revision `121cec42581122016ec17343a55a0ee82917de29` with the frozen configuration in `configs/m1_recovery.json`.

| Metric | Required | Observed | Result |
|---|---:|---:|---|
| Difficulty RMSE | < 0.15 | 0.2138646913 | FAIL |
| Theta Spearman | > 0.98 | 0.9917795029 | PASS |

The estimator converged in 10 EM iterations. A same-data oracle calculation using the true model abilities yielded difficulty RMSE 0.1757007915. The Cramér–Rao RMS scale was 0.1668800354. These diagnostics exceed the required 0.15 threshold under the frozen 50-response-per-item regime and indicate that the observed failure cannot be attributed solely to latent-ability estimation.

Per the ordered milestone contract, M1-B was not run and no M2 implementation or experiment was started. No post-observation tuning was performed.
