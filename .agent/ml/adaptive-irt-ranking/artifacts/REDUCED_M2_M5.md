# Reduced M2–M5 result

The user-authorized reduced prototype ran from clean revision `6b1565f` in 31.1154 seconds.

- Both pinned matrices were replayed over seeds 10000–10019 at five fixed-budget checkpoints.
- Random, balanced-random, CAT-SE proxy, rank-aware, and no-contest-ablation policies were compared under unit observation cost.
- Rank-aware did not clearly outperform the baselines; at 5% budget it reached tau 0.8803 on MMLU and 0.7193 on SWE-bench Verified.
- The fake-provider integration queried all five models, cached repeated results, made eight allowed provider calls, and rejected an unseen ninth request before calling the provider.

This does not satisfy or claim the original 200-seed, fixed-confidence, dollar-cost, live-provider, or 5x-improvement gates.
