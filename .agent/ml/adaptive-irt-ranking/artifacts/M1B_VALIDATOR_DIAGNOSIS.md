# M1-B-v1 validator diagnosis

The E-M1-004 measurements are reproducible, but the acceptance inference is invalid because the validator compared non-equivalent estimators and used a bootstrap inconsistent with its declared population model.

## Estimator mismatch

- The local 1PL implementation uses Bock-Aitkin EM and updates item difficulties from posterior expected response counts over complete response patterns.
- `girth 0.8.0` `rasch_mml` with fixed discrimination solves each difficulty from the item's marginal pass rate under a normal latent density.
- On the real matrix, girth's implied marginal pass rates match the observed rates with RMSE 0.00102; the local joint-EM parameters have marginal-rate RMSE 0.05154. These are different estimating equations under model misspecification.

## Scale confounding

The v1 result had difficulty Spearman 0.999944 despite raw RMSE 0.66304. A diagnostic affine relationship was:

`b_local ~= 1.18693 * b_girth - 0.37255`

After affine alignment, RMSE fell to 0.32145. Raw latent coordinates therefore mixed global location/scale differences with itemwise disagreement.

## Bootstrap mismatch

The v1 runner generated outcomes from fixed local EAP abilities. Those abilities had empirical mean -0.28342 and standard deviation 1.96974, rather than being drawn from the declared `N(0,1)` population. The resulting q95 was not a clean Monte Carlo tolerance under a common model.

## Revised validator

M1-B-v2 compares the local and girth joint 2PL MML paths, removes only constant non-estimable items, links the reference scale by Stocking-Lord test-characteristic-curve matching, and draws new abilities from `N(0,1)` in every Monte Carlo replicate.

The failing-before regression test could not import the missing estimable-item and scale-linking primitives. After implementation, the focused suite passes. A 120-model x 20-item like-for-like synthetic smoke comparison produced:

- difficulty RMSE: 0.00510;
- log-discrimination RMSE: 0.00134;
- difficulty Spearman: 1.0;
- discrimination Spearman: 0.99850;
- item-characteristic-curve RMSE: 0.00094.

E-M1-004 remains preserved as evidence about the v1 pipeline, but it no longer supports a scientific M1-B failure decision.
