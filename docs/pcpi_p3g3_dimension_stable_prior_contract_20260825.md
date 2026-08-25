# PCPI P3G.3 dimension-stable function-prior contract

P3G.2 completed all 24 real calibration audits and preserved the three P3G.1
rejections. Its added diagnostics identify central PIT concentration rather
than heavy tails: the rejected PIT variances are `0.0560`, `0.0571`, and
`0.0597`, below the uniform value `1/12`, with positive quadratic betting-basis
means. P3G.2 remains immutable negative evidence; acquisition count is zero.

The remaining defect is response-independent and mathematical. Every
non-intercept column is standardized to unit empirical variance, but the old
prior assigns every coefficient variance one. A structure with `p`
non-intercept terms therefore has average prior function variance `p`, so
epistemic uncertainty grows with the arbitrary representation dimension.
This is especially material when 32 initial observations confront the generic
Gas structures containing dozens of polynomial terms.

P3G.3 replaces only this prior geometry. For a structure with `p>0`
standardized non-intercept columns it assigns each corresponding coefficient
precision `p`, retaining precision one for the intercept. Consequently,

`trace(D_nonintercept Cov(beta) D_nonintercept^T) / n = 1`

on the frozen initial-development design for every registered structure. This
equalizes prior function energy without
observing a response, deleting a term, selecting a seed, or fitting a penalty.
The ordinary normal-inverse-gamma posterior, exact marginal likelihood,
rank-one update, heteroscedastic mixture, and joint nuisance EIG remain intact.

The construction is related to design-aware `g`-prior and prior-on-model-fit
principles, while avoiding an inverse Gram matrix that is undefined for the
registered `p>n` structures:

- Bondell and Reich, *Bayesian Variable Selection Using an Adaptive Powered
  Correlation Prior*, 2012: <https://pmc.ncbi.nlm.nih.gov/articles/PMC2772159/>
- Zhang et al., *Bayesian Regression Using a Prior on the Model Fit: The R2-D2
  Shrinkage Prior*: <https://arxiv.org/abs/1609.00046>

Seeds, splits, validation order, budgets, PIT betting family, `1/100`
familywise allocation, `1/2400` per-run boundary, candidate sealing, failure
policy, and held-out closure are unchanged. A 24/24 calibration pass is still
required before the same process can construct the oracle and run 96
matched-budget acquisition comparisons.
