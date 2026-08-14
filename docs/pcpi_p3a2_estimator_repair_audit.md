# P3A.2 class-EIG estimator repair audit

Status: **development Gate passed; formal source-paired run pending**.

## Audited failure

The uploaded P3A.1 result passed its frozen single-fixture thresholds, but it
was not publication-ready. It covered one posterior/class partition, its
largest-budget RMSE was not monotone, one seed had 0.36 marginal interval
coverage, and the result manifest named a different source-archive SHA-256
than the accompanying archive.

Multi-scenario stress tests exposed the root cause. Inverse-CDF Student-t
sampling left very small posterior strata with too few tail observations.
Replicate and within-stratum variance estimates were unstable for
near-degenerate class posteriors. Changing a coverage threshold would not
repair that estimator.

## Target-level repair

For a standardized Student-t predictive variable (T\sim t_\nu),

\[
U=\frac{\nu}{\nu+T^2}\sim\operatorname{Beta}(\nu/2,1/2),
\]

with an independent equiprobable sign. The production estimator now evaluates
the sign-antithetic information integrand with a Gauss--Jacobi rule whose
weight matches this Beta law. Half of the node-pair budget is allocated
uniformly over structures and the remainder by posterior structure mass. This
preserves support for low-mass structures without a dataset, target, formula,
or direction branch.

The estimator returns a nested numerical-error envelope equal to four times
the fine/coarse discrepancy plus a floating-point roundoff floor. The ranking
diagnostic requires the leader's lower envelope to exceed every competitor's
upper envelope. This envelope is an asymptotic diagnostic validated against an
independent adaptive quadrature reference; it is not claimed as a rigorous
quadrature theorem or probabilistic confidence interval.

## Development evidence

The repaired development Gate uses eight scenarios: two fixture-generating
seeds crossed with 4, 6, 12, and 20 observations. Their class entropies span
4.87 orders of magnitude. Evaluation budgets are 32, 64, 128, 256, and 512.

At 512 evaluations, the development run obtained:

- mean Spearman correlation: 1.0;
- worst-scenario Spearman correlation: 1.0;
- top-1 agreement: 8/8;
- mean normalized RMSE: (3.00\times10^{-11});
- worst-scenario normalized RMSE: (1.06\times10^{-10});
- simultaneous error-envelope coverage: 8/8;
- rank certification: 8/8;
- false rank certification across every scenario and budget: 0;
- log--log normalized-RMSE slope: -7.14.

All 106 tests passed after the repair. These numbers are development evidence,
not a formal result package, because they were not produced from the final
canonical archive identity.

## Claim boundary and next Gate

The repair can support a formal class-EIG numerical-correctness claim only
after one run whose `source_package_hash` exactly equals the delivered source
ZIP SHA-256 and whose EvidenceRegistry verifies. It cannot support real-data
acquisition superiority, scientific discovery efficacy, held-out
confirmation, motif safety, VED discovery, physical intervention, or a new
law. P3B.2 remains blocked until the formal P3A.2 result is returned and
audited.
