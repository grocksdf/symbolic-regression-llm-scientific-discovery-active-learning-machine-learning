# P3J.15 pointwise-information repair

The first P3J.15 user execution at source commit `306c6b0f59b70d673717bbf9546b93b10507a19e`
terminated on `uci_ccpp`, seed `2026080705`, query 16 before publishing a
decision or opening its response. The terminal record reports
`FloatingPointError: P3J batched quadrature produced negative information`;
held-out access, validation-based selection, seed replacement, and retry in
that terminal workspace all remained false or forbidden.

The failure exposed a numerical representation problem rather than a changed
estimand. The former implementation evaluated

`sum_c p(c) E_{q_c}[log(q_c(Y) / sum_j p(j) q_j(Y))]`

on a different deterministic response grid for every class. Although the
complete expression is mutual information and is nonnegative exactly, finite
quadrature can lose the cancellation between its signed class contributions.

The repaired implementation evaluates the exactly equivalent identity

`E_{Y ~ sum_c p(c) q_c}[KL(p(C | Y) || p(C))]`.

Every response-node integrand is now a normalized posterior-class KL computed
with log-sum-exp and relative entropy before mixture quadrature. Thus
nonnegativity follows pointwise from the information functional; it is not a
score floor, result-conditioned tolerance, changed ranking rule, or empirical
tuning parameter. Scalar and batched paths use the same helper, and tests bind
the identity, extreme-log-density stability, batch equivalence, refinement,
checkpoint, and source-integrity contracts.

The failed output remains immutable evidence. Because the source tree changed,
it cannot be resumed. Any repaired formal execution must use a new output
directory and freeze the new commit and tree while retaining the same P3J.15
configuration, datasets, seeds, budgets, decision rule, and sealed held-out
state.
