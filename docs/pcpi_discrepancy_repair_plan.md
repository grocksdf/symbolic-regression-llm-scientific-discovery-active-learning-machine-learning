# PCPI discrepancy-aware repair (P3C proposal)

The P3B.10 real-data run is retained as negative development evidence. Its
finite likelihood-power envelope changed the ranking, but did not demonstrate
an advantage over random on CCPP. The repair therefore treats model mismatch as
a predictive discrepancy problem rather than adding another data-specific
power, threshold, seed, or regularizer.

## Contract

`discrepancy_predictive_profile` is a covariate-only profile built from:

- response sufficient statistics already present in the observed conjugate
  states (posterior-weighted residual mean square);
- the registered target-domain covariates and the currently observed design;
- a median positive target-domain squared-distance bandwidth.

For candidate `a`, the profile adds

\[
\delta^2(a)=\widehat\sigma^2_{\rm excess}
  \left(1+\min_{x\in A_t}\|a-x\|^2/h^2\right)
\]

to the predictive variance. The same construction is applied to the registered
target domain. The finite Student-t component is moment-matched to preserve its
location and degrees of freedom, and the class-conditional predictive term is
recomputed with the independent candidate/target discrepancy variances. Zero
excess variance is an exact no-op.

This is an acquisition-only model-misspecification envelope. It does not alter
the reporting posterior, validation metrics, classes, held-out state, or the
frozen P3B.10 policy. The new entry point is intentionally separate so the old
negative result remains reproducible.

## Frozen P3C.1 protocol

The candidate is exposed through the thin adapter
`scripts/run_pcpi_p3c_real.py` and the frozen configuration
`configs/p3c_1_discrepancy_real_acquisition.json`. The generic real runner is
parameterized by an immutable protocol object; P3B.10 remains the default
protocol and retains its original schema and policy tuple.

P3C.1 preserves the P3B.10 datasets, seeds, splits, budgets, baselines,
nominal reporting posterior, likelihood-power ambiguity family,
representative safe set, assessment rules, and closed held-out state. The
only policy change is the discrepancy envelope used by PCPI ranking. Query
evidence records the method, residual excess variance, support bandwidth,
selected candidate discrepancy variance, and mean registered-target
discrepancy variance.

## Gate before real data

The controlled Gate must establish:

1. finite, permutation-equivariant profile values;
2. exact zero-discrepancy recovery of the previous predictive components and
   conditional EPIG;
3. no response access in the scoring interface;
4. finite, auditable discrepancy-aware ranking with the same representative
   safe-set rule and closed target partition.

The controlled implementation Gate now passes, including exact quadrature,
closed-loop oracle isolation, frozen-config drift rejection, P3B.10 backward
compatibility, EvidenceRegistry identity/export, static integrity, and
held-out leakage tests. This establishes implementation correctness only; it
cannot establish real-data superiority. P3C.1 may therefore run once on the
unchanged real development protocol, with held-out remaining closed.

The real-run manifest now also separates `formal_protocol_evidence` (all
registered protocol checks passed) from `formal_efficacy_evidence` (the
pre-registered positive efficacy assessment passed). A protocol-valid negative
run therefore cannot be misread as positive efficacy evidence.
