# P3L.2 failure audit and P3M action-conditional freeze (2026-09-07)

## Evidence boundary

The completed P3L.2 real-development run is immutable, protocol-valid negative
evidence.  Its corrected audit verifies 96/96 runs, 3072/3072 queries, the
registered information-risk score on every PCPI query, valid evidence lineage,
and a closed held-out set.  It does not demonstrate real-data advantage.

Across the two registered data families, paired uncertainty intervals for mean
gain and nAULC include zero.  Negative-transfer rates remain 0.25 for PCPI in
both families, and at least one baseline is no worse in each family.  The
selected score is nearly uncorrelated with subsequent realized gain.  The
mechanistic boundary is structural: P3L uses one global residual innovation
law, so every candidate is evaluated under the same learned misspecification
shape.  Lower-tail risk cannot identify where that misspecification occurs.

## Frozen P3M estimand

For frozen scientific class `c`, action `x`, base conditional CDF `F_c(.|x)`
and density `f_c(.|x)`, P3M defines

```
q_c(y|x) = g_x(F_c(y|x)) f_c(y|x).
```

Here `g_x` is reconstructed only from earlier pairs `(X_i, U_i)`, where `U_i`
is the raw PIT of the already revealed response under the preceding mixture
forecast.  One `g_x` is shared across all classes.  This preserves class
identifiability while allowing the predictive law to vary across candidates.
Substitution `u = F_c(y|x)` gives `integral q_c(y|x)dy = integral g_x(u)du = 1`;
therefore current class probabilities remain the marginal of a normalized
class/response joint law.

The context transform and kernel anchor use conditioning covariates only.  At
prefix size `n` and action dimension `d`, the bandwidth is

```
h_n = h_anchor n^(-1/(d+4)).
```

Consequently `h_n -> 0` and `n h_n^d -> infinity`.  Under the usual local
smoothness, positive design-density and fixed-dimension assumptions, these are
the standard kernel conditional-density consistency limits.  The correctness
Gate does not claim finite-sample calibration or real-data efficacy.

Within each local neighborhood, Gaussian RBF weights replace integer counts
in every Jeffreys/KT dyadic split.  Each split remains a proper Bernoulli
predictive distribution and the induced leaf masses sum to one.  Bandwidth,
tree depth and weights cannot inspect any response beyond the revealed prefix.

## Acquisition contract

For each visible candidate, P3M uses that candidate's own `g_x` for both the
source response measure and every class likelihood in the posterior update.
The lower-tail CVaR of frozen-class entropy reduction is then computed from
this normalized joint.  Mean information remains an audit quantity.  No
candidate, validation or held-out response enters scoring.

P3M.1 is correctness only.  It must establish normalization, strict-prefix
ordering, affine coordinate invariance, response-free bandwidth construction,
candidate discrimination, deterministic quadrature and leakage isolation.
It does not authorize a real run.  Production lifecycle, all four
likelihood-power states, checkpoint identity and supervised fail-closed runner
composition require later Gates before a new real experiment can be frozen.

## Claim boundary

P3M is a failure-informed model repair, not a result-conditioned regularizer.
Its form is fixed by the diagnosed missing conditional estimand.  P3L.2 is not
rerun, no seed or threshold changes, and no result is used to choose a
dataset-specific parameter.  Any later real comparison remains development
evidence and must preserve the existing datasets, splits, seeds, label budget,
compute accounting, representative guard, tail probability and held-out
closure.
