# PCPI gap and rebuild plan

## Architecture

Only `hypothesis_mvp.pcpi` owns posterior state. The dependency direction is:

`proposal catalog -> stochastic kernel -> posterior -> predictive classes -> acquisition`

LLM, engine, and agent components cannot create inferential weights.

## Frozen evidence

P2A.1, P2B, P3A.2, and P3B.6--P3B.10 controlled Gates establish only
their bounded correctness claims. P3B.9 returned real evidence is protocol
valid but fails efficacy. The P3B.10 real rerun is likewise protocol-valid
negative development evidence. None establishes broad real scientific-discovery
superiority.

## P3 — P3B.10 representative-safe maximin joint acquisition

P3B.9 returned 96/96 real policy runs with closed held-out and a valid
representative guard throughout, but its CCPP frozen-class gain remains
significantly worse than random. The selected joint score is essentially
uncorrelated with realized local class gain.

P3B.10 preserves the MMD safe set, nominal reporting posterior, classes,
budgets, seeds, splits, baselines, and efficacy rules. PCPI ranking alone uses
the minimum joint score over the four likelihood powers frozen before this
repair. The family receives the same observed history, bank, class map,
candidate covariates, and target domain.

The controlled Gate passed all 27 joint, representative, and finite-maximin
decisions together with the focused regression, static, and leakage suite. The
allowed real rerun completed under the frozen protocol, but failed the
pre-registered efficacy assessment. The next repair target is the general
posterior/decision model, not a data-specific threshold, direction, seed, or
regularizer.

P3C.1 is now the isolated general repair. It adds a discrepancy variance
estimated from acquired-history posterior residual sufficient statistics and
covariate support, while retaining the P3B.10 nominal posterior and all frozen
real-data roles. Its controlled implementation, exact-reference, registry,
static, and leakage Gates pass. One held-out-closed real-development run is
eligible; no P3C.1 efficacy evidence exists until that run is returned and
audited.

## P4--P7

P4 remains optional and requires proposal-target invariance. P5 requires a
successful P3C.1 real audit plus broader matched-budget evidence and the 2x2
factorial. P6 is one-shot held-out confirmation after algorithm freeze. P7 VED
starts only after all preceding Gates.

## Stop rules

- official data hash or held-out boundary failure: invalidate immediately;
- missing or substituted seed: Gate failure;
- P2A.1/P2B/P3A.2 correctness failure: repair that general method first;
- P3C.1 controlled failure: do not run real acquisition;
- P3C.1 real efficacy failure: stay in P3 and reassess the general statistical
  or decision model, never a dataset formula, response direction, regex,
  result-derived threshold, or held-out value;
- motif target-invariance failure: remove motif from the core.
