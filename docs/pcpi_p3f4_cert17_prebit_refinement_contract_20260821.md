# P3F.4-CERT.17 pre-bit refinement contract

Status: **RESPONSE-FREE CONDITIONAL REFINEMENT THEOREM; ALL EXECUTION BLOCKED**

Baseline: `3f7416b3559496464c99c42d4a6078dcd3612d7a`

CERT.16 exposed a genuine obstruction: a runtime rejection when a fixed
512-bit enclosure exceeds `rho` has conditional failure probability one.  A
single fixed precision is also the wrong uniform object for the countably open,
unbounded raw-AST state space.  CERT.17 changes the numerical algorithm, not
the scientific target, failure probability, threshold grid, particle count,
island count, response, or result.

## 1. Threshold-blind refinement

For refinement round `r`, the preregistered precision is

\[
  p_r=512\,2^r.
\]

At each round an operational evaluator must return valid outward probability
boundary enclosures for the same exact target.  Consecutive enclosures are
intersected exactly.  Disjoint enclosures are a proof failure.  The algorithm
continues while the complete CERT.15 ambiguity bound exceeds the already
frozen CERT.16 allocation `rho`.

No threshold bit may be read during refinement.  The first eligible envelope
is certified before the unchanged one-shot 256-bit threshold is accessed by a
future separately authorized source.  The stopping rule therefore cannot
select a favourable random threshold.  It also does not use a scientific
result, held-out value, empirical success, tolerance, seed, or relaxed budget.

## 2. Exact grid floors

If a comparison has `J` internal decision boundaries and the fixed threshold
cell width is `2^-256`, its limiting grid contribution is

\[
  g_J=2J\,2^{-256}.
\]

For MH, `J=1`.  For one full multinomial inverse-CDF draw over `N` particles,
`J=N-1`.  CERT.17 verifies with exact rational arithmetic that both

\[
  g_1 < \rho,
  \qquad
  g_{N-1} < \rho
\]

for the frozen `N=212408` CERT.16 plan.  Thus the 256-bit grid itself does not
exhaust the comparison allocation; adaptive threshold-bit extension is neither
needed nor authorized.

## 3. Conditional termination theorem

For every fixed finite pre-threshold state, suppose the valid nested outward
boundary widths converge to zero along the registered precision schedule.
Then their ambiguity upper bounds converge to `g_J`.  Since `g_J < rho`, the
definition of convergence implies that a finite eligible round exists for
that state.

This is a pointwise finite-round result with one uniform probability
allocation.  It does not claim that one finite precision round works for all
countably many states, nor does it impose an invalid uniform runtime ceiling.

CERT.17 deliberately leaves

```text
reachable_state_evaluator_convergence_verified = false
```

because the actual CERT.13/14 Arb covariance, validated solve, log-marginal,
normalization and acceptance evaluators have not yet been refactored onto this
schedule or proved convergent for every legal finite state.  The abstract
termination theorem cannot substitute for that source-level proof.

## 4. Failure closure

- Every envelope binds the CERT.17 plan, CERT.16 integration plan, complete
  comparison coordinate, purpose, round, precision and all decision
  boundaries.
- A multinomial envelope must contain all `N-1` internal boundaries; an MH
  envelope contains exactly one.
- Cross-plan, skipped-round, wrong-precision, incomplete-boundary and disjoint
  enclosures fail closed.
- An exhausted finite prefix requests more registered precision without
  reading bits or returning partial output.
- No midpoint, nearest value, `nextafter`, tolerance, regularizer, jitter,
  response-derived schedule, precision cap, partial ancestor vector or
  successful-island subset is introduced.

## 5. Randomness and authorization boundary

CERT.17 changes only the pre-bit numerical enclosure algorithm.  It does not
turn distinct Philox addresses into mathematical product randomness.  The
external ideal-bit premise remains explicit and its implementation remains
unauthorized.

Only the standalone refinement theorem is authorized.  Operational evaluator
refinement, threshold access, product-bit materialization, island execution and
resident SMC all remain false.  No response, particle, entropy, real data,
held-out data, simulated experiment or formal experiment is accessed.

## 6. Next theorem phase

The next admissible phase must bind the actual CERT.13/14 evaluators to the
registered precision schedule and prove valid-enclosure convergence for every
legal finite state.  It must also replace the current quadratic-in-particle
normalization construction with a rigorously outward linear-time aggregation
before any operational feasibility claim.  The ideal-bit/product-law premise
must remain explicit in the statistical theorem and implementation audit.
