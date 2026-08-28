# PCPI P3H.8 interval-frontier resolution

## Problem and boundary

Strict interval dominance is a partial decision rule. Increasing quadrature
order cannot guarantee finite termination when two utilities are equal or
closer than the registered numerical resolution. Merely increasing the maximum
order postpones the same failure and can consume unbounded computation.

P3H.8 makes the operational rule total without changing the transformed
semiparametric utility or using a response-derived fallback. It applies only
when the primary P3H maximin ranking has not separated at the already frozen
maximum order.

## Total decision rule

For every representative-safe candidate `i`, let `[L_i, U_i]` be its final
lower-envelope numerical interval and set `B = max_j L_j`. Define the possible
maximizer set

`A = {i : i is representative-safe and U_i >= B - rho}`,

where `rho` is a scale-relative machine-roundoff allowance. Every candidate
with `U_i < B - rho` is interval-dominated and cannot enter `A`. The set is
nonempty because a candidate attaining `B` has an ordered interval.

If strict interval dominance already holds, the original maximin argmax is
used unchanged. Otherwise the selector:

1. minimizes the pre-existing covariate-only augmented MMD inside `A`;
2. retains all machine-indistinguishable MMD minimizers;
3. selects the smallest registered global candidate ID.

This defines a deterministic lexicographic operational target: transformed
maximin information up to the registered numerical partial order,
representativeness inside the unresolved frontier, then immutable ID order.
It is not posterior-variance fallback and does not relabel the secondary choice
as a uniquely resolved EIG maximizer.

## Leakage and anti-hardcoding properties

The resolver receives only the utility intervals, representative-safe mask,
augmented MMD values and registered candidate IDs. It receives no candidate
target, validation response, held-out value, dataset name, seed-specific rule,
realized acquisition gain or empirical tolerance. The same rule is permutation
equivariant before the final global-ID tie break and applies to every finite
candidate bank.

Every query records whether primary dominance held, the possible-maximizer IDs,
whether secondary resolution was used, the secondary-admissible IDs and the
selected ID. Thus primary numerical resolution and total operational validity
remain separately reportable.

## Computation

Nested refinement now reuses the preceding fine quadrature result as the next
coarse result. This is algebraically identical to the old computation because
the node order and fine/coarse error formula are unchanged. It removes one
duplicated evaluation at every refinement look after the first, reducing the
worst registered refinement work by about one third without changing scores,
intervals or candidate choices.

## Primary-source relationship

Confidence-bound action elimination is established in Even-Dar, Mannor and
Mansour, *JMLR* 7 (2006),
<https://www.jmlr.org/papers/v7/evendar06a.html>. P3H.8 uses only the algebraic
interval-dominance idea; its fine/coarse quadrature envelopes are deterministic
numerical diagnostics, not stochastic confidence intervals, so none of that
paper's PAC or sample-complexity guarantees is imported. Gonnet's review of
adaptive-quadrature error estimation, <https://arxiv.org/abs/1003.4629>, also
motivates keeping numerical error diagnostics distinct from probabilistic
coverage claims. No external best-arm implementation is composed into PCPI;
the production resolver is the small audited function in
`hypothesis_mvp/pcpi/real_acquisition.py`.
