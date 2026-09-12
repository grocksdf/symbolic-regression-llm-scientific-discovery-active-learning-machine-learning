# P3M.7 decision-aligned utility design

## Purpose

P3M.6 is a valid real-development result, but its selected score is a
lower-tail CVaR of operational-class entropy reduction and its score-to-realized
gain correlation is near zero. P3M.7 therefore tests a distinct, pre-registered
utility: expected reduction in Bayes 0--1 decision risk for the frozen
operational class decision.

P3M.6 remains immutable and is not relabeled. P3M.7 is a new method identity,
configuration, checkpoint schema, and output namespace.

## Utility

Let `p` be the frozen operational-class posterior before a candidate response,
and `p_y` the posterior after response `y` under the same strict-prefix
candidate-specific predictive law.

```text
R(p)   = 1 - max_c p[c]
g(y)   = R(p) - R(p_y)
U(a)   = CVaR_alpha_lower(g(y) | action=a)
```

The action score is the robust lower-tail envelope of `U(a)` over the frozen
likelihood-power ambiguity set. The response quadrature, candidate pool,
representative guard, initial budget, acquisition budget, and rank certificate
remain unchanged from P3M.6.

No validation, held-out, observed outcome, seed, or post-hoc threshold enters
the utility. The only estimand change is the loss function from class entropy to
Bayes 0--1 decision risk.

## Required response-free correctness checks

1. `R(p)` is in `[0, 1]` for every normalized positive class posterior.
2. The prior risk is invariant across candidate actions within one query.
3. Posterior risk is computed from the same calibrated class posterior already
   used by P3M.6; no second posterior or oracle label is admitted.
4. The weighted quadrature mass is normalized before CVaR evaluation.
5. The P3M.7 checkpoint plan hashes utility identity, candidate actions, state,
   partition, quadrature order, and tail probability.
6. P3M.6 checkpoints cannot be opened by P3M.7 and vice versa.
7. Safe-set compression and predictive-law caching produce identical full
   candidate vectors to the uncompressed reference implementation.

## Pilot gate

The first real execution is a short, pre-registered pilot covering CCPP, CO,
and NOx with two fixed seeds per dataset and all registered baselines. It is a
development diagnostic, not confirmation evidence. A full confirmation matrix
is authorized only if the pilot shows positive score--realized-gain alignment,
consistent paired class-gain direction against random in every family, and
negative transfer no greater than the frozen assessment limit.

The pilot must use a new output directory and must not modify P3M.6 evidence.
