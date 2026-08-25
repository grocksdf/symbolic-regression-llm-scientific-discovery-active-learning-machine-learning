# PCPI P3G.2 heteroscedastic mainline contract

## Frozen negative predecessor

P3G.1 commit `9b57a4ee66c9f97f0f45e5d5c45d6c6ffaa2080c`
completed all 24 provenance-verified UCI calibration runs. Three crossed the
unchanged familywise PIT e-process boundary: CCPP seed `2026080706`, Gas CO
seed `2026080703`, and Gas CO seed `2026080707`. The gate correctly returned
`CALIBRATION_NO_GO`, opened no candidate response, opened no held-out response,
and ran no acquisition comparison. This negative evidence remains immutable.

The crossings were not marginal: their maximum e-values were approximately
`2.66e6`, `1.13e7`, and `5.43e3` against a threshold of `2400`. P3G.2 therefore
does not change the test, alpha allocation, seeds, budgets, order, datasets, or
failure policy.

## Root repair

P3G.1 represented structure-dependent mean discrepancy but retained one
homoscedastic observation-variance law. P3G.2 makes input-dependent noise part
of the same ordinary Bayesian generative posterior. Before any validation or
candidate response is accessed, it constructs a finite log-variance sieve
from the first three leading principal scores of the registered covariate
domain. Each score contributes fixed positive and negative `log(2)` laws, all
normalized to unit geometric-mean variance. A homoscedastic state receives
prior probability `1/2`; the other `1/2` is allocated equally across the six
heteroscedastic laws.

This construction is dataset-agnostic and response-independent. Responses may
update state probabilities only through their generative marginal likelihood.
The conditional law remains conjugate: for variance multiplier `v_i`, the
rank-one update uses predictive inflation

`v_i + d_i^T V d_i`.

The acquisition target is expanded from `(structural class, discrepancy
kernel state)` to `(structural class, discrepancy kernel state, noise-variance
state)`. No fitted acquisition weight is introduced.

The approach follows the modeling motivation of heteroscedastic GP regression,
where observation noise is input dependent, while retaining the finite exact
mixture needed by the PCPI correctness surface:

- Lázaro-Gredilla and Titsias, *Variational Heteroscedastic Gaussian Process
  Regression*, ICML 2011: <https://icml.cc/2011/papers/456_icmlpaper.pdf>
- Chaudhuri, Jain, and Natarajan, *Active Heteroscedastic Regression*, ICML
  2017: <https://proceedings.mlr.press/v70/chaudhuri17a.html>
- Sloman et al., *Bayesian Active Learning in the Presence of Nuisance
  Parameters*, UAI 2024: <https://proceedings.mlr.press/v244/sloman24a.html>

## Unchanged execution boundary

One command first executes all 24 calibration runs. If any rejects, the run
publishes one terminal `CALIBRATION_NO_GO` archive and performs zero
acquisition queries. Only a 24/24 pass creates the candidate oracle and
automatically executes all 96 matched-budget runs. Held-out remains closed,
and even a positive development result is not formal efficacy evidence.

The CSV now records PIT mean, variance, tail rates, and the three registered
basis means. These are audit outputs only and cannot alter the posterior,
gate, or acquisition score.
