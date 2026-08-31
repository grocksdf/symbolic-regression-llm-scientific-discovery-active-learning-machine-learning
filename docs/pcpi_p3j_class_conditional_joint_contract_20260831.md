# PCPI P3J class-conditional joint-law contract

## Why P3J is necessary

P3I.4 is a valid negative real-development result, not an execution defect.
Its common monotone response transport cannot change mutual information and
therefore cannot change the Student-t class-EIG ranking. P3J changes the
probability model at the identified failure boundary rather than tuning a
score against the observed result.

For each frozen operational class `c`, let the current base conditional
forecast have CDF `F_c` and density `f_c`. Let `g_c` be the predictive density
of a Krichevsky--Trofimov dyadic Pólya tree trained on that class forecast's
strict-prefix raw PIT values. P3J defines

`q_c(y) = g_c(F_c(y)) f_c(y)`.

The change of variables `u = F_c(y)` gives
`integral q_c(y) dy = integral_0^1 g_c(u) du = 1`. Thus
`p(c) q_c(y)` is a normalized class/response joint law and its class marginal
is exactly the registered `p(c)`. Unlike a common marginal transport, distinct
`g_c` can change class mutual information without selecting an unidentified
copula.

This construction is related to Bayesian nonparametric predictive
calibration, but the claim here is restricted to the implemented normalized
conditional composition and its tests; it is not a general consistency or
optimality theorem. Relevant primary references include [Bassetti et al.
(2015)](https://arxiv.org/abs/1502.07246) for Bayesian nonparametric predictive
calibration and [Christensen and Ma (2020)](https://academic.oup.com/jrsssb/article/82/1/127/7056033)
for hierarchical adaptive Pólya-tree density modeling.

## Prequential and leakage boundary

Every frozen class maintains a counterfactual residual state. Before a newly
revealed response is admitted, the response is scored under every class's
current prefix forecast. Only then is its class-specific PIT appended to every
state. No latent class label or response-dependent soft assignment is used.

The initial data roles remain separate. Base-conditioning responses fit the
initial conjugate state. Residual-training responses reconstruct the
strict-prefix calibration densities. These initial residual-training
responses do not retroactively reweight the class posterior: the resulting
density estimator is frozen as part of the initial empirical-Bayes predictive
state. After acquisition starts, every newly revealed response contributes
both its base conjugate likelihood and its pre-reveal class calibration factor.

Candidate responses, validation responses, held-out data, RNG-based retries,
power selection and external response oracles have no P3J.1/P3J.2 input path.

## Acquisition and inference coherence

P3J integrates class mutual information directly under `p(c) q_c(y)` using
deterministic nested quadrature. It takes the lower envelope over the complete
registered likelihood-power family. The conditional predictive nuisance term
remains zero because the scientific target is frozen-class log risk.
Representative safety and the already frozen interval-frontier resolution are
constraints, not replacement utilities.

The same `g_c(F_c(y))` factor used in acquisition multiplies the class target
after the selected response is opened. Internally, the implementation keeps
the conjugate base target and cumulative class log-calibration factors
separate, then reconstructs the reporting weights. For the nominal `eta=1`
Bayesian target, a numerical identity check requires those class weights to
equal the direct `p(c)q_c(y)` Bayes update after every reveal.

For `eta<1`, the likelihood-power targets are generalized-Bayes sensitivity
models: their one-step powered-likelihood evidence is not the ordinary
Student-t predictive density. P3J therefore applies the same pre-reveal
calibration factor to each target's own generalized update, but does not call
the resulting weights an ordinary Bayesian posterior or assert the nominal
joint-law update identity for them. They enter only the registered robust
lower envelope; `eta=1` remains the reporting and scientific posterior.

P3J EIG estimates are deterministic approximations with safety-inflated nested
fine/coarse difference envelopes, not exact EIG or rigorous integration bounds.
A primary interval separation certificate conditional on those envelopes or the
registered response-free frontier resolver is required before selection.

## P3J.1 and P3J.2 claim boundary

P3J.1 covers algebra and estimator correctness. Its hand-authored diagnostic
fixture verifies that uniform class calibration recovers base Student-t EIG,
that a nonidentity class-conditional law can reverse a separated action
ranking, that each conditional law normalizes, and that acquisition and
posterior updating use the same factors. The fixture is not an efficacy
experiment.

P3J.2 composes the complete four-power family into a typed score--select--
reveal--advance-once lifecycle. The state erases raw initial response history
after reconstruction. A decision commits the prior state hash, selected ID and
action. Only one exactly matching reveal may advance every ambiguity model
once; stale, changed, uncertified or incomplete inputs fail closed.

Neither stage authorizes a real experiment. Operational runner integration,
runtime-cost bounds, frozen measured-data protocol, and user-only execution
remain separate future Gates. P3I.4 remains immutable negative evidence.
