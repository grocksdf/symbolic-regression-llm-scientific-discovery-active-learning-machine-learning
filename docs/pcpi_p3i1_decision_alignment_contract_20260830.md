# PCPI P3I.1 decision-target alignment contract

## Scope

P3I.1 is a response-free correctness phase following the immutable P3H.9
negative result. It changes neither P3H.9 evidence nor any held-out boundary.
It repairs the probability/decision contract before another real experiment
can be designed.

The primary scientific target is the frozen operational class `C`, not all
predictive nuisance uncertainty. This follows targeted Bayesian experimental
design: information about parameters or nuisance predictions is useful only
when it reduces the registered target risk. Prediction-oriented EIG likewise
requires an explicit target distribution rather than equating arbitrary model
uncertainty with useful prediction information. Relevant primary sources are
Bickford Smith et al., *AISTATS 2023*,
<https://proceedings.mlr.press/v206/bickfordsmith23a.html>, and Sloman et al.,
*UAI 2024*, <https://proceedings.mlr.press/v244/sloman24a.html>. The finite
likelihood-power lower envelope remains an explicitly robust utility rather
than a nominal posterior; compare Go and Isaac, *UAI 2022*,
<https://proceedings.mlr.press/v180/go22a.html>.

## C1: decision-regret operational classes

For structure `s`, let

`m_s(x) = E[Y | x, s, H0]`

be its posterior Bayes action under squared prediction loss. Let `s0(x)` be
the predictive standard deviation of the complete `H0` posterior mixture,
computed once on the registered target domain and shared by every structure
pair. P3I freezes

`d_R(s,s')^2 = E_nu[((m_s(X)-m_s'(X))/s0(X))^2]`.

This is standardized excess squared-prediction regret. Unlike P3H.9's
pair-specific predictive-quantile distance, observation-noise or residual
shape alone cannot distinguish two symbolic mean laws. The common scale also
makes the distance an ordinary Euclidean RMS profile distance. Complete-link
clustering and the pre-existing budget-derived threshold remain deterministic;
the action domain, common-scale hash, threshold and class map are frozen at
`H0`.

## P3H marginal transport and information invariance

For one action, let `F` be the continuous base mixture CDF and `G` the positive
P3H residual CDF. The corrected marginal is `H(y)=G(F(y))`. Starting with a
base response `Y`, the canonical copula-preserving transport is

`T(Y) = F^-1(G^-1(F(Y)))`.

`T` is common to every structure and strictly monotone. Consequently

`I(C;T(Y)) = I(C;Y)`.

At an output PIT coordinate `u=F(T(Y))`, the corresponding base PIT is
`G(u)`. Class responsibilities must therefore be evaluated at
`F^-1(G(u))`, not at `F^-1(u)`. This pushforward preserves the base class
marginal without selecting a new KL-projected copula. The residual law is
genuinely represented in the output response distribution, but it is not
allowed to manufacture a new class-information ranking from a marginal-only
calibration choice.

P3I.1 implements the transport as
`mixture-pit-copula-preserving-monotone-transport-v1`. Its quadrature class
marginal is a convergence diagnostic, not an exact finite-order theorem. The
production class score uses the independently established base class-EIG
integral together with the transport-invariance certificate, rather than the
finite transport quadrature as a second estimator.

## C3: target-only robust utility

P3I.1 registers class logarithmic risk as the primary decision loss. For each
likelihood-power ambiguity model `eta`, the action utility is class-EIG
`I_eta(C;Y_a)`. The robust score is

`U(a) = min_eta I_eta(C;Y_a)`.

The old conditional Gaussian-moment EPIG term is identically excluded from
the primary score. This is not an empirically chosen weight: it removes a
nuisance target that was never the primary scientific estimand. Predictive
performance remains a separately reported outcome and limitation; it cannot
be added to class risk after observing which mixture gives better results.

The covariate-only representative MMD remains a response-free feasibility
constraint. If there is no representative-safe candidate, P3I fails closed
instead of changing utility. Numerical lower-envelope intervals use the
registered P3H.8 possible-maximizer resolver; primary separation and secondary
resolution remain distinct audit fields.

## Correctness decisions and result

The deterministic P3I.1 Gate verifies:

1. a nonidentity residual law changes output responses;
2. the copula pushforward preserves class information at the pullback rule;
3. the historical P3H.3 I-projection is not dispatched by P3I;
4. the class map uses one common mean-regret scale and no quantile levels;
5. the robust primary score contains class-EIG and zero conditional EPIG;
6. production source has no validation, held-out, candidate-response or old
   transformed-joint dispatch surface.

All seven unit tests and all six Gate decisions pass. On the hand-authored
finite-mixture fixture the historical projected information is
`0.5615051362`, while the copula-transport pullback information is
`0.6484352361`; its independent base reference is within `1e-4`. The
transport/pullback information invariance error is exactly zero at the
registered arithmetic path. The finite transport class-marginal quadrature
error is about `1.77e-6` and is retained rather than projected away.

The affected 63-test acquisition/P3H/integrity regression passed, and the
complete repository regression passed `736/736` with exit code zero.
`git diff --check` also passed. No historical failure was masked or relaxed.

## Authorization boundary

P3I.1 uses no simulated experiment, real dataset, validation response,
candidate response or held-out value. It establishes an estimand and source
correctness repair only. Operational execution, a new real protocol,
independent confirmation, efficacy evidence and paper superiority claims all
remain unauthorized. Before any measured-data command, the historical runner
must be composed with the new class constructor and target-only scorer under a
separate source Gate, and the reporting/assessment contract must be frozen
without inspecting held-out data.
