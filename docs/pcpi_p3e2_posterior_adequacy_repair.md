# P3E.2 union-orthogonal posterior-adequacy Gate

Status: **CORRECTNESS FIXTURE PASS; REAL ADEQUACY AND EFFICACY UNTESTED**

## Why P3E.2 exists

P3D.2 is valid negative real-development evidence. Its numerical decision
contract worked, yet CCPP had PCPI-minus-random frozen-class gain `-0.253805`
with 95% interval `[-0.482951,-0.024660]`. Every CCPP seed used ordinary Bayes
(`eta=1`), so P3E.1's generalized-update correction cannot explain that
failure.

The finite-bank likelihood assumes a candidate symbolic mean plus independent,
homoscedastic Gaussian noise. P3C.1 did not change that likelihood or the
structure posterior: it estimated one nonnegative residual-excess variance and
used it only to inflate acquisition predictive components. On CCPP the
registered excess was essentially zero. A conditional residual pattern can
therefore remain invisible to a scalar variance check even when the mean model
is wrong.

## Single task-independent repair

Let `D_s` be the frozen registered-domain design matrix of structure `s`, and
let

\[
\mathcal U=\operatorname{span}([D_1,\ldots,D_S]).
\]

P3E.2 constructs an RBF covariance `K` from registered covariates only. Its
bandwidth is the median positive squared pairwise distance after response-free
coordinate standardization. If `P_U` is the orthogonal projector onto
`mathcal U`, define

\[
K_\perp=(I-P_\mathcal U)K(I-P_\mathcal U).
\]

The retained positive eigenspace of `K_perp` supplies a discrepancy design
`B`, with numerically negligible eigenmodes removed at the registered relative
tolerance. Hence `D_s^T B=0` for every candidate structure on the frozen
domain. The augmented model is

\[
y=D_s\beta_s+B\gamma+\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,\sigma^2I),
\]

with conjugate zero-mean Gaussian priors on `beta_s` and `gamma`, an
inverse-Gamma noise prior, and a spike/null indicator for whether `gamma` is
present. The basis cannot use any response, dataset name, target name, formula
answer, validation label, or held-out value.

This is an identifiability repair, not a claim that an orthogonal Gaussian
process is novel. Orthogonal stochastic residuals are established methodology;
see [Plumlee and Joseph (2018)](https://doi.org/10.5705/ss.202015.0404). The
project-specific question is whether such a generative alternative can be
coupled to a fail-closed acquisition evidence contract without relabelling it
as scientific superiority.

## Prequential adequacy e-process

Let `m_0(y_1:t)` be the exact structure-averaged marginal likelihood under the
nominal spike and `m_1(y_1:t)` the corresponding marginal under the orthogonal
discrepancy slab. P3E.2 records

\[
E_t=\frac{m_1(y_{1:t})}{m_0(y_{1:t})}
=\prod_{i=1}^t
\frac{m_1(y_i\mid y_{<i})}{m_0(y_i\mid y_{<i})}.
\]

Conditional on the frozen, response-free action order and the declared nominal
marginal model, `E_t` is a nonnegative unit-initialized martingale. Ville's
inequality therefore gives

\[
\Pr_0\!\left(\sup_t E_t\ge 1/\alpha\right)\le\alpha.
\]

This is the standard Bayes-factor/test-martingale connection, not a new
e-process theorem; see [Shafer et al. (2011)](https://doi.org/10.1214/10-STS347)
and [Ramdas et al. (2023)](https://doi.org/10.1214/23-STS894).

At the frozen correctness level `alpha=0.01`. Crossing `E_t>=100` makes the
nominal posterior ineligible for targeted acquisition and forces the registered
reference-only mode. Non-crossing means only “not rejected by this registered
alternative”; it is not proof that the real posterior is adequate.

## Exact fixture result

The fixture has a 16-point finite registered domain and two candidate designs:
constant and linear. The RBF discrepancy is computed without responses and is
orthogonal to their union. Two deterministic cases are evaluated:

| Case | log Bayes factor | Gate decision |
|---|---:|---|
| Exact linear null | -0.713366 | nominal remains eligible |
| Linear plus registered orthogonal residual | 4.936581 | reference-only at round 16 |

The structured-residual Bayes factor is approximately `139.29`, above the
registered threshold `100`. The Gate also verifies every component marginal
likelihood against an independent observation-covariance calculation,
posterior normalization, domain-permutation invariance, exact telescoping of
predictive likelihood ratios, structure-coefficient invariance,
malformed-input rejection, and absence from every real runtime. All 11 frozen
diagnostic decisions pass.

These fixed outcomes are algebraic correctness fixtures. They are not
simulated efficacy data, do not count as a dataset family, and cannot enter a
real-performance table.

## Current boundary and required next evidence

P3E.2 has not accessed CCPP or Gas data and has not tested whether their
nominal posterior is rejected. It has not selected a real discrepancy prior,
shown real predictive improvement, or authorized the augmented posterior for
acquisition. Therefore:

- `formal_correctness_evidence=true` for the finite Gate only;
- `formal_real_posterior_adequacy_evidence=false`;
- `formal_efficacy_evidence=false`;
- held-out remains unavailable;
- no new real acquisition run is authorized.

The next admissible experiment is a separately frozen, initial-development-only
real posterior-adequacy diagnostic. It must use the existing official data
roles and fixed response order, report the entire e-process for every seed and
family, and make no acquisition comparison. Only after that audit may we decide
whether the augmented posterior deserves a new exact predictive-calibration
Gate and, later, one preregistered real acquisition run.
