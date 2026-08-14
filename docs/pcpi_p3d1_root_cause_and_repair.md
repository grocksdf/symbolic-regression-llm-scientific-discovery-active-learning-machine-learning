# P3D.1 posterior/decision root-cause matrix and repair contract

Status: **P3D.1 controlled Gate passed; P3D.2 real protocol passed but efficacy
failed; P3E.1 supersedes the generalized-update interpretation**
Stage: `P3D.1`  
Method: **certified reference-dominance acquisition (CRDA)**  
Primary estimand: the initial-frozen operational predictive class `C0`

Historical note: this document records the repair chosen before P3D.2 was run.
The returned P3D.2 result is audited in
`docs/pcpi_p3d2_result_audit_20260814.md`. For `eta<1`, its ordinary class-EIG
score is not the expected entropy reduction of the implemented generalized
update; the corrected root-cause matrix is
`docs/pcpi_p3e1_root_cause_and_update_coherence.md`.

## 1. Evidence boundary

P3C.1 completed 96/96 matched-budget policy runs with zero failures. Its
97-event EvidenceRegistry and all nine evidence-export hashes verified, and
held-out remained closed. The result is nevertheless
`REAL_ADVANTAGE_NOT_DEMONSTRATED`: CCPP frozen-class gain relative to random is
`-0.28378374200721335`, with 95% interval
`[-0.5046926179404461, -0.06287486607398057]` and 7/8 negative-transfer seeds;
the grouped Gas-family difference is `0.06591138158268409`, with interval
`[-0.22906202511823345, 0.3608847882836016]`.

These outcomes identify a failed method candidate, not a failed protocol. They
must not be used to choose a new seed, threshold, likelihood power, discrepancy
kernel, formula family, or dataset-specific branch. This document therefore
separates structural facts visible in the frozen method from hypotheses that
would require new development evidence.

## 2. Root-cause matrix

| Layer | Candidate mechanism | Status from current evidence | Consequence | P3D.1 response |
|---|---|---|---|---|
| Estimand | Frozen-class information may have little action-dependent headroom in some posterior states. Since `I(C0; Ya | Ht) <= H(C0 | Ht)`, a collapsed or nearly flat target cannot support reliable directed acquisition. | Established as a general capacity bound; low capacity occurred in earlier CCPP stages, but its exact P3C.1 round-wise role is not identified by the aggregate audit. | Optimizing numerical differences near zero can select an extreme action without meaningful primary-target benefit. | No fitted entropy threshold. A directed action must instead certify improvement over the registered reference policy; flat or zero-capacity utilities automatically return to the reference.
| Decision target | P3B.8--P3C.1 rank a joint class--predictive score, while the primary efficacy endpoint is frozen-class gain. | Structural fact in code and contract. | A candidate can win through the predictive term without improving the primary class target. | P3D.1 uses class-EIG alone for the primary handover Gate. Predictive risk remains a separately reported endpoint, not an additive term chosen without a registered loss.
| Decision fallback | An uncertified joint-EIG ranking switches to posterior epistemic variance. | Structural fact in code. | Numerical uncertainty changes the scientific objective instead of abstaining from an unsupported decision. | An uncertified decision executes the registered reference policy. It is never relabelled EIG.
| Numerical approximation | Existing fine/coarse Gauss--Jacobi envelopes are asymptotic diagnostics rather than rigorous finite-error bounds. | Established limitation. | Their interval-dominance label cannot by itself support a mathematical no-harm theorem on real runs. | P3D.1 proves the rule conditional on valid intervals. P3D.2 replaces the fine/coarse envelope with analytic information-inequality bounds and checks them against independent adaptive quadrature on continuous correctness fixtures. Numerical Student-t special functions remain ordinary floating-point evaluations, not formal interval arithmetic.
| Predictive utility | The class-conditional predictive term is a Gaussian-moment surrogate for finite Student-t mixtures. | Structural fact; exact only under the matched Gaussian case. | Joint-score rankings can differ from the intended `I(C0,Y*;Ya)` ranking. | The surrogate is excluded from the P3D.1 primary decision. Exact joint information remains a separate future estimator problem.
| Posterior model | A finite polynomial bank with one generalized-likelihood power and scalar noise/discrepancy mechanisms may omit regimes, covariates, heteroskedasticity, correlated bias, or cross-target structure. | Plausible and consistent with the failure, but the omitted mechanism is not identified by P3C.1. | Model-based EIG can be confidently wrong under the real sampling law. | P3D.1 makes no posterior repair claim and no real-world safety claim. A future generative repair must be independently specified and exact-reference tested rather than tuned from CCPP/Gas outcomes.
| Robustness set | P3B.10 varies only four likelihood powers; P3C.1 adds independent scalar variance. | Structural fact and documented limitation. | The lower envelope does not cover general likelihood or structural misspecification. | P3D.1 does not call this ambiguity set a universal robustness set. Its correctness fixture is posterior-family agnostic.
| Representativeness | The RBF-MMD guard controls covariate discrepancy, not posterior calibration or class-information gain. | P3B.9 guard was active on CCPP while negative transfer remained. | Covariate coverage is insufficient as an efficacy guarantee. | The new handover compares the declared scientific utility directly with a registered action policy; it does not add another geometry guard.
| Evidence semantics | A protocol-valid real run can still fail efficacy. | Established by P3B.2--P3C.1. | Correct implementation cannot be promoted as superiority. | The P3D.1 fixture is labelled `inference_correctness_diagnostic_fixture`; held-out and all real data are unavailable.

## 3. Why the repair is decision-theoretic

Let the visible candidate set at round `t` be `At` and let

\[
U_t(a)=I(C_0;Y_a\mid H_t)
\]

be the primary class-EIG under the frozen target posterior. Before the run,
register a response-free reference policy `q_t` over the same visible
candidates. For the matched-random comparison,

\[
q_t(a)=1/|\mathcal A_t|.
\]

Suppose a numerical procedure returns simultaneous valid bounds

\[
L_t(a)\le U_t(a)\le R_t(a),\qquad a\in\mathcal A_t.
\]

The reference-policy bounds are obtained without another estimator:

\[
L_t(q)=\sum_a q_t(a)L_t(a),\qquad
R_t(q)=\sum_a q_t(a)R_t(a).
\]

Choose the candidate with the largest lower bound,

\[
\widehat a_t=\arg\max_a L_t(a),
\]

using original candidate identity for deterministic ties. The CRDA handover is

\[
A_{t+1}\sim
\begin{cases}
\delta_{\widehat a_t},&L_t(\widehat a_t)>R_t(q)+\tau_{\rm num},\\
q_t,&\text{otherwise},
\end{cases}
\]

where `tau_num` is a scale-aware floating-point allowance, not a fitted
scientific threshold.

### Proposition 1: model-relative reference dominance

If all displayed bounds contain their utilities, then in the targeted branch

\[
U_t(\widehat a_t)>U_t(q),\qquad
U_t(q)=\sum_a q_t(a)U_t(a).
\]

In the reference branch, expected utility equals `U_t(q)`. Consequently,

\[
\mathbb E[U_t(A_{t+1})\mid H_t]\ge U_t(q).
\]

The proof is immediate: containment gives
`U_t(ahat) >= L_t(ahat) > R_t(q) >= U_t(q)` in the targeted branch, and the
reference branch samples from `q_t` itself.

This is not a guarantee against real-world negative transfer under model
misspecification. It certifies only that numerical approximation has not
authorized a target-seeking decision worse than the registered reference under
the frozen model and estimand.

### Proposition 2: automatic zero-capacity abstention

Because `U_t(a) <= H(C0 | Ht)`, a one-class posterior has `U_t(a)=0` for every
action. Valid bounds then cannot establish strict dominance over `q_t`, so CRDA
executes the reference policy without an entropy threshold or dataset branch.

## 4. Correctness Gate

P3D.1 may pass only on an exactly enumerable finite discrete fixture. The Gate
must verify all of the following:

1. exact class-EIG agrees with direct entropy reduction;
2. exact class-EIG is nonnegative and bounded by class entropy;
3. registered reference utility and its bounds equal probability-weighted
   action values and bounds;
4. a separated informative action triggers targeted handover;
5. overlapping intervals trigger the reference policy;
6. a one-class zero-capacity state triggers the reference policy;
7. every targeted decision satisfies exact utility dominance over the
   reference;
8. reference fallback is deterministic for a fixed seed and changes only with
   the registered seed;
9. class-label, outcome-label, and candidate-order permutations preserve the
   decision after mapping back to stable candidate identities;
10. malformed probabilities, non-containing intervals, duplicate candidate
    identities, and non-finite values fail closed;
11. the CLI exposes no data root, validation role, held-out role, real dataset,
    LLM, motif, or discovery runtime;
12. evidence and exported diagnostics retain the
    `inference_correctness_diagnostic_fixture` claim boundary.

The clean-Git Gate passed all 14/14 frozen decisions at source commit
`5d71f588398daac3a7c8d982ec3eac0b5834d73c`; see
`docs/pcpi_p3d1_result_audit_20260814.md`. Passing this Gate authorizes only the
statement that the CRDA handover is
implemented consistently with Propositions 1--2 on an exact fixture. It does
did not by itself authorize integration into the real acquisition loop or a
CCPP/Gas efficacy claim. P3D.2 separately freezes that integration and still
requires a new measured-pool run before any empirical conclusion.

## 5. P3D.2 production interval construction

For each action, P3D.2 registers probability levels
`(0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95)`. They define deterministic
thresholds from the action's response-free predictive mixture mean and
standard deviation. If `Q_a(Y_a)` denotes the resulting bin, data processing
gives the lower bound

\[
L_t(a)=I(C_0;Q_a(Y_a)\mid H_t)\le I(C_0;Y_a\mid H_t).
\]

Let `S` index the finite structure bank and let each conditional predictive
component be Student-t with scale `sigma_s(a)` and degrees of freedom `nu_s`.
Maximum entropy at fixed variance and concavity of differential entropy give

\[
h(Y_a)\le \tfrac12\log(2\pi e\,\operatorname{Var}(Y_a)),
\qquad
h(Y_a\mid C_0)\ge \sum_s p_t(s)h(t_{\nu_s,\sigma_s(a)}).
\]

Therefore

\[
R_t(a)=\min\!\left\{H(C_0\mid H_t),
\tfrac12\log(2\pi e\,\operatorname{Var}(Y_a))
-\sum_s p_t(s)h(t_{\nu_s,\sigma_s(a)})\right\}
\]

is an upper bound in exact arithmetic. The implementation evaluates Student-t
CDFs and entropy terms in floating point and expands outward by the frozen
`1e-10` tolerance. Correctness tests verify `L <= exact class-EIG <= R` against
independent adaptive quadrature, positive-affine response invariance, one-class
zero capacity, stable candidate permutation behavior, and malformed-input
failure. This is strong numerical validation, not a proof that SciPy special
functions are correctly rounded.

P3D.2 registers the uniform distribution over currently visible candidate
identifiers as `q_t`. It applies no P3B/P3C EPIG, MMD, likelihood-power maximin,
discrepancy, or epistemic-fallback term. Implementation commit `cd05e1a` passed
the expanded 199-test regression and the production static audit. No real P3D.2
result is currently attached.

## 6. Relation to prior work and novelty boundary

Robust EIG based on ambiguity sets studies sensitivity of information utility
to prior or model perturbations, including Go and Isaac's
[REIG](https://proceedings.mlr.press/v180/go22a/go22a.pdf) and the recent
[maximin robust BED](https://arxiv.org/abs/2603.14094). Those works motivate
careful separation of model robustness from numerical error. P3D.1 does not
claim a new universal ambiguity set.

Conservative exploration and safe policy improvement return to a baseline
when an alternative is unsupported; representative examples include
[Katariya et al. (AISTATS 2019)](https://proceedings.mlr.press/v89/katariya19a.html)
and
[Laroche et al. (ICML 2019)](https://proceedings.mlr.press/v97/laroche19a.html).
P3D.1 specializes this principle to Bayesian experimental design with a
frozen scientific estimand and separates three objects that earlier PCPI stages
mixed: target utility, numerical certification, and the reference action
policy.

The AISTATS contribution is therefore not “a safe wrapper.” It is the combined
statistical contract: operational-class Bayes risk, target-correct posterior
approximation, and reference-relative certified handover with explicit
abstention semantics. Any real-data claim still requires the frozen P3D.2
matched-budget experiment to be executed on the official local measurements
and audited without changing the method contract.
