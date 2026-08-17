# P3F.3 scalable open-target particle approximation contract

Decision: **ADMISSIBLE AS CORRECTNESS-ONLY WORK. REAL DATA, ACQUISITION,
HELD-OUT, MOTIF, VED, AND EFFICACY REMAIN BLOCKED.**

This contract is the next incremental step after P3F.2a--c. It does not
silently replace the P3F.2 target with a broader noise family, measurement-error
model, nonlinear-constant parameterization, transcendental equivalence rule, or
response-adapted grammar. Those are future target versions and require their own
contracts and exact references.

## 1. Frozen target

The target remains the registered P3F.2 countably-open typed AST prior,
conditioned on the explicit finite node-count slice for the exact reference:

\[
  \pi(\zeta\mid D) \propto p_{\mathcal G}(S)
  p(\theta_S,d,\lambda,\sigma^2\mid S)\,
  p(D\mid S,\theta_S,d,\lambda,\sigma^2).
\]

The registered state is the one-amplitude dimensionless algebraic expression,
structure-wise projected discrepancy, Gaussian homoscedastic
Normal--Inverse-Gamma noise, and no measurement-error state. The omitted
open-prior tail is always reported; a finite reference slice must never be
labelled as the complete open posterior.

When running against the P3F.2 exact reference, the particle configuration's
finite node-count cutoff must equal the contract's registered reference slice
maximum. A different cutoff is a new target version, not an implementation
parameter.

The proposal is not the target. No response, dataset identifier, benchmark
answer, held-out state, failed seed, validation residual, or result-derived
threshold may enter the target contract, grammar prior, equivalence map, or
proposal support.

## 2. Scalable approximation

Implement one particle engine over the same latent component semantics as the
exact reference. The engine may use:

- countably-open typed grammar expansion with explicit truncation metadata;
- local typed subtree birth/death/replacement moves;
- discrepancy spike/kernel-state transitions;
- adaptive likelihood bridges selected by conditional ESS;
- systematic resampling with complete local and root genealogy;
- rejuvenation kernels that leave every registered bridge target invariant; and
- a proposal mixture with a nonzero base grammar component.

For a proposal mixture

\[
q(\zeta'\mid\zeta)=
\rho_0q_{\rm grammar}(\zeta'\mid\zeta)+
\rho_1q_{\rm local}(\zeta'\mid\zeta)+
\rho_2q_{\rm learned}(\zeta'\mid\zeta),
\]

the post-filtering probability must be evaluable and included in every MH or
importance correction. A learned or LLM proposal is not admitted until its
actual filtered probability is auditable. Until then, only the registered
grammar and local proposals are allowed. The base grammar weight must remain
strictly positive so learned mode collapse cannot remove target support.

For bridge temperatures

\[
0=\beta_0<\beta_1<\cdots<\beta_J=1,
\qquad
\pi_{\beta_j}(\zeta)\propto p(\zeta)p(D\mid\zeta)^{\beta_j},
\]

CESS chooses the numerical path only. Every rejuvenation kernel must satisfy
\(\pi_{\beta_j}K_j=\pi_{\beta_j}\). No generalized likelihood power is
introduced in P3F.3; update-coherence changes require a separate target version.

The registered sequential path uses the equivalent prequential form: previous
rows are fixed at power one and only the next row is bridged from power zero to
one. Fractional powers exist only along this Feynman--Kac path; the terminal
state commits the ordinary likelihood and is the only state exposed for
posterior prediction.

## 3. Exact-reference Gate

The scalable engine is admissible only after comparison with the existing
exhaustive P3F.2 reference on multiple small, hand-constructed fixtures. The
Gate must check, at minimum:

1. normalized prior and explicit open-prior tail mass;
2. raw-AST and equivalence-class posterior mass conservation;
3. posterior probabilities and predictive functionals against the exact slice;
4. batch/sequential evidence telescoping;
5. log-normalizer dispersion and particle-count convergence;
6. ESS, CESS, weight entropy, distinct root ancestors, root entropy, and
   coalescence diagnostics;
7. move-wise proposal/acceptance counts and semantic jump distance;
8. proposal invariance: distinct evaluable proposals converge to the same target;
9. row-order equivariance of the batch posterior; and
10. inference-to-decision perturbation, with a registered numerical error
    certificate rather than a silent utility fallback.

The exact finite slice is the reference truth for this Gate. A pass supports
only scalable inference correctness. It does not support search coverage,
predictive calibration on real data, acquisition advantage, or scientific-law
discovery.

## 4. Implementation boundaries

- Extend the canonical `hypothesis_mvp.pcpi.open_target` path; do not create a
  second production posterior or a `legacy/final/v2` implementation.
- Reuse exact target evaluators and equivalence aggregation from P3F.2 where
  semantics are unchanged.
- Keep proposal generation, target evaluation, evidence logging, predictive
  computation, and acquisition interfaces separate.
- Record source/config/fixture hashes, seeds, particle counts, bridge schedule,
  proposal identity, held-out state, and all failure events in the canonical
  evidence namespace.
- Never import real-data, acquisition, or held-out modules from the P3F.3
  correctness runner.

## 5. No-go conditions

Stop P3F.3 if any of the following occurs:

- particle and exact-reference posterior differ beyond the registered Monte
  Carlo/error envelope;
- two valid proposals produce materially different posterior functionals;
- equivalence-class mass is not conserved;
- evidence does not telescope;
- row order changes the batch target;
- a proposal has no auditable post-filtering probability;
- a learned/LLM proposal changes target support after responses are observed; or
- implementation requires a dataset-specific rule, result-derived threshold,
  extra budget, or held-out access.

After a No-Go, diagnose target specification, generative model, proposal
support, or numerical approximation. Do not repair the result by selecting
seeds, adding post-result regularization, or changing the claim boundary.

## 6. Downstream order

Only after this Gate passes may a separate predictive-calibration contract be
written. Only after that calibration Gate passes may a new held-out-closed real
development protocol be considered. No P3F.3 result authorizes acquisition,
motif transfer, VED, held-out confirmation, or superiority claims.
