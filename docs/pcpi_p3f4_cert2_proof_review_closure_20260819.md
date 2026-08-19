# P3F.4-CERT.2 proof-review closure

Status: **STATIC DEVELOPMENT AUTHORIZED; RESIDENT INTEGRATION BLOCKED**

Reviewed source baseline:
`eeb1fbaf1b9e82568caf1ed9c341d3fb114d30d4`

This review is independent of the implementation added after it.  It checks the
algebra and the current target source, narrows one state-space claim, and
authorizes only a response-energy static development layer.  It does not alter
the permanent `P3F.4-CERT.CF.1` NO-GO or authorize new confirmatory responses.

## 1. Review decision

The RE-1 and RE-2 envelope/tail results are valid for the registered zero-mean
Gaussian/NIG target.  The bridge lower bound is valid for every non-negative
likelihood-power vector, including `2 * beta_next - beta_previous > 1`.

The MK-1 independence anchor is valid on the partial quotient state space that
collapses raw ASTs of size at most `J` into exact polynomial-semantic states and
leaves larger raw ASTs explicit.  Its proposal is normalized and its
minorization lower bound is valid.

The practical composition statement requires an additional condition that was
implicit in the contract:

> The envelope anchor and every composed local/RJ kernel must be Markov kernels
> on the same measurable state space and must share the same invariant target.

The existing resident kernels act on raw AST states, whereas MK-1 is currently
specified on a hybrid semantic-core/raw-tail state space.  Separate invariance
on those two representations is not enough.  Before resident integration, one
of the following must be proved and implemented:

1. an exact lift of the hybrid anchor to the raw-AST target, including exact
   conditional sampling of a raw AST inside a semantic core class; or
2. a push-forward of every local/RJ kernel to the hybrid target together with a
   lumpability/intertwining proof.

CERT.2 development therefore registers the macro-kernel as **anchor-only on the
hybrid state space**.  No current raw-AST resident kernel is composed with it.

## 2. Checklist closure

### A. Target identity — closed

- [x] The raw grammar, operators, countably open support, and geometric size
  prior remain those of `pcpi-p3f2-countably-open-typed-grammar-v1`.
- [x] `semantic_multiplicity_shells` conserves the exact analytic raw-AST count
  in every shell; the quotient retains raw multiplicities and unnormalized
  grammar prior mass.
- [x] The certification layer does not redefine the operational predictive
  estimand.  Its TV result is exposed only for bounded class-probability/CDF
  functionals.
- [x] The registered coefficient mean is exactly zero.  Discrepancy
  coefficients also have zero prior mean.
- [x] Registered coefficient and discrepancy precisions are strictly positive,
  hence every component prior precision is positive definite.
- [x] All powers accepted by the layer are finite and non-negative.
- [x] Spike/slab and kernel-state weights are checked to sum to one.

### B. RE-1 algebra — closed

- [x] Direct conjugate integration gives
  `A_nu |V|^(-1/2) (b0 + u^T V^(-1) u / 2)^(-c_nu)`.
- [x] The matrix determinant lemma gives the registered determinant ratio.
- [x] The residual quadratic is `u^T V^(-1) u`.
- [x] With `H = V^(-1)` and `t = e^T H e`, the Schur complement gives
  `|H| <= t` because `0 < H <= I`.
- [x] The scalar maximizer is `t_star = 1` at zero response energy and otherwise
  `min(1, b0 / ((c_nu - 1/2) R))`.
- [x] The derivation needs only non-negative powers and therefore covers
  fractional and second-moment powers above one.
- [x] The response-energy envelope is pointwise no larger than the CERT.1 flat
  envelope.
- [x] The implementation obligation is log-domain arithmetic with explicit
  finite, underflow, and target-assumption checks.

### C. Tail and bridge certificates — closed for static development

- [x] Exact core evidence uses the original unnormalized grammar prior mass.
- [x] At cutoff `J`, the unresolved prior mass is exactly `rho^J`.
- [x] Development registers the fixed cutoff schedule `[17]`; no
  response-selected escalation is permitted.
- [x] The posterior-tail ceiling remains `0.01`.
- [x] Only bounded class-probability/CDF consequences are reported; pointwise
  predictive-density error is not claimed.
- [x] RE normalizer uppers are used for both denominator moments of every bridge
  lower bound.
- [x] A bridge that cannot take a certified positive registered step fails;
  there is no forced terminal step.

### D. Practical-kernel mixing — narrowed and closed only for anchor scope

- [x] On the hybrid state space, core mass `Z_J/C` plus tail mass `U_J/C`
  equals one exactly.
- [x] The unresolved-size draw is the conditional geometric law
  `J + Geometric(1-rho)`; no finite fallback is allowed.
- [x] Conditional on size, the existing recursive raw-AST sampler is uniform on
  the analytic grammar shell, so its exact prior probability is auditable.
- [x] Independence-MH acceptance uses the exact target and proposal log masses;
  no uncorrected envelope draw is a posterior sample.
- [x] The static macro-kernel contains the envelope anchor only.  Local/RJ
  forward/reverse support, ratios, Jacobians, and invariance are not claimed
  and remain an integration Gate.
- [x] One macro-sweep means one anchor transition and its registered target
  evaluation cost.
- [x] A failed tail certificate is the single root blocker; the derived anchor
  mixing status is `blocked_by_tail_certificate`, not a second scientific
  failure.
- [x] Raw/hybrid kernel composition is explicitly blocked pending a lift or
  lumpability/intertwining proof.

### E. Evidence and workflow — closed for this implementation role

- [x] The CERT.CF.1 config, runner, tests, archive identity, and formal NO-GO
  remain immutable.
- [x] AF--AI responses may appear only as labelled postmortem development
  diagnostics.
- [x] No threshold, seed, cutoff, frequency, or resource budget is selected to
  make AF pass.  Tail `0.01`, TV `0.01`, relative-ESS `0.8`, cutoff `17`, grid
  `1/32`, and 64-step bridge ceiling are inherited unchanged.
- [x] No new confirmatory fixture, coefficient, action grid, seed, or response
  is registered or materialized in this phase.
- [x] Any future freeze must be committed and pushed before response
  materialization; that future action remains blocked.
- [x] User commands must use
  `D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe` explicitly.
- [x] Development output records source identity, clean state, configuration
  and dependency hashes, interpreter/package identity, and claim boundary.
- [x] Output paths fail closed if they already exist.

## 3. Authorized implementation surface

The following are authorized:

- a log-domain response-energy envelope with explicit zero-mean target checks;
- a wrapper around the existing exact semantic-core evidence calculation;
- RE tail, bridge, normalized-anchor, and dependency-aware decisions;
- an exact conditional raw-tail prior sampler used only for correctness tests;
- response-free algebraic tests and labelled AF--AI postmortem diagnostics;
- provenance-complete, non-overwriting development output.

The following remain unauthorized:

- composition with the existing raw-AST resident SMC or any local/RJ kernel;
- a new unseen confirmatory bank or response materialization;
- predictive-calibration, real-data, acquisition, sealed-test, efficacy,
  scientific-discovery, or submission-level superiority claims.
