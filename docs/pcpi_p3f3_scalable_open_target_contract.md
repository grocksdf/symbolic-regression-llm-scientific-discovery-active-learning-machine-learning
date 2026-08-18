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
- an explicitly registered unbiased resampler (systematic, stratified, or
  residual) with complete local and root genealogy;
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

The correctness implementation registers two auditable finite-slice kernels:
prior-independence and complete-uniform. Both include their exact forward and
reverse probabilities in the MH ratio; a deterministic finite-support
certificate checks row stochasticity, detailed balance, and stationarity over
every integer prequential prefix. The CESS scheduler never inserts a
budget-sized beta increment. If a previous fractional bridge leaves the global
ESS below the registered CESS target, the next bridge begins with one explicit
resampling event; the registered resampling kernel and that event are included
in that bridge's genealogy record. If the registered bridge budget still cannot
reach the terminal state under the CESS path, it fails closed and records a
NO-GO rather than changing the target path.

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

## 4. Fidelity audit boundary

The separate `open_target_particle_fidelity_audit` runner freezes particle
counts `[128, 512, 2048]`, the two registered proposal kernels, and three
response-free seeds. It reports raw-AST, equivalence-class, predictive, and
log-evidence error against the same hand-constructed exact slice, together
with paired proposal differences and descriptive count convergence.

This audit is diagnostic-only until an error envelope is preregistered. Its
output cannot authorize calibration, acquisition, held-out access, real-data
efficacy, or discovery. A larger particle count or a favorable proposal
comparison must be reported as evidence about approximation fidelity, not as a
post-hoc Gate threshold or a superiority claim.

## 5. Mechanism audit boundary

The mechanism audit freezes particle counts `[512, 2048]`, both registered
proposal kernels, `rejuvenation_steps` `[0, 1]`, and four additional fixed
seeds. It records proposal acceptance, ordinary and pre-bridge resampling,
root genealogy, parent-offspring concentration, terminal structural diversity,
and exact-reference fidelity. Paired comparisons are made at fixed seed,
particle count, and the other mechanism setting.

The output remains diagnostic-only. It is used to decide whether the observed
fidelity error is dominated by particle variance, proposal mixing, resampling,
or rejuvenation; it does not authorize an error envelope or a downstream
scientific claim.

### 5.1 Accepted-move semantic audit

Every rejuvenation proposal now emits a response-free `OpenTargetMoveDiagnostic`
record.  The record contains the proposal kind, accept/reject outcome, exact
forward/reverse MH decision already used by the kernel, raw AST identifiers,
polynomial-equivalence identifiers, typed-tree structural distance, exact
polynomial ℓ1 semantic distance on the registered algebra, node-count change,
discrepancy/kernel state, and a registered move type:

- `self-transition`;
- `within-equivalence-class`;
- `cross-equivalence-class`;
- `discrepancy-state-change`; or
- `cross-equivalence-and-state-change`.

The mechanism runner aggregates accepted structural and semantic jump
distances, accepted node-count changes, move-type-specific acceptance rates, cross-class and
discrepancy-state transition fractions, and a complete accepted transition
table.  These summaries are descriptive and are not available to the sampler
while it is running.  In particular, no acceptance threshold, distance cutoff,
proposal mixture, or particle count may be changed as a function of an
observed fidelity result.

The audit is designed to distinguish three mechanisms that can make a lower
acceptance proposal more faithful: (i) fewer accepted self-transitions but
larger accepted semantic jumps, (ii) better coverage of posterior equivalence
classes, or (iii) lower correlation between accepted moves and resampling
genealogy.  A conclusion about any of these mechanisms requires stability over
the preregistered seeds and both particle counts; one favorable cell is not a
fidelity envelope.

## 5.2 Resampling variance and genealogy audit

The resampling audit freezes the complete-uniform rejuvenation kernel,
`rejuvenation_steps = 1`, particle counts `[512, 2048]`, and four additional
seeds while comparing three registered unbiased population transforms:
systematic, stratified, and residual resampling. It records paired posterior,
predictive, and evidence errors, pre-bridge versus ordinary resampling, root
ancestor survival, parent-offspring concentration, move statistics, and
accepted structure transitions. The resampling choice is a configuration-level
numerical kernel, not a response-adapted correction.

This audit is diagnostic-only. It may identify avoidable resampling variance or
genealogy collapse, but it cannot select a resampler from the same observed
fidelity result and cannot define the finite-particle fidelity envelope. A
resampler is eligible for a future envelope only if its pre-registered
comparison is stable across both particle counts and all seeds, and if the
resulting posterior and predictive functionals satisfy a separately frozen
error certificate.

## 5.3 Bridge-boundary resample-move schedule

The resampling comparison does not authorize a resampler: systematic,
stratified, and residual kernels can trade posterior, predictive, evidence,
and genealogy error across particle counts. The next mechanism-level
comparison therefore freezes systematic resampling and compares two
target-invariant orderings of the same operation:

- `pre-bridge`: resample at the start of the next bridge when the previous
  bridge reaches the CESS boundary;
- `post-bridge`: resample immediately after a nonterminal bridge reaches that
  boundary, then apply rejuvenation at the current fractional target before
  starting the next bridge.

The operation is a standard resample-move transformation. Moving the boundary
event changes genealogy and the temperature at which the invariant kernel is
applied, but does not change the registered target or likelihood path. The
diagnostic records an explicit resampling reason so pre-bridge boundary events,
post-bridge boundary events, and ordinary ESS-threshold events cannot be
confused. This schedule audit remains diagnostic-only and cannot be selected
from a single favorable fidelity result.

## 5.4 Target-invariant kernel-mixture candidate

The semantic audit showed a stable mechanism pattern on the exact slice:
complete-uniform accepts fewer proposals but its accepted moves have larger
typed-tree and polynomial-semantic jumps and more cross-equivalence-class
transitions.  The principled next candidate is therefore an independent
mixture proposal, not an acceptance-rate adjustment:

\[
q_{\mathrm{mix}}(z') =
  \omega q_{\mathrm{prior}}(z') + (1-\omega)q_{\mathrm{uniform}}(z'),
\qquad \omega=1/2.
\]

The equal weight is frozen before evaluating the mixture audit.  For every
proposal the MH ratio uses the full mixture probability
\(q_{\mathrm{mix}}(z)/q_{\mathrm{mix}}(z')\); it never substitutes the
probability of whichever component happened to generate the draw.  Thus the
mixture changes exploration only and leaves the registered posterior target
unchanged.  The component label is recorded for later diagnostics but is not
used for adaptation during a run.

The mixture audit is diagnostic-only and compares all three kernels at the
same four seeds, particle counts `[512, 2048]`, and one rejuvenation step.  A
favorable mixture result cannot define a fidelity envelope; its first required
check is the exact finite-slice detailed-balance/stationarity certificate.

## 5.5 Random-scan kernel mixture

The independent mixture above uses the full mixture density in one MH ratio.
Because its audit result was intermediate rather than uniformly dominant, the
next candidate is a distinct construction:

\[
K_{\mathrm{scan}}
 = \omega K_{\mathrm{prior}}
 + (1-\omega)K_{\mathrm{uniform}},
 \qquad \omega=1/2.
\]

At each rejuvenation attempt, one already-valid reversible kernel is selected
first.  The selected component then uses its own exact MH ratio: prior-ratio
for the prior-independence kernel and the symmetric ratio for complete-uniform.
The result is a convex combination of invariant kernels, not a new learned
target and not an adaptive response-dependent proposal.  The certificate
constructs both component transition matrices and verifies the convex
combination directly.  Its audit is separate from the independent-mixture
audit and remains diagnostic-only.

## 5.6 Invariant-rejuvenation depth audit

The resampling and bridge-boundary comparisons do not produce a uniform
posterior-fidelity winner. The next isolated mechanism audit freezes the
complete-uniform proposal, systematic pre-bridge resampling, particle counts
`[512, 2048]`, and four new seeds while comparing `rejuvenation_steps` in
`[0, 1, 2, 4]`. Each sweep is a registered target-invariant kernel application
at the current fractional bridge target; increasing the sweep count changes
mixing effort but never changes the posterior target.

The audit records acceptance, accepted semantic jump distance, move type,
posterior and predictive error, evidence error, ESS, and genealogy for every
depth. It is diagnostic-only: no depth is promoted from a favorable cell, and
the result cannot authorize a finite-particle fidelity envelope or downstream
real-data claim.

## 5.7 Confirmatory multi-fixture fidelity envelope

The confirmatory Gate no longer compares numerical candidates. Before any
confirmatory result is observed, it freezes `particle_count = 2048`, four new
seeds, complete-uniform rejuvenation, systematic pre-bridge resampling, and
four invariant rejuvenation sweeps. It evaluates that one mechanism on three
registered one-dimensional exact-reference fixtures whose action grids and
polynomial construction coefficients are committed in configuration. Fixture
selection, predictive evaluation points, and thresholds cannot depend on a
particle response, runtime failure, or exact-reference discrepancy observed
during the confirmatory run.

The candidate is registered by mechanism and prior audit support, not selected
from the confirmatory responses: 2048 is the largest already-audited particle
count, complete-uniform has a direct finite-support invariance certificate,
systematic/pre-bridge is the canonical registered resample-move ordering, and
four sweeps is the largest previously registered invariant depth. The
confirmatory Gate cannot change any of these choices after a fixture or seed
fails.

Every fixture/seed cell reports raw-AST and equivalence-class posterior error,
predictive density and CDF error on the registered evaluation grid,
log-evidence error, ESS/CESS, normalized weight entropy, resampling, distinct
root ancestry, normalized root entropy, and maximum parent-offspring
concentration. The formal envelope uses two simultaneous requirements:

1. the global worst cell must remain inside every registered posterior,
   predictive, evidence, ESS, and genealogy bound; and
2. the span of the fixture-specific seed medians must remain inside every
   registered cross-fixture stability bound.

Means are descriptive only. All twelve fixture/seed cells must complete, and
mass conservation, evidence telescoping, terminal bridge completion, exact
batch/sequential agreement, and proposal invariance remain mandatory. Missing
cells or unavailable metrics fail closed; they cannot be replaced by a mean
over the successful cells.

A pass is finite-slice scalable-inference fidelity evidence only. It authorizes
writing and executing a separate predictive-calibration Gate, but it does not
itself constitute calibration evidence. Real data, acquisition, and held-out
access remain blocked until both the confirmatory fidelity Gate and that
separate predictive-calibration Gate pass.

## 5.8 Independent waste-free variance-reduction development stage

The failed confirmatory envelope does not authorize another confirmatory run.
The next stage instead isolates one variance-reduction mechanism on three new
response-free development fixtures and four new development seeds. The old
confirmatory fixtures and seeds are not used for selection.

The baseline retains only the terminal state from each of the four registered
rejuvenation transitions. The candidate records every state after each
transition in a transient chain-major pool. If source particle (i) has SMC
weight (w_i) and the registered depth is (R=4), every retained chain state
has pool weight (w_i/R). The same registered unbiased resampler compresses
the (NR) transient states back to the frozen resident population (N).
Compression adds no target or proposal evaluation and creates no new support.
Every accepted and rejected transition inherits its source-chain weight.

This is a bounded-memory waste-free pool-compression candidate inspired by
waste-free SMC, not a claim that the compressed implementation is identical to
the canonical uncompressed construction. Its target argument is limited to:
each (K^r) is invariant at the current bridge target, their equally weighted
mixture is invariant, and the registered resampling transform is marginally
unbiased. Exact-reference fidelity must still be demonstrated empirically.

The matched dominant computational budget is a vector fixed before execution:

- the same (N=2048) resident particles;
- the same four invariant transitions per source particle and bridge;
- exactly (4N=8192) proposal and target evaluations per bridge;
- the same target, proposal, CESS path, resampler, schedule, fixtures, and
  paired seeds.

The candidate's transient (4N) storage, pool normalization, and compression
resampling are not called free. Pool size, compression draws, total actual
resampling events, and wall-clock time are reported separately. Wall-clock
means are descriptive because they are execution-environment dependent; the
registered evaluation counts and resident population define the matched
dominant budget.

Every fixture/seed/method cell reports raw-AST and equivalence-class error,
signed and absolute log-evidence error, and every registered predictive
density and CDF point with signed and absolute error. Selection uses paired
worst-case noninferiority or improvement; means are descriptive only.

Genealogy is evaluated at every actual resampling event, including ordinary
CESS/ESS resampling and waste-free pool compression. For event (e), let
(A_e^-) and (A_e^+) be the numbers of distinct root ancestors immediately
before and after the event, and let (H_e^-) and (H_e^+) be root entropy
normalized by (log N). The registered event metrics are

\[
  a_e=-\log(A_e^+/A_e^-), \qquad
  \ell_e=\max(0,H_e^- - H_e^+).
\]

The event record also retains the signed entropy change, distinct parents,
maximum parent-offspring fraction, event kind, observation/bridge index, and
the before/after raw root counts and entropies. Existing bridge-level minimum
root fraction, terminal root fraction/entropy, parent concentration, ESS,
CESS, and the legacy terminal-loss-divided-by-event-count summaries remain in
the evidence record, but they do not replace the per-event Gate.

This development stage can only authorize freezing a new, still-unseen set of
confirmatory fixtures and seeds. It cannot pass the prior confirmatory Gate and
cannot authorize predictive calibration, real data, acquisition, or held-out
access. The unseen confirmatory fixture specifications and seeds are frozen
only after the development mechanism is eligible; they are never generated or
chosen after observing their particle errors.

## 6. Implementation boundaries

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

## 7. No-go conditions

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

## 8. Downstream order

Only after this Gate passes may a separate predictive-calibration contract be
written. Only after that calibration Gate passes may a new held-out-closed real
development protocol be considered. No P3F.3 result authorizes acquisition,
motif transfer, VED, held-out confirmation, or superiority claims.
