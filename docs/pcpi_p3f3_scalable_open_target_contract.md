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

## 5.9 Strict total-budget terminal-pool estimator development

The P3F.3-VR.1 result is retained as negative development evidence. All 24
runs completed and the compressed pool improved worst log-evidence error and
per-event genealogy loss, but it failed predictive-density, predictive-CDF,
and raw-AST noninferiority. It also exposed a protocol bug: the runner matched
proposal evaluations per bridge but not the total number of bridges. The
terminal-only runs used 10--11 bridges while compressed-pool runs used 8--9,
so their total proposal counts differed by 10--27 percent even though the old
budget decision returned true. That decision is invalid and cannot authorize
another confirmatory Gate.

P3F.3-VR.2 corrects the design rather than changing a threshold. Both methods
use the response-free fixed grid

\[
  \beta \in \{0, 0.25, 0.5, 0.75, 1\}
\]

for every observation. With eight observations, $N=2048$, and four
rejuvenation proposals per resident particle and bridge, both methods must
execute exactly 32 bridges and 262144 proposal/target evaluations per run.
The runner checks paired total counts, not only per-bridge counts. CESS remains
reported and must exceed the registered lower bound; the fixed grid changes
only the numerical path, not the target.

Predictive-functional computation is also matched at 8192 component
evaluations per registered density or CDF point. The terminal-only estimator
repeats its 2048 weighted components four times with weights divided by four;
this leaves its numerical estimate unchanged while preventing the candidate's
larger terminal estimator population from receiving an unreported evaluation
budget advantage.

The new candidate still compresses intermediate pools to $N$ for propagation,
but terminal posterior functionals do not use that lossy draw. At the final
registered target, all $4N=8192$ intermediate states retain normalized source
weights $w_i/4$. Raw-AST mass, equivalence mass, predictive density, and
predictive CDF are evaluated directly on this weighted terminal pool. The
resident $N$-particle population remains separately available for propagation
and genealogy. Log evidence remains the ordinary SMC normalizer and receives no
post-hoc correction.

VR.2 uses three new response-free development fixtures and four new seeds,
disjoint from both VR.1 and the failed confirmatory audit. It retains the same
signed/absolute pointwise errors and per-resampling genealogy records. Only a
pass of every preregistered strict-budget decision can authorize freezing a new
unseen confirmatory fixture bank; it still cannot authorize calibration or any
real-data, acquisition, or held-out path.

## 5.10 Full-population waste-free propagation development

P3F.3-VR.2 is also retained as negative development evidence. Its 24 runs
matched the fixed bridge count and the registered MH proposal count. The
terminal-pool estimator improved the worst raw-AST, equivalence, predictive
density, and predictive CDF errors, but its worst absolute log-evidence error
was larger than the terminal-only comparator. The result therefore cannot
freeze another confirmatory bank.

The VR.2 audit additionally found that the open-target engine's private
`_conditional_ess` routine still returned the ordinary ESS after reweighting.
That statistic equals the current ESS at a zero increment, whereas registered
CESS must equal $N$ at a zero increment even for non-uniform incoming weights.
P3F.3-VR.3 reuses the canonical `conditional_effective_sample_size` primitive
and freezes the zero-increment identity as a regression test. The VR.2 CESS
decision is consequently invalid even though its other recorded estimators
remain useful negative development evidence.

VR.3 removes the propagation compression that prevented the intermediate
population from entering later potential and normalizing-constant estimates.
Both methods carry the same full population size $N=8192$ through every fixed
bridge. The standard comparator resamples $N$ parents and takes one invariant
MH step per parent. The waste-free candidate resamples $M=2048$ source chains,
takes $P=4$ successive invariant MH steps per source, and retains every
post-MH state as the next population, so $N=MP$. No pool is compressed before
the next incremental potential or terminal posterior functional.

Per bridge, each method must execute exactly 8192 incremental-potential
evaluations, 8192 MH proposal/target evaluations, and 8192 component
evaluations for each registered posterior functional. Across 32 bridges this
is 262144 potential evaluations plus 262144 proposal/target evaluations, or
524288 registered target evaluations per run. Wall clock remains descriptive.

Every resampling event retains the raw ancestry attrition and raw normalized
root-entropy loss. Because the waste-free transform intentionally maps $N$
states through at most $M=N/P$ source chains, even perfect source coverage has
the structural raw floors

\[
  a_{\mathrm{capacity}}=\log P, \qquad
  \ell_{\mathrm{capacity}}=1-\frac{\log M}{\log N}.
\]

VR.3 therefore also preregisters capacity-adjusted event losses while keeping
the raw quantities visible and gated. The raw limits are the analytical
capacity floors plus the previously registered event allowances of 0.25 for
log attrition and 0.05 for normalized entropy loss; capacity-adjusted limits
remain 0.25 and 0.05. This adjustment is fixed before the new VR.3 fixtures are
run and cannot be inferred from their outcomes.

VR.3 uses new response-free J/K/L development fixtures and seeds disjoint from
VR.1, VR.2, and the previous confirmatory bank. It may only authorize freezing
a new unseen confirmatory bank if every matched-budget, CESS, target-invariance,
fidelity, evidence, and raw-plus-adjusted genealogy decision passes. It cannot
itself authorize predictive calibration or any real-data, acquisition, or
held-out path.

## 5.11 Observation-terminal waste-free propagation development

P3F.3-VR.3 is retained as negative development evidence. Its 24/24 runs
completed with exact paired evaluation budgets, valid CESS, and source hashes
matching the registered code and configuration. The candidate nevertheless
lost to the standard comparator on worst equivalence-class, predictive-density,
predictive-CDF, and absolute log-evidence error. These failures occurred across
multiple fixture/seed pairs, not as a single outlier. The candidate also ended
with substantially fewer root ancestors and lower root entropy.

The causal implementation feature is the repeated $N\rightarrow M\rightarrow N$
source contraction at every fractional bridge. With four bridges per
observation and eight observations, VR.3 applies 32 such contractions. Its
capacity-adjusted *per-event* genealogy checks can pass while the repeated
events still accumulate terminal ancestry collapse. Lowering a fidelity limit,
selecting a seed, or treating the event adjustment as proof of global diversity
would therefore be an invalid repair.

P3F.3-VR.4 keeps the same target, the same $N=8192$ resident population, and
the same response-free fixed beta grid. Incremental potentials are evaluated at
all 32 registered bridges, but resample-move is restricted to $\beta=1$, once
per completed observation. Each method consequently performs eight
resample-move events. At every such event the standard comparator evaluates
$N$ one-step MH proposals, and the waste-free candidate evaluates $M=2048$
chains of $P=4$ successive proposals while retaining all $MP=N$ post-MH
states. Per run, both methods execute 262144 potential evaluations, 65536
proposal/target evaluations, 8192 components per posterior functional, and
327680 total target evaluations. The reduction relative to VR.3 is a frozen
mechanism change applied identically to both methods; it is not an unequal
candidate budget.

VR.4 keeps every raw and capacity-adjusted per-resampling event metric from
VR.3 and adds cumulative terminal requirements. Candidate terminal root
fraction and normalized root entropy retain the pre-existing raw lower bounds.
After division by the analytical source-capacity ceilings $M/N$ and
$\log M/\log N$, candidate terminal root fraction and entropy must also be
noninferior to the standard comparator. Proposal acceptance and the fraction of
accepted moves that cross an equivalence class are reported descriptively and
do not replace the exact-reference fidelity Gate.

VR.4 uses response-free M/N/O development fixtures and four seeds disjoint from
VR.1, VR.2, VR.3, and the prior confirmatory audit. No new confirmatory fixture
or seed is frozen in this stage. Only if every registered target, budget,
fidelity, evidence, ESS, and local-plus-terminal genealogy decision passes may
a separate unseen confirmatory bank be frozen. Predictive calibration, real
data, acquisition, and held-out access remain blocked.

## 5.12 Conditional acceptance Rao--Blackwell development

P3F.3-VR.4 is retained as negative development evidence. Its 24/24 runs used
the registered code and configuration, completed the exact matched budget, and
showed that observation-terminal scheduling fixed the cumulative genealogy
failure. After analytical source-capacity adjustment, the candidate terminal
root fraction and entropy were noninferior to the standard population. The
candidate nevertheless failed worst raw-AST, equivalence-class,
predictive-density, predictive-CDF, and absolute log-evidence fidelity. Raw-AST
and absolute log-evidence error were worse in 10 of 12 paired fixture/seed
comparisons. Proposal acceptance and accepted cross-equivalence movement were
nearly identical across methods. The remaining failure is therefore attributed
to within-chain dependence from representing $N=8192$ resident states by
$M=2048$ length-four chains, not to a threshold, seed, bridge, or genealogy
defect.

This limitation is consistent with the scope of Dau and Chopin's waste-free
SMC theory: the method is consistent and asymptotically normal, and its benefit
depends on the MCMC mixing regime; it is not a universal finite-budget
dominance result. Delmas and Jourdain likewise show that a naive
waste-recycling estimator can increase asymptotic variance. Fixed $P=4$
waste-free propagation is therefore retired from confirmatory consideration
rather than tuned against its development outcomes.

P3F.3-VR.5 instead removes only the auxiliary accept/reject-uniform noise from
the terminal MH estimator. The standard and candidate methods use identical
$N=8192$ resident particles, systematic resampling indices, complete-uniform
proposals, evaluated targets, acceptance uniforms, accepted resident states,
bridge schedules, genealogy, and log-evidence path. For each of the final $N$
evaluated proposals, the candidate additionally records the conditional
transition measure

\[
  (1-\alpha_i)\,\delta_{x_i}+\alpha_i\,\delta_{y_i},
\]

where $x_i$ is the current state, $y_i$ is the evaluated proposal, and
$\alpha_i$ is the exact MH acceptance probability. Thus each pair carries
total mass $1/N$, proposal evaluations are unchanged, and the resident SMC
target is untouched. This is the local conditional-expectation principle used
by Rao--Blackwellized MH methods. It does not by itself assert a universal
log-evidence or SMC-wide variance theorem; that boundary remains an
exact-reference development Gate.

Both methods execute 262144 incremental-potential evaluations, 65536 MH
proposal/target evaluations, and 327680 total target evaluations per run. The
candidate posterior functional uses $2N=16384$ weighted current/proposed
branches. The standard functional repeats each of its $N$ resident components
twice with half contribution, leaving its estimator unchanged while matching
16384 component evaluations per registered point. The runner requires bitwise
identity of paired resident populations, bridge schedules, moves, genealogy,
and log evidence, and separately verifies every Rao--Blackwell pair weight.

VR.5 uses new response-free P/Q/R development fixtures and seeds disjoint from
all earlier development and confirmatory banks. A pass may authorize only a
subsequent freeze of still-unseen confirmatory fixtures and seeds. Predictive
calibration, real data, acquisition, and held-out access remain blocked.

## 5.13 Acceptance Rao--Blackwell confirmatory fidelity Gate

The registered P3F.3-VR.5 development archive completed all 24 fixture/seed/
method runs with identical paired resident populations, bridge schedules,
moves, genealogy, log evidence, and matched target-evaluation budgets. Its
candidate worst errors were 0.01867 for raw AST mass, 0.01803 for equivalence
mass, 0.01285 for predictive density, 0.00243 for predictive CDF, and 0.02420
for absolute log evidence. All development decisions passed. This authorizes
freezing an independent confirmatory bank; it is not itself confirmatory
fidelity evidence. In particular, predictive-density error improved in only
3 of 12 paired development comparisons even though its global worst case
improved. The confirmatory decision must therefore include per-fixture
stability rather than rely on a mean or one global maximum alone.

P3F.3-CF.RB.1 freezes four new response-free S/T/U/V exact fixtures and five
new seeds before any response is evaluated. Each of the 40 method runs retains
the VR.5 target, population size, fixed beta schedule, observation-terminal
rejuvenation schedule, systematic resampling, complete-uniform proposal, and
matched proposal, incremental-potential, total-target, and posterior-
functional evaluation budgets. The candidate is evaluated against both:

- the absolute worst-case and cross-fixture seed-median-span envelope inherited
  unchanged from the first preregistered P3F.3 confirmatory fidelity audit; and
- paired global noninferiority plus a minimum median improvement within every
  fixture, preventing good fixtures from masking an unstable fixture.

The evidence record retains signed log-evidence error and every pointwise
predictive density/CDF signed and absolute error. Genealogy is gated using the
preregistered per-resampling-event ancestry log attrition and root-entropy
loss limits inherited from VR.4, while the original bridge and terminal root,
entropy, and parent-offspring diagnostics remain present. A failure is final
for this frozen bank: seeds, fixtures, envelopes, schedules, and budgets may
not be adapted to the responses.

Only a complete P3F.3-CF.RB.1 pass authorizes construction of a separate
predictive-calibration Gate. Real data, acquisition, and held-out access stay
blocked until both confirmatory fidelity and predictive calibration pass.

## 5.14 Negative CF.RB.1 evidence and non-terminal acceptance knots

The frozen P3F.3-CF.RB.1 archive completed all 40 runs with the registered
source, target, fixture bank, seeds, and matched budgets. Forty-five of 51
decisions passed, but the formal Gate failed six decisions. The candidate
worst predictive-density error was 0.0156595, exceeding both the absolute
0.015 envelope and the paired standard worst error 0.0154163. Its
cross-fixture seed-median spans also exceeded the frozen limits for CESS, ESS,
minimum root fraction, and terminal root fraction. This is permanent negative
confirmatory evidence. The S/T/U/V fixtures, seeds, or envelopes must not be
changed, rerun as a new confirmatory bank, or reused to choose a replacement
mechanism.

The paired resident paths in CF.RB.1 were identical, so the ESS and genealogy
span failures cannot be attributed to the terminal Rao--Blackwell estimator.
They measure response-dependent variation in the underlying SMC population.
Moreover, integrating only the final accept/reject uniform is a terminal
post-processing operation: it does not alter prior bridge weights, evidence,
resampling, or genealogy, and it does not guarantee pathwise dominance of a
maximum absolute-error functional. P3F.3-VR.5 is therefore retired from further
confirmatory use without retroactively changing its valid development result.

P3F.3-VR.6 registers a new development-only non-terminal mechanism. At the
last fractional bridge for an already observed response, both methods first
sample the same complete-uniform proposal and evaluate both accept/reject
branches under the next potential. The standard comparator samples the MH
acceptance uniform and weights the selected branch. The candidate instead
uses

\[
  K(G)(x,\tilde{x})
  = (1-\alpha(x,\tilde{x}))G(x)
    + \alpha(x,\tilde{x})G(\tilde{x})
\]

as its predictive potential, then samples from the potential-twisted
two-branch kernel with probabilities proportional to the two terms. This is a
specialized adapted knot of the accept/reject kernel: it preserves the
terminal Feynman--Kac measure while moving potential information into a
non-terminal transition, where variance ordering is theoretically available.
It uses no future observation, acquisition response, held-out value, or exact
answer.

Both sides retain $N=8192$, four bridges per observation, one proposal per
particle and observation, systematic resampling, and identical primitive
budgets: 65536 proposal-target evaluations, 327680 incremental-potential
evaluations (including both terminal branches), 393216 total target
evaluations, and 8192 posterior-functional component evaluations per point.
The standard side deliberately evaluates and discards its counterfactual
branch only to keep this comparison matched.

VR.6 uses new response-free W/X/Y development fixtures and seeds disjoint from
all previous banks. Absolute fidelity and genealogy safety limits remain
inherited from earlier preregistrations. Cross-fixture spans for fidelity
errors remain absolute; response-difficulty-sensitive CESS, ESS, and genealogy
spans are compared candidate-to-standard on the same fixture/seed bank rather
than treated as an invariant property of heterogeneous targets. This estimand
change is frozen before VR.6 responses and does not alter the failed CF.RB.1
decision. Only a complete VR.6 pass may authorize a new unseen confirmatory
freeze. Predictive calibration and all real/acquisition/held-out paths remain
blocked.

## 5.15 Negative VR.6 evidence and the terminal-safe knotset

The frozen VR.6 W/X/Y archive completed all 24 registered runs with matched
proposal, potential, posterior-functional, population, bridge, and resampling
budgets. Its source and target identities match GitHub commit `e6bf9e8`. The
candidate passed every absolute fidelity and safety bound, but only 39 of 49
registered decisions passed. In particular, its candidate-to-standard
worst-error ratios were 1.00524 for log evidence, 1.20454 for predictive
density, and 1.16558 for predictive CDF. Its minimum-root, terminal-root, and
terminal-root-entropy cross-fixture span ratios were also above one. W and Y
showed negative within-fixture median improvements for several posterior
functionals. This is final negative development evidence; W/X/Y, their seeds,
and their envelopes cannot be reused or changed to authorize the mechanism.

The failure exposes a mismatch between the VR.6 name and its actual horizon.
VR.6 applied the adapted accept/reject knot at every observation, including
the eighth observation, which is the terminal time of the complete
Feynman--Kac sequence. The general knot variance ordering applies to knots at
times $p<n$. A terminal knot requires a separate test-function-aware
construction to order arbitrary terminal posterior functionals; a generic
terminal adaptation only has a general variance guarantee for the normalizing
constant. Therefore the local conditional-potential optimality used by VR.6
does not imply a variance ordering for raw AST mass, equivalence mass, or
predictive density/CDF at the terminal observation.

P3F.3-VR.7 registers a terminal-safe knotset before observing any new
development response. It preserves the VR.6 target, complete-uniform proposal,
$N=8192$ population, four fractional bridges, systematic resampling, and every
primitive evaluation budget. The candidate applies the adapted acceptance
knot only at observations $1,\ldots,T-1$; observation $T$ uses the same sampled
acceptance branch, incremental potential, and resampling construction as the
standard comparator. Both sides still evaluate both terminal branches at all
observations so the proposal and potential budgets remain exactly matched.

The absolute, paired worst-case, per-fixture median-improvement, error-span,
and event-genealogy envelopes are inherited unchanged from VR.6. VR.7 uses new
response-free Z/AA/AB fixtures and new seeds disjoint from W/X/Y and every
prior confirmatory bank. A new decision also requires exactly $T-1$ candidate
knots and zero terminal knots. Only a complete VR.7 pass may freeze another
unseen confirmatory bank. It does not itself authorize predictive calibration,
real data, acquisition, or held-out access.

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
