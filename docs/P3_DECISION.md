# P3 decision — representative-safe maximin joint acquisition

Status: **P3B.10, P3C.1, and P3D.2 real efficacy failed. P3E.1 repairs an exact
generalized-update decision mismatch. P3E.2 passes its union-orthogonal
posterior-adequacy correctness Gate and completes a protocol-valid CCPP audit
without rejection, but this is not a posterior-adequacy certificate. P3E.3
passes predictive-calibration correctness, but its returned CCPP
validation-role audit rejects three of eight seed-wise PIT sequences and fails
global eligibility. P3F.1 and P3F.2a--c now pass model-class and exact-reference
correctness Gates only. Another acquisition run, P4/P5, held-out confirmation,
and superiority claims remain blocked.**

## Evidence retained

P3A.2 validates class-EIG against independent adaptive quadrature. P3B.6
validates a single posterior design coordinate system. P3B.7 validates
budget-resolved predictive classes. P3B.8 validates joint class/predictive
information. P3B.9 validates the response-free MMD guard and completed 96/96
matched-budget real runs with zero failures, an intact EvidenceRegistry,
matched budgets and splits, and closed held-out.

The returned P3B.9 efficacy assessment is nevertheless
`REAL_ADVANTAGE_NOT_DEMONSTRATED`. On CCPP, paired frozen-class gain relative to
random is `-0.2509967941549854`, with 95% interval
`[-0.4616616356504576, -0.04033195265951314]`; seven of eight seeds are negative
transfer. The pooled selected-score versus realized local class-gain Spearman
correlation is `-0.007906557564660103`. The guard was active, nonempty, and
non-increasing at all 256 CCPP PCPI queries, so this is not a guard or protocol
failure.

The unchanged held-out-closed P3B.10 rerun is also protocol-valid but does not
demonstrate advantage. It completed 96/96 policy runs with zero failures and a
valid 97-event EvidenceRegistry. On CCPP, PCPI versus random has mean paired
frozen-class gain delta `-0.28378374200721335`, with 95% interval
`[-0.5046926179404461, -0.06287486607398057]`; 7/8 seeds are negative. The
grouped Gas Turbine family has mean delta `0.034959682233189106`, but its 95%
interval is `[-0.2603092503809608, 0.3302286148473398]`. The preregistered
assessment is `REAL_ADVANTAGE_NOT_DEMONSTRATED`, with
`strong_evidence=false` and `strong_structural_evidence=false`.

## P3B.10 repair

Let `E_eta(a)` denote the P3B.9 joint information utility for visible candidate
`a` under generalized-posterior likelihood power `eta`. The ambiguity set is
the candidate set frozen before P3B.10:

`H = {0.125, 0.25, 0.5, 1.0}`.

P3B.10 ranks each candidate by

`U_min(a) = min_{eta in H} E_eta(a)`.

Each `E_eta` uses the same observed responses, finite structure bank, frozen
design preconditioner, initial-frozen structure-to-class map, visible candidate
covariates, and registered target domain. The calibrated nominal posterior
continues to own posterior reporting, validation metrics, classes, and efficacy
evaluation. The ambiguity family changes PCPI ranking only.

The P3B.9 representative set remains unchanged:

`S_t = {a : MMD^2(A_t union {a}, D*) <= MMD^2(A_t, D*) + tau_num}`.

- If `S_t` is nonempty and the lower-envelope leader is certified, select the
  maximin joint-information leader in `S_t`.
- If the robust leader is uncertified, use the existing nominal
  posterior-epistemic fallback in `S_t`.
- If `S_t` is empty, use the unchanged explicit minimum-MMD fallback.
- If multiple powers attain the lower envelope, report the smaller likelihood
  power deterministically.

No response outside the acquired history, validation response, held-out value,
dataset or target name, formula, result direction, new threshold, split, or
additional candidate budget enters the rule.

## P3B.10 correctness Gate

The controlled fixture must pass 27 decisions: all seventeen P3B.9 regression
decisions plus ten finite-family checks covering the frozen ambiguity set,
pointwise lower envelope, least-favorable-model audit, independent exact
lower-envelope agreement at the registered maximum quadrature budget, interval
containment, exact safe-set winner, robust rank certification, model-order
invariance, singleton recovery, and deterministic tie handling.

Full regression, static integrity, leakage audit, source identity, patch
installation, and rollback checks must also pass. Only then may P3B.10 run once
on the unchanged CCPP and grouped Gas Turbine development protocol with
untouched-heldout closed.

The controlled Gate result on 2026-08-13 was **PASS for entry to the real-data
run only**. The subsequent real result is **NO-GO for efficacy**. Source
identity was bound directly to a clean Git worktree by commit, Git tree,
tracked-source SHA-256, production-code SHA-256, configuration hashes, and
exact runtime dependencies; a ZIP is optional rather than required.

## Claim boundary

P3B.10 controlled evidence supports correctness of the finite maximin decision
rule only. The real P3B.10 run supports protocol-valid negative development
evidence, not acquisition superiority. The finite likelihood-power ambiguity
set is not a universal misspecification guarantee. The next work must repair
the general posterior/decision model and pass a new controlled Gate before any
real rerun. Open-grammar superiority, motif safety, held-out confirmation,
physical intervention, VED discovery, and scientific-law claims remain
unsupported.

## P3C.1 discrepancy-aware repair

P3C.1 does not tune a new likelihood power or branch on dataset identity. It
estimates a common residual-discrepancy scale from posterior sufficient
statistics in the acquired history and distributes that scale over candidate
and registered target covariates according to distance from the observed
design. The additional variance is moment-matched into each finite-bank
Student-t predictive component before evaluating the same finite maximin joint
class/predictive utility.

The old P3B.10 policy, schema, and result remain reproducible. P3C.1 has a
separate policy identifier, config schema, hypothesis identity, claim boundary,
and EvidenceRegistry lineage. Protocol validity and positive efficacy evidence
are separate booleans. The real P3C.1 run is protocol-valid but negative:
CCPP remains negative-transfer relative to random, while the Gas-family result
does not exclude zero and does not establish a family-uniform structural gain.
See `docs/pcpi_p3c1_result_audit_20260813.md`. No P4/P5 or superiority claim
is authorized.

## P3D.1 correctness-only decision repair

The next task-independent candidate is certified reference-dominance
acquisition. It returns to the initial-frozen class-EIG and registers a
response-free reference policy over the same visible candidates. A targeted
action is authorized only when its valid numerical lower bound strictly
exceeds the reference policy's probability-weighted upper bound; otherwise the
rule samples from the reference policy. This removes the earlier semantic
switch from uncertified EIG to posterior epistemic variance and automatically
falls back when class information capacity is zero.

P3D.1's exact finite-outcome controlled Gate passed 14/14 decisions at commit
`5d71f588398daac3a7c8d982ec3eac0b5834d73c`. This supports the
interval-handover implementation and model-relative proposition only; it does
not repair posterior misspecification or establish real-data safety or
efficacy. See
`docs/pcpi_p3d1_root_cause_and_repair.md` and
`docs/pcpi_p3d1_result_audit_20260814.md`.

Upstream commit `81f7cde` restored the manifest-listed `hypothesis_mvp.data`
package. All six files match their historical SHA-256 and byte counts, and the
merged branch passed the complete `186`-test suite and closed the
source-completeness blocker.

## P3D.2 frozen real-only integration

P3D.2 implements analytic class-EIG inequality bounds. Deterministic response
quantization gives a data-processing lower bound; Gaussian maximum entropy and
within-class mixture-entropy concavity give an upper bound. The numerical
implementation uses a frozen outward tolerance and agrees with independent
adaptive quadrature on continuous correctness fixtures. Because the Student-t
special functions are not evaluated by formal interval arithmetic, the
resulting handover remains a numerically validated, model-relative decision
contract rather than a real-world no-harm theorem.

The real runner and frozen config were added at implementation commit
`cd05e1a`. The returned v2 archive is bound to clean source commit `2effdee`.
It completes 96/96 runs with zero failures, a valid 97-event EvidenceRegistry,
verified official hashes, and closed held-out. The registered efficacy status
is nevertheless `REAL_ADVANTAGE_NOT_DEMONSTRATED`. CCPP PCPI-minus-random
frozen-class gain is `-0.253805` with 95% interval
`[-0.482951,-0.024660]`; the grouped Gas effect is `-0.004906` with interval
`[-0.223369,0.213557]`. See
`docs/pcpi_p3d2_result_audit_20260814.md`.

## P3E.1 update-coherence correctness repair

The P3D.2 posterior is power-likelihood generalized Bayes. Its update uses
`q(z)p(y|z)^eta`, while its ordinary class-MI score is computed from the
nominal mixture `q(z)p(y|z)`. When `eta != 1`, that score is not the expected
frozen-class entropy reduction induced by the actual update. Thirteen of the
sixteen Gas target/seed calibrations used `eta < 1`.

P3E.1 therefore enumerates the signed expected entropy change under the
implemented update and requires a targeted lower bound to exceed both the
registered reference upper bound and zero. At `eta=1` it recovers ordinary
class-EIG exactly; at `eta=0.25` the frozen fixture proves that the two
utilities can reverse the action ranking. The isolated correctness Gate passes
10/10 decisions and has no real-data or held-out surface.

This does not resolve CCPP, where every seed used `eta=1` and negative transfer
still excludes zero. P3E.1 is thus a necessary semantic correction, not a
posterior-adequacy repair and not authorization for another real experiment.
See `docs/pcpi_p3e1_root_cause_and_update_coherence.md`.

## P3E.2 posterior-adequacy correctness repair

P3E.2 addresses the remaining CCPP `eta=1` blocker without modifying the real
runner. It constructs a response-free RBF discrepancy basis orthogonal to the
union of every candidate structure design on a frozen finite domain. Exact
conjugate marginal likelihoods compare the nominal spike with this structured
residual slab. Their prequential Bayes factor is a unit-initialized test
martingale under the declared nominal marginal; crossing the frozen
`1/alpha=100` boundary makes nominal targeted acquisition ineligible and
forces the registered reference-only mode.

The deterministic fixture passes 11/11 decisions. Its exact null has log Bayes
factor `-0.713366` and remains eligible; the registered structured residual has
log Bayes factor `4.936581` and crosses at round 16. Orthogonality preserves
the structural coefficient posterior means exactly to numerical tolerance.

The frozen initial-development-only real audit was then executed on official
UCI CCPP data and its uploaded archive passed the archive-level identity and
structure checks: 8/8 seeds completed, all 97 e-process rounds per seed are
present, source/config/runner hashes match the clean canonical tree, and the
held-out and acquisition flags remain closed. No seed crossed the registered
`E_t >= 100` boundary. This is protocol-valid **non-rejection** against one
registered discrepancy alternative, not a posterior-adequacy certificate;
`formal_real_posterior_adequacy_evidence=false` and
`formal_efficacy_evidence=false` remain binding.

The result is recorded in
`docs/pcpi_p3e2_real_posterior_adequacy_result_20260815.md`. It does not
validate the discrepancy-augmented posterior, does not authorize another
acquisition comparison, and does not open held-out data. Gas remains outside
this audit because its `eta<1` branches need a separately proved
update-coherent adequacy contract. A predictive-calibration Gate is still
required before any downstream acquisition decision is revisited.

## P3E.3 predictive-calibration compatibility audit

P3E.3 is a task-independent diagnostic of the finite-bank posterior predictive
CDF. Its correctness fixture passes all five registered decisions: fixed PIT
basis moments, balanced-fixture non-rejection, concentrated-fixture
rejection, predictive-CDF row-order equivariance, and prequential prefix
isolation. This is implementation evidence only; the fixture contains no
measured data and cannot support a calibration, adequacy, or efficacy claim.

The real protocol is frozen to CCPP, eight registered seeds, 32
initial-development rows for power selection, and 256 validation-role rows for
the PIT e-process. The validation role is opened only for this diagnostic;
the untouched held-out role remains closed and acquisition comparison and
authorization remain false. A selected `eta<1` is fail-closed for the proper
nominal-marginal interpretation of the e-process. A no-crossing real result
would therefore be only non-rejection against the fixed PIT betting family,
not a predictive-calibration or posterior-adequacy certificate. The local
execution is specified in
`docs/pcpi_p3e3_predictive_calibration_protocol_20260815.md`.

The returned audit completed all eight seeds. Seeds `2026080701`,
`2026080706`, and `2026080707` rejected the registered PIT-uniformity null;
only five seeds were eligible for a proper nominal-marginal interpretation and
the global Gate failed. Held-out remained closed and no acquisition policy was
run. See `docs/pcpi_p3e3_real_predictive_calibration_result_20260816.md`.

## P3F.1 structure-wise generative discrepancy

P3F.1 moves discrepancy from a union-orthogonal diagnostic into the proper
finite generative posterior. Each symbolic structure receives its own
response-independent projected GP factor, constructed by whitening and a
null-space basis so PSD and structure-wise orthogonality hold by construction.
The spike is represented once per structure; slab mass is divided over frozen
kernel states. Structure, spike/slab, kernel, coefficient, discrepancy, and
noise uncertainty all enter one normalized Student-t predictive mixture.

The correctness runner passes all nine decisions and has no data-root,
held-out, or acquisition surface. This establishes algebra and implementation
only; it does not establish calibration or efficacy.

## P3F.2a--c open target correctness

P3F.2a defines a proper countably-open typed AST prior with a geometric node
count. The exact finite enumeration is explicitly conditional and records its
omitted tail. Raw AST mass is aggregated into exact polynomial equivalence
classes without losing prior or posterior probability.

P3F.2b combines that exact slice with the P3F.1 generative posterior. The
registered one-amplitude expression model makes its design column the exact
parameter tangent, so the structure-wise projection is tangent-correct within
this bounded algebraic contract. Unsupported noise families, measurement
error, nonlinear constants, and transcendental identities fail closed.

P3F.2c uses an exhaustive collapsed-state sequential SMC reference and exact
Metropolis--Hastings matrices for complete-uniform and prior-independence
proposals. All 15 registered decisions pass, including open-prior
normalization, class-mass conservation, batch/sequential identity, evidence
telescoping, RJMCMC detailed balance/stationarity, proposal invariance, and
row-order equivariance.

These results do not validate scalable open-grammar exploration. The next
admissible work is a scalable particle approximation checked against this
reference. No new real experiment is authorized. See
`docs/pcpi_p3f2_open_target_method_contract_20260816.md` and
`docs/pcpi_p3f2_open_target_correctness_result_20260816.md`.

## P3H semiparametric residual-law reconstruction

P3G.5 retained eight predictive-calibration rejections after adding the
observed Gas year nuisance. The aggregate improvement with exchanged seed
failures closes the finite-state response-family sequence. P3H keeps the
scientific posterior unchanged and reconstructs only its probability residual
law: at round `t`, the base forecast produces the raw PIT `F_t(Y_t)`, and a
dyadic Pólya-tree trained only on earlier raw PITs transforms the next base CDF.

P3H.1 freezes independent Beta(1/2,1/2) split prediction and the count-only
depth `floor(log2(max(n,1))/2)`. Its correctness surface contains no real data,
RNG, dataset/target/seed branch, response-selected bandwidth or fixed depth
cap. The Gate verifies normalized positive leaf masses, continuous CDF and
inverse composition, the transformed-density chain rule, count-order
invariance, sublinear active leaves, an unchanged scientific posterior update
and future-response isolation.

P3H.2 preserves the P3G.5 real datasets, seeds, budgets, validation order, PIT
betting family and familywise allocation, but is deliberately calibration-only.
Candidate responses and held-out are sealed regardless of its result. A real
non-rejection would be compatibility with the fixed PIT betting family, not a
calibration theorem. Acquisition remains blocked until a later correctness
Gate represents the transformed predictive density in the decision utility.
See `docs/pcpi_p3h_semiparametric_residual_contract_20260825.md`.

The unique P3H.2 real calibration-only execution then completed all 24
registered audits with zero rejection. Its largest maximum e-value was
`31.856806` against the unchanged `2400` boundary; the overall geometric mean
maximum e-value was about `0.0068` times the immutable P3G.5 value. Dataset and
runtime hashes matched P3G.5, P3H.1 passed before data loading, and candidate
responses, acquisition and held-out remained closed. The terminal status is
`CALIBRATION_COMPATIBLE_ACQUISITION_BLOCKED`, not an efficacy or calibration
proof. Only a response-free transformed-law acquisition correctness phase is
authorized next. See `docs/pcpi_p3h2_real_calibration_result_20260825.md`.

P3H.3 replaces the old Student-t class-EIG object with the mutual information
of a marginal-preserving KL-projected joint. Directly combining the P3H
response density with the old Student-t class responsibilities would generally
change the frozen class posterior and is forbidden. The deterministic Gate
preserves the exact P3H grid marginal and frozen class marginal, reduces
numerically to the base EIG under the identity residual law, changes under a
nonidentity residual law, and returns no selection when nested quadrature
intervals overlap. Fine/coarse radii remain asymptotic numerical diagnostics,
not finite-order theorems. P3H.3 accesses no real data or response and does not
authorize operational acquisition. See
`docs/pcpi_p3h3_semiparametric_acquisition_contract_20260825.md`.

The frozen P3H.3 Gate then passed all nine decisions on commit `ef8ffcf9`.
Targeted P3H.3, prior acquisition and discrepancy regressions passed 13/13,
63/63 and 8/8 respectively; the complete repository passed 674/674. No real or
simulated data were accessed. The transformed fixture changed the class-EIG by
up to `0.08693`, both projected marginals met their frozen tolerances, and the
overlapping fixture returned no selection. The status remains correctness-only
and operational execution remains unauthorized. See
`docs/pcpi_p3h3_correctness_result_20260825.md`.

## P3H.4 likelihood-power-specific residual states

P3H.4 freezes the missing mapping between the robust likelihood-power family
and P3H.3. Every preregistered `SequentialReferencePosterior(eta)` now
reconstructs its own raw PIT sequence from the same chronological already-opened
history, using only its strict prefix at each step. The resulting posterior and
residual law are bound by exact engine/posterior object identity; the scorer no
longer accepts an unlabelled residual-law tuple. Raw X/y history is erased from
the bundle after reconstruction, leaving an observation count and commitment.

This also closes an overclaim: P3H.2's 24/24 compatibility result is for its
eta=1 structurewise discrepancy target and is not transferred to the finite
likelihood-power reference family. P3H.4 is correctness-only and keeps real
acquisition blocked. The next admissible phase is a separately frozen
family-wide calibration-only Gate with candidate and held-out responses closed.
See `docs/pcpi_p3h4_likelihood_power_residual_family_contract_20260826.md`.

The frozen P3H.4 Gate passed all nine decisions on commit `c702d408`. The
three correctness-fixture powers produced distinct own-target PIT sequences;
raw history was absent from the acquisition bundle; exact object binding
rejected a separately refitted equivalent posterior; and chronological order
changed residual states while final batch posterior probabilities agreed to
`8.61e-16`. The combined P3H.3/P3H.4 suite passed 22/22 and the complete
repository passed 683/683. Both P3H.2-to-family transfer and family-wide
calibration remain explicitly false, so real acquisition is still blocked.
See `docs/pcpi_p3h4_correctness_result_20260826.md`.

## P3H.5 family-wide calibration-only Gate

P3H.5 freezes a 96-coordinate corrected-PIT audit over three datasets, eight
seeds and likelihood powers `[0.125, 0.25, 0.5, 1.0]`. The global false-alarm
budget remains `1/100`; equal allocation is therefore `1/9600` per coordinate,
with boundary 9600. Reusing P3H.2's 2400 boundary is forbidden because it would
not close the expanded familywise budget.

The initial 32 observations are now role-separated before new real results:
the first 16 fit the response standardizer, design transform and base warmup;
the final 16 form the strict-prefix residual-training sequence. This removes
the hidden dependence that would arise from fitting a target standardizer on
future responses inside the same claimed prequential residual sequence.
Validation states are used only for the family calibration audit and discarded
afterward. Candidate rows contribute only a subset commitment; candidate
responses, acquisition and held-out remain closed for both terminal outcomes.
See `docs/pcpi_p3h5_likelihood_power_family_calibration_contract_20260826.md`.

The unique P3H.5 real run on commit `53c081f6` completed all 96 registered
coordinates with zero rejections at the fixed boundary 9600. Its terminal state
was `FAMILY_CALIBRATION_COMPATIBLE_ACQUISITION_BLOCKED`; manifest, source,
config, runtime and data hashes all verified. The largest maximum e-value was
`164.9035`. One Gas CO coordinate (seed `2026080707`, eta `1.0`) retained a
terminal e-value `137.5947` with PIT variance `0.06410`, so this descriptive
local concentration signal remains visible without changing the registered
decision or selecting eta post hoc. P3H.5 validation-updated states remain
discarded and real acquisition is blocked pending a P3H.6 operational-lifecycle
correctness Gate. See
`docs/pcpi_p3h5_likelihood_power_family_calibration_result_20260826.md`.

## P3H.6 operational semiparametric lifecycle

P3H.6 repairs the production-source gap left after family calibration. A typed
four-model lifecycle now reconstructs only the base-warmup and strict-prefix
residual-training history, binds candidate scoring to those exact posterior and
residual-law objects, freezes a response-free selected ID/action/state hash,
and admits exactly one matching revealed response before advancing every target
once. Changed histories, wrong candidates, stale decisions, incomplete powers,
uncertified transformed rankings and legacy fallbacks fail closed.

The historical real runner can consume this lifecycle only under an explicit
new protocol flag and only for PCPI. It records before/after family hashes per
query while historical protocols retain their old paths. This is a
correctness/source-composition repair, not real efficacy evidence. A separate
formal configuration must still freeze the P3H.5 16/16 roles, fixed four-power
family and matched comparison before user execution. See
`docs/pcpi_p3h6_operational_semiparametric_lifecycle_20260826.md`.

## P3H.7 formal semiparametric real acquisition freeze

P3H.7 freezes the first matched-budget real runner that enables the P3H.6
lifecycle. The three registered targets, eight seeds, 32 initial observations,
32 queries, 128 candidates, 256 validation rows and four policies remain
matched. The response/design transforms use only the first 16 initial rows;
the remaining 16 initialize each power-specific residual law. Every policy
reports with eta one, while PCPI alone retains the complete four-power maximin
family. SafeBayes selection and all uncertified/fallback PCPI actions are
forbidden.

The canonical runtime is checked before output creation or data access. Every
PCPI query must bind a response-free decision and form an uninterrupted family
hash chain after its single oracle reveal. All 96 policy runs must complete;
failures are recorded without replacement or retry. This freeze authorizes only
the user's unique held-out-closed real-development execution, not Codex
execution or any confirmation claim. See
`docs/pcpi_p3h7_semiparametric_real_acquisition_protocol_20260826.md`.

## P3H.7 failure and P3H.8/P3H.9 repair

The user stopped P3H.7 in run 92/96 after four PCPI runs had already failed
closed because final transformed-utility intervals overlapped. The incomplete
archive is protocol-invalid and supplies no efficacy claim. Its audit is frozen
in `docs/pcpi_p3h7_failure_audit_20260828.md`.

P3H.8 replaces no utility and changes no data coordinate. It makes the
numerical decision total: interval-dominated candidates are removed; an
unresolved possible-maximizer set is resolved by the existing covariate-only
representative MMD and then global candidate ID. Primary separation and
secondary resolution are logged separately. Nested quadrature also reuses the
preceding fine look as the next coarse look without changing numerical results.
The theory and leakage boundary are in
`docs/pcpi_p3h8_interval_frontier_resolution_20260828.md`.

P3H.9 freezes the corresponding failure-informed real-development rerun. It is
not independent confirmation; its protocol is documented in
`docs/pcpi_p3h9_interval_resolved_real_protocol_20260828.md`.

The returned P3H.9 run completed all 96 policy runs and 3072 queries with zero
execution failures, a valid evidence chain, verified official data hashes and
held-out closed. The protocol Gate passed, but the terminal assessment was
`OPERATIONAL_CLASSES_DEGENERATE_NO_CLASS_CLAIM`; strong predictive and
structural evidence were both false. This is a permanent development negative,
not a runtime failure, and it must not be rerun. Its frozen audit is
`docs/pcpi_p3h9_result_failure_audit_20260830.md`.

## P3I.1 decision-target alignment

P3I.1 repairs the C1/C3 mismatch diagnosed from P3H.9 without fitting a new
threshold or inspecting held-out data. Operational classes now use the frozen
posterior mean profile under squared prediction loss, standardized by one
common `H0` posterior-mixture scale. Observation-noise and residual quantiles
remain nuisance and cannot create a scientific-law class by themselves.

The P3H corrected marginal is composed with the base class/response joint by a
common strictly monotone PIT transport. Mutual information is invariant under
that transport, so P3I does not reuse P3H.3's optional KL-projected copula to
make marginal calibration change class-EIG. The primary utility is the robust
lower envelope of frozen-class EIG only; the old conditional Gaussian-moment
EPIG nuisance term is exactly zero. Representative safety and registered
interval-frontier resolution remain response-free constraints.

The P3I.1 correctness suite passes 7/7 and its six source/algebra decisions
pass. It accesses no real, validation, candidate-response or held-out data and
does not authorize an operational run. See
`docs/pcpi_p3i1_decision_alignment_contract_20260830.md`.

## P3I.2 decision-targeted source composition

P3I.2 composes the P3I.1 class constructor and target-only robust class-EIG
into the historical measured-pool runner under an explicit protocol flag.
Every policy shares the new initial-frozen decision-regret partition; PCPI
alone receives the exact likelihood-power posterior/residual family and is
dispatched to the P3I scorer before either legacy joint scorer. Query, run,
manifest and assessment records expose the class metric and scale hash,
transport-invariance identity, zero conditional EPIG and target-only usage.

The primary assessment is now frozen-class log-risk reduction. Predictive
RMSE/NLL remain secondary reported outcomes and cannot be folded back into the
primary decision. The first execution failure is fsync-recorded in a unique
terminal file and immediately aborts instead of allowing later runs to hide it.
The P3I.2 candidate config is nevertheless hard-blocked before data-path or
output access. Its eight source decisions and focused 50-test regression pass,
and the complete repository regression passes 749/749, but it authorizes no
real execution. See
`docs/pcpi_p3i2_source_composition_contract_20260831.md`.

## P3I.3 supervised real-development freeze

P3I.3 promotes the blocked P3I.2 candidate to one user-only formal execution.
After removing schema, stage and the authorization boolean, the two configs are
identical. Thus no scientific method, budget, seed, numerical schedule, class
threshold, utility or assessment rule changes at authorization time. The run is
failure-informed development, not independent confirmation.

The supervised launcher checks the exact final branch, commit, tree and config
hash; isolates only the user's historical untracked evidence; refuses existing
outputs; streams fsync-backed progress; preserves stdout/stderr; exposes the
first terminal failure; stops its child on interruption; and restores evidence.
Success additionally requires 96/96 runs, no failures, a passed protocol Gate,
valid evidence registry, exact manifest identity and held-out closure. The
response-free Gate passes eleven decisions, the focused regression passes 55
tests, PowerShell parsing passes and the complete repository regression passes
764/764. Codex execution remains forbidden. See
`docs/pcpi_p3i3_real_execution_protocol_20260831.md`.

## P3I.3 protocol failure and P3I.4 shared-H0 repair

The unique P3I.3 run completed all 96 policy runs and 3072 queries with zero
execution failures, valid evidence and held-out closed, but its protocol Gate
failed exactly one decision: the initial frozen class-partition hash was not
shared across policies. In every one of 24 dataset/seed groups, the three
baselines shared one hash and PCPI had another, even though class counts agreed
in all groups and initial entropy differed by at most `3.02e-14`. P3I.3 is
therefore immutable `INVALID_PROTOCOL_FAILURE`, not efficacy evidence. See
`docs/pcpi_p3i3_protocol_failure_audit_20260831.md`.

P3I.4 fixes the actual state boundary rather than weakening identity. It
constructs one complete-initial-history eta-one posterior and decision-regret
class target per dataset/seed before the policy loop, then injects the same
typed immutable target into all policies. PCPI retains its four strict-prefix
prequential residual states; its eta-one sufficient statistics must be
numerically equivalent to the shared target under a response-independent
floating accumulation bound. Material mismatch fails closed, and class hashes
are neither rounded nor copied.

P3I.4 changes no data coordinate, seed, budget, utility, quadrature schedule,
threshold, assessment boundary or tie break. Its PowerShell 5.1 supervisor also
replaces the redirected `Start-Process` composition that produced a null exit
code with an encoded child whose nonzero exit code is preserved. The original
canonical CPython 3.11.9 environment and all historical binary identities are
retained.
See `docs/pcpi_p3i4_shared_h0_real_protocol_20260831.md`.
