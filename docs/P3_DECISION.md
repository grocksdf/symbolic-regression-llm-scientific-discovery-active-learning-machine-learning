# P3 decision — representative-safe maximin joint acquisition

Status: **P3B.10, P3C.1, and P3D.2 real efficacy failed. P3E.1 repairs an exact
generalized-update decision mismatch, and P3E.2 adds a union-orthogonal
posterior-adequacy e-process. Both pass correctness fixtures only. Real
posterior adequacy remains untested; another acquisition run, P4/P5, held-out
confirmation, and superiority claims remain blocked.**

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

This does not show that CCPP or Gas passes or fails the adequacy Gate.
Non-rejection would not prove adequacy, and rejection would not by itself
validate the discrepancy-augmented posterior for acquisition. A separately
frozen, initial-development-only real adequacy diagnostic is the next possible
evidence step; another acquisition-efficacy run remains blocked. See
`docs/pcpi_p3e2_posterior_adequacy_repair.md`.
