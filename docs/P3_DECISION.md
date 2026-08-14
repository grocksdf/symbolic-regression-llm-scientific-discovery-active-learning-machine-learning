# P3 decision — representative-safe maximin joint acquisition

Status: **P3B.10 and P3C.1 real efficacy failed. P3C.1 passed its controlled
implementation Gate and completed the unchanged held-out-closed real run, but
returned `REAL_ADVANTAGE_NOT_DEMONSTRATED`. Mainline remains NO-GO; P4/P5,
held-out confirmation, and submission claims remain blocked.**

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
