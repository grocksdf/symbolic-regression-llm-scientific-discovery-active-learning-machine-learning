# Common-target transfer audit and M-open design contract

Status: research implementation and response-free correctness only. The
2026-09-29 realized pilot and MatSci confirmation remain immutable failures.
No new real-data experiment or held-out access is authorized by this note.

## Why the observed gap is structurally possible

1. `bank_selection.py` ranks feasible portfolios by the lower bound of *their
   own* expected class-risk reduction. Adding a plausible but false LLM law
   can create classes that are easy to discriminate under that bank's own
   predictive distribution. Numerical integration certifies that conditional
   calculation, not the predictive law that generates real observations.
2. `freeze_discovery_target()` computes a fresh H0 partition for each selected
   bank. A Full-versus-No-LLM difference in normalized posterior Bayes-risk
   AULC therefore compares different class spaces and different initial-risk
   denominators. It is not a paired loss difference on a common scientific
   state. Within one bank, targeted-versus-random does share a class map.
3. `run_realized_drr_trajectory()` calls the change in its own posterior risk
   after seeing a real response “realized risk reduction.” This can fall even
   when the posterior confidently selects the wrong law. It is a posterior
   functional after a real update, not external decision regret.
4. Source stacking optimizes weights and checks empirical fold gains on the
   same out-of-fold profile. The fold check is an in-sample constraint on the
   chosen weights; it provides no population guarantee after portfolio search.
   A core reserve and matched-random numerical fallback are useful safeguards,
   but neither establishes Full > No-LLM on new responses.

These are testable mechanisms, not a claim that one explains every observed
task. In particular, the 2026-09-29 MatSci confirmation did not call the LLM,
so it diagnoses acquisition transfer separately from LLM synthesis.

## Implemented first boundary: an external common response target

`run_realized_drr_trajectory(..., reporting_data=..., 
reporting_excluded_from_selection=True)` optionally records predictive MSE
and mean log predictive density after every prefix. The role must be
`VALIDATION`; exact development/action response rows cannot overlap it. The
caller must independently establish that these reporting responses were
excluded from candidate generation, source admission, bank selection, and
action ranking. The routine never uses them in those operations.

`compare_paired_reporting(full, no_llm)` refuses a contrast unless the
reporting fingerprint, initial-data fingerprint, action-covariate fingerprint,
and budget match. Its raw MSE and log-score AULC contrasts have the same units
and the same observed target in both conditions. MSE is the squared-loss
decision risk for a predictive mean, while log score checks the full response
law. Neither is a ground-truth symbolic-class label. No condition-specific
initial-risk division is applied. A freeze must also bind row identities and
the reporting set's exclusion from every selection step; a role flag alone
does not prove this. Existing benchmark freezes do not provide this contract,
so the optional report is deliberately not wired into their runners.
In the benchmark snapshot, `drr_adapter.split_training_samples()` already
assigns 0--50% to discovery development, 50--70% to discovery validation,
70--90% to inference H0, and 90--100% to acquisition covariates. The existing
discovery validation segment cannot be silently reused for independent
reporting. A new prospective split must reserve distinct training rows before
candidate generation; test/OOD and untouched held-out remain closed.

## Proposed algorithm after the audit, before any new efficacy freeze

The research target should be specified on a **common registered population**
`nu(x)` and a fixed loss `ell(d,y,x)` for both candidate banks. For predictive
decisions, a predeclared choice is squared loss over `nu`; if the scientific
claim instead concerns class recovery, one must first define a class label
independent of the bank under comparison. The present per-bank clustering
cannot supply that label without further assumptions.

Construct an M-open predictive family from the protected core, admitted LLM
structures, and an explicitly calibrated residual/discrepancy component.
Estimate its predictive adequacy with source-blind, action-aware cross-fitting
on data that did not rank the portfolio. The action-aware check must cover
the registered target population and plausible acquisition covariate shift,
not just average H0 log score. If the diagnostic is unresolved, retain the
core/reference policy and report abstention. Do not call this a no-harm
guarantee unless the uncertainty set demonstrably covers the real response
law.

For a *frozen* common decision problem and independently constructed
plausible-law set `Q_t`, a candidate objective is

```
V_t(a) = inf_{q in Q_t} E_{Y_a ~ q}[R_q(p_t) - R_q(p_{t+1}^{a,Y_a})]
```

where the update and loss must refer to one well-defined target. The robust
score must be evaluated at the same computation budget for Full and No-LLM.
`robust_common_target.py` implements the finite-law interval algebra:
the lower and upper bounds for the worst-law value are the respective minima
across laws, and a strictly separated positive leader is selected only when
every law declares the identical target identity. An empty or unresolved law
set returns the registered matched-random action. This module is not connected
to the measured runner: the current per-bank class values have different
target identities and there is not yet an independently calibrated plausible
law set. Plugging those class scores into this module would violate its
contract.
Numerical rank intervals decide only numerical ordering. If `Q_t` is empty,
not independently calibrated, or all lower bounds are unresolved, use the
registered reference policy. In a finite acquisition budget, a response can
also test model adequacy: specify the target-versus-discrepancy tradeoff
*before* looking at development outcomes. Do not multiply post-hoc penalties
into the current utility or claim `min_q` alone proves external safety.

Implementation gates before real development:

1. Freeze one shared external evaluation population and loss; audit that
   Full/No-LLM share all response and budget identities.
2. On controlled correctness fixtures only, demonstrate that a false novel
   structure can increase model-relative class utility while worsening common
   predictive risk, and that the report detects it. This is not efficacy.
3. Establish source/discrepancy calibration with disjoint fit, model-selection,
   and audit roles, or use an explicit simultaneous selection-aware bound.
4. Validate the robust utility and fallback on exact finite references,
   including a correctly specified case, an omitted-law case, a covariate
   shift case, and a vacuous uncertainty-set case.
5. Freeze a genuinely new matched-budget protocol, with task-level paired
   independent loss, failure/abstention accounting, and no use of the prior
   failed confirmation tasks as a success criterion. User executes any real
   experiment. Existing results remain in the evidence index.

Research context: Tang, Sloman and Kaski, *AISTATS 2026*, identify
representativeness and error amplification under BOED model misspecification
(https://proceedings.mlr.press/v300/tang26d.html). Sloman et al., *UAI 2024*,
show how nuisance misspecification can negatively interfere with targeted
Bayesian active learning (https://proceedings.mlr.press/v244/sloman24a.html).
Go, Qian and Yoon (2026) motivate decision-aligned rather than parameter-only
design (https://arxiv.org/abs/2605.26093). Murphy (2026) studies an M-open
LLM proposer with predictive checks and later model expansion
(https://arxiv.org/abs/2608.09696). These works motivate the contract; none
establishes an automatic external no-harm guarantee for the present code.
