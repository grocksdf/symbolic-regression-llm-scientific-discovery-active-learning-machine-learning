# Posterior-gap-directed composition: PCPI research contract

Status: research interface and response-free correctness checks only. No
production provider call, formal experiment, revised gate result, or efficacy
claim. The earlier negative real-response result remains part of the record.

## Diagnosis is a decision problem, not a residual alone

RESTART diagnoses an incumbent equation through a correction
`g(f_t(x), x)` and gives the correction and successful structure snippets to
an LLM. PCPI has a different state: a frozen bank of structures and its
posterior. There is no single privileged `f_t`, and a mixture can be wrong
even if its members agree. A PCPI proposal should respond to **independent
evidence of a consequential bank inadequacy**, with between-model disagreement
as one exploratory cue, rather than treating disagreement itself as error.

For a frozen domain `X*` with registered nonnegative weights `nu` summing to
one, frozen regions `r(x)`, fixed operational classes `c(h)`, and the exact
finite-bank posterior `p(h|H0)`, let `mu_h(x)` be each structure's conditional
predictive location and `mu_c(x)` its posterior-conditional class mean.
`posterior_gap_diagnosis.diagnose_frozen_bank` reports per region:

```
B_r = sum_{x in r} nu(x) sum_c p(c|H0) [mu_c(x)-mu(x)]^2
W_r = sum_{x in r} nu(x) sum_h p(h|H0) [mu_h(x)-mu_{c(h)}(x)]^2
N_r = sum_{x in r} nu(x) sum_h p(h|H0) Var(Y|h,H0,x)
```

It stores conditional averages and region mass as well as `B_r` (the weighted
between-class term). `B_r + W_r` is the weighted variance of structure
locations; `N_r` is conditional predictive variance (requires Student-t
degrees of freedom greater than two). No pool or report response is read.
The action domain must exactly match the frozen target hash; weights and
region identity are separately recorded. This is model-relative uncertainty,
**not** a posterior probability that the bank lacks a law and **not** a
certificate of external decision value. Even `B_r = 0` cannot rule out a
shared, confident but wrong prediction. A large `B_r` might instead call for
measuring an informative action among *existing* classes.

The optional `proposal_brief` contains only region bounds on one covariate
axis, predictions of existing classes, top existing basis supports and
explicit uncertainty semantics. The current discovery runner does not send
it to an LLM. No single target-aware production loop is claimed.

## Required closed loop and data dependencies

1. **Freeze the comparison before proposal.** Declare one engine-only bank,
   covariate region map, weighted external prediction target, decision loss,
   eligible grammar, candidate and LLM budget, and task identities. Use
   separate roles for fitting the initial bank (`D_fit`), diagnosing proposal
   gaps (`D_diag`), admitting hypotheses (`D_admit`), and untouched reporting
   (`D_report`). Do not use current benchmark discovery validation as an
   untouched report: it already participates in selection. A fit on the same
   outcomes used to choose structures is conditional on that search and does
   not make an unconditional Bayes posterior over undiscovered laws.
2. **Establish an actual gap.** Freeze the engine posterior on `D_fit`, then
   score its predictive mixture on `D_diag` before reading any `D_admit`
   outcomes. Predeclare a localized proper-score or calibration comparison
   against a calibrated reference, accounting for region selection and
   multiplicity. If no independent inadequacy is established, report
   disagreement only and optionally acquire data to distinguish existing
   classes. Do not tell the LLM that an interaction or nonlinearity has been
   observed merely because predictions diverge. A blind spot may have small
   `B_r` yet large independent predictive error.
3. **Compose from a constrained diagnostic.** An LLM sees the supported
   region, class-predictive contrast, existing basis supports, and an
   independent error summary from `D_diag`; it proposes a typed structural
   correction, optionally of the form `g(mu_c(x),x)`. The compiler must
   materialize `g` with a frozen incumbent expression *before* closed-basis
   adaptation and verify that the resulting function, not just the text
   containing `y_hat`, is novel and legal. Current
   `ExplorationRuntime._admissible_grammar_expression` checks `y_hat*x`
   without substituting the incumbent, and `pcpi_adapter.structural_terms`
   distributes numeric but not symbolic outer products. Thus some useful
   `g` paths are currently unreachable. The standalone
   `closed_basis_composition.materialize_closed_basis_composition` gives a
   bounded, exact substitution-and-expansion reference with algebraic tests;
   it does **not** change production admissibility. Integrating it requires
   a new frozen method identity, complete budget accounting and lifecycle
   gates. Unsupported nested functions remain rejected.
4. **Separate the three acceptance arrows.** After generating hypotheses,
   freeze/refit both banks on the same `D_fit`. Compare each candidate's
   *independent* predictive contribution on `D_admit`, including selected
   region and multiple-candidate correction. Allocate candidate-specific
   prior mass prospectively, with the core engine bank intact in the
   quality-first `48E` versus `48E+L` diagnostic. Do not call the
   `candidate_region_expansion.py` selector the original PCPI law posterior:
   it is a separate categorical mixture of fixed experts. A genuinely
   region-specific symbolic posterior needs a joint generative model or an
   expressly modular estimand. Finally compare Full and ablated policies
   under **one** predeclared external loss and target, including numerical
   and plausible-law uncertainty; expected internal class-risk gain alone
   never certifies realized gain.
5. **Retain only audited lessons.** This is an upgrade of the existing
   `KnowledgeRuntime`, not a proposal to add a second library. Today the
   proposal payload already receives a verified structure library, accepted
   refinements and task-local memory; entries are staged on validated
   lineages, and persistent promotion requires a passing untouched
   confirmation. Its success measure is validation fit, however, not
   candidate-specific regional posterior or common-target decision effect.
   Extend its entry schema with the typed composed fragment, region and
   target identities, attempted-candidate count, independent admission and
   action-effect provenance, plus reasons for rejection. Keep task-local
   examples separate. Cross-task promotion for future confirmations requires
   task-disjoint provenance, declared capacity/merge rules and a policy that
   prevents a former sealed result from becoming a prompt for the same
   benchmark's later confirmation. An apparent gain on `D_diag` cannot
   validate the same fragment that `D_diag` selected.

## Ablations that distinguish the cause

Use the same initial engine 48, observation/report target, data roles,
candidate compilation, final capacity, measured-label budget and frozen
decision loss. Declare LLM requests, tokens, time and attempted structures
before the run. Record the additional compute of `48E+L` and report it;
quality-first establishes incremental hypothesis value, not equal-compute
superiority. A later matched-compute or matched-total-bank study addresses
deployment value.

| Arm | Proposal source | Question answered |
| --- | --- | --- |
| E | 48 engine structures | Shared control. |
| E + G | 48 engine plus grammar corrections generated without LLM | Is the diagnostic and compiler sufficient? |
| E + L-blind | 48 engine plus LLM proposals under the same proposal budget but no gap summary | Does targeted analysis improve over extra LLM candidates? |
| E + L-gap | 48 engine plus LLM proposals conditioned on independently supported gaps | Does the diagnostic increase admissible incremental value? |

Log the complete funnel per independent task: proposed, syntactically valid,
adaptable, structurally new, predictive-admitted, posterior mass by region,
decision-effect sign under the common loss, action differences, abstentions,
report MSE/log score, and actual paired decision loss. Keep the old negative
tasks as *development* stress tests; freeze all design choices before new
independent tasks. No promise that Full will universally exceed No-LLM: when
the engine bank is already adequate, a principled LLM should abstain, and
that outcome is part of the research result.
