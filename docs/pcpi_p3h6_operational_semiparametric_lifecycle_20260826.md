# PCPI P3H.6 operational semiparametric lifecycle

P3H.6 repairs the final source gap between the P3H transformed acquisition
utility and the measured-pool runner. Before this change, the historical runner
could reconstruct ordinary Student-t posteriors with `fit_batch` and a
SafeBayes-selected nominal likelihood power; it had no operational input for
the calibrated likelihood-power residual family. Thus P3H.3--P3H.5 could pass
without guaranteeing that a later acquisition decision used their object.

The production lifecycle now has one explicit state and two ordered operations:

1. initialize the complete powers `(0.125, 0.25, 0.5, 1.0)` from a base-only
   conditioning prefix and a strict-prefix residual-training history;
2. score candidate covariates with the exact four posterior objects and their
   own residual laws, freeze one certified response-free decision, reveal only
   its matching response, and advance all four targets exactly once.

The state recomputes the canonical history commitment from its stored opened
history. Equal-length but changed histories are rejected before scoring. A
decision binds the prior state hash, candidate ID and exact transformed action.
A different ID, changed action, stale/replayed decision, non-finite target,
incomplete power family, representative fallback or unresolved transformed
ranking cannot authorize a response.

The historical runner accepts this lifecycle only when its protocol explicitly
sets `semiparametric_lifecycle=true` and only for the PCPI policy. In that mode:

- the nominal reporting posterior is the eta-one member of the same family;
- posterior models are taken from the family rather than independently refit;
- the family is passed to the discrepancy-aware P3H scoring path;
- old posterior-epistemic and minimum-MMD fallback modes are not valid decision
  modes;
- the selected response is admitted through the bound decision transaction;
- before/after family hashes are recorded for every query.

The integration correctness fixture invokes the real `_run_policy` loop with a
target-sealing pool oracle and proves score--select--reveal--advance ordering.
Separate P3H.3 tests cover the actual transformed-law numerical scorer; the
lifecycle fixture isolates orchestration correctness and is not an efficacy or
simulation result.

P3H.6 still does not itself authorize a formal real run. The next freeze must
define a new protocol/config that enables this lifecycle, uses the P3H.5 16/16
initial roles, fixes all four powers without SafeBayes selection, preserves
matched baselines and budgets, and fails the complete run on any uncertified
PCPI ranking. No historical P3B/P3C result is reinterpreted.
