# S effective comparator: correctness contract

This change is for a **fresh prospective S comparator**, identified as
`effective-response-free-v1`. The public baseline's
`scripts/matched_gap_structural_generator.py` is preserved byte for byte.
Its `historical-slots-v1` policy remains the default. A frozen historical
NO-GO cannot be reinterpreted by selecting the new policy.

## Allowed inputs and planning

The planner reads parent expressions and lineages, covariate-only parent
dispersion, the gap identity and its proposal authorization, and registered
development/selection/action covariates. It accepts no response array. It
projects fields instead of copying bank or gap evidence. Admission,
reporting and held-out rows are excluded from numerical preflight. Current
bank membership can reflect earlier admissions, as in the existing iterative
bank protocol, but admission scores do not rank alternatives.

Each cycle retains the same three islands; the two-cycle coordinate therefore
has the same six nominal slots. Before each structural attempt is issued:

1. Compile parent supports. Invalid parents are individually audited. Collapse
   duplicate supports so aliases cannot invent a second structural parent.
2. Rank eligible parents by covariate dispersion and stable hash ties; the
   frozen parent cap is recorded, including exclusions.
3. Visit a deterministic cyclic ordering of all six operation families.
   `subtree_insert`, `unary_wrap`, and `bounded_power` require one parent.
   Addition, ratio and interaction require two distinct parent supports.
4. Predict the exact expression AST size and source length by substituting
   parent node counts into the operator template. Use the compiler's complete
   `ast.walk(ast.parse(..., mode='eval'))` convention, including the expression
   wrapper, operator and context nodes. Do not enlarge its budget.
5. Compile and check novelty, including the compiler's term-count/canonical
   representation constraints. Check finite-point definedness on the allowed
   covariates before fitting. The ratio, log and sqrt templates explicitly
   include the pre-existing `Abs` operations; no failed expression is repaired
   with an implicit Abs, clamp, epsilon or algebraic simplification.
6. Freeze at most one candidate for the slot. Later fit, semantic or admission
   failure consumes that attempt. There is no replacement or top-up.

The default budgets are 128 AST nodes, 4096 source characters, 32 ranked
unique parents, and 4096 planning checks per slot. Candidate checks include
infeasible/duplicate alternatives, not just successes. Bank compilation,
candidate compilation, preflight calls and submitted point evaluations are
also recorded. These costs must enter the **new** compute registration.
Finite-point definedness is not a proof over a continuous input box.

## Accounting and impossible coordinates

Retain separate counts for registered slots, gap-activated slots,
response-free feasibility abstentions, materialized candidates, fit survival,
validation survival, structure retention, admission evaluations and admissions.
A feasible, selected candidate is an effective opportunity. A gap-activated
slot with no feasible candidate remains visible as an abstention; it is not
silently removed from the coordinate or counted as a successful opportunity.
A planning cap miss is `planning-budget-exhausted`, not an adaptive retry.

This policy repairs opportunities wasted on unavailable parents or incompatible
budgets. It does **not** guarantee that every bank has a novel feasible
candidate, that every finite candidate can be fitted, or that L and S have
equal materialized/evaluable counts. The frozen matching basis is explicitly
`nominal-slots-with-response-free-feasibility`. Actual opportunity equality
needs its own prospective two-arm registration and complete funnel evidence.
Do not select coordinates or pad either arm after observing outcomes.

All operation families and unary operators remain in the grammar; controlled
fixtures cover their accessibility. This does not establish equal diversity
or candidate quality on real tasks. Retain operation/support diversity,
parent-limit exclusions and feasibility-abstention counts in a new experiment.

## Version and evidence binding

The new policy must be explicitly present in the prospective protocol, with
the exact `policy_contract()` object under `s_generator_contract` and
`historical_no_go_preserved: true`. Generator, composition and end-to-end
Gates must all record this policy, contract, and the same clean benchmark and
mainline source identities. The freeze hashes the new planner and contract.
The matrix forwards only that frozen policy; child artifacts must match it.
Missing policy means historical S. Crossed or incomplete identities fail
closed before data selection or child execution.

For the new no-data generator and synthetic composition Gates, add
`--generator-policy effective-response-free-v1` to the existing Gate commands.
Use the same flag on `check_matched_attribution_correctness.py` on the
registered host with its required Julia/provider dependency identities.
Correctness Gates have `execution_authorized: false`; a test result does not
authorize a real experiment.

## Frozen scientific boundaries

- LLM generation, prompts and candidate-validity implementation are unchanged.
- Independent admission implementation, thresholds, alpha allocation, reporting
  scoring and existing GO rules are unchanged.
- This patch does not audit or overwrite historical result packs.
- It supplies no evidence that S's extreme-negative candidates become useful,
  that L reasoning beats an equal-effective-opportunity search, or that the
  typed new-skeleton path caused a previous predictive advantage.
- Formal runs remain user-only, on fresh registered coordinates after new
  source/dependency/Gate identities are frozen.

The Python AST convention is documented at
https://docs.python.org/3/library/ast.html#ast.walk; the running compiler and
interpreter define the actual frozen node identity.
