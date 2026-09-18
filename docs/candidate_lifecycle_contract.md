# Scientific system candidate lifecycle repair

The Gas-CO r3 bank passed numerical capacity checks but failed composition:
full lacked a retained MCTS hypothesis and single-engine lacked a compatible
LLM hypothesis. This is not evidence of efficacy or dataset unsuitability.

The comparison bank now uses all successfully evaluated hypotheses rather
than a truncated incumbent leaderboard. Incumbent selection and its existing
dominance/safety policy are unchanged. Raw engine candidates, canonical seed
registry, evaluation rejections, provider protocol audit, proposal expression,
refitted expression and post-refit PCPI adapter rejections are preserved.
An engine that executed but whose candidate was rejected is not relabelled as
a retained engine contribution. Missing LLM hypotheses still fail composition.

PCPI compatibility is checked on the actual refitted expression. A valid
proposal may become incompatible when global constant fitting introduces a
scale or shift inside a nonlinear primitive. Such expressions are rejected,
not silently rewritten to an unscaled basis. Future prevention requires a
structure-preserving refit contract, independently validated before a real run.

The opt-in `pcpi-closed-basis-amplitudes` refit policy now checks the input
closed support, fits only external additive amplitudes using the existing
amplitude solver, and checks the output support. The intercept is explicit;
zero/constant columns can disappear, but no nonconstant support can be added.
Internal nonlinear literals and exponents are not optimized. All evaluation,
pruning and structure-ablation refits share this policy through EquationRuntime.
Unknown policies and unsupported input expressions fail closed. Legacy global
constant fitting remains the default for historical configurations. This repair
does not guarantee retained LLM/engine composition or a positive real result.

Existing negative outputs and freezes remain immutable. These code changes
are correctness repairs only; they authorize no new real experiment and do
not establish scientific superiority. All screening and system claims remain
conditional on the registered development split and finite hypothesis bank.

Closed-basis protocols reserve one first-evaluation slot for each canonical
initial seed before auxiliary pruning can spend the deterministic budget.
Oversized banks fail before evaluation instead of silently starving later
engines. Every refit and pruning trial remains charged to the unchanged total
cap; successful evaluation or source retention is never guaranteed. The LLM
reserve is identical with and without a provider in this opt-in protocol;
unused no-LLM reserve is not reassigned to extra deterministic work. Legacy
protocols retain their historical allocation. Canonical aliases remain in the
registry but do not count as distinct retained engine contributions.

Exploration always retains the finite, aligned current prediction as its
mandatory `y_hat` baseline, including constant and near-constant predictions.
Only optional grammar primitives are subject to low-variance filtering.
The baseline values are copied exactly, never clipped or replaced; an exactly
constant centered column contributes zero and the existing fitted intercept
supplies its baseline. Invalid predictions fail explicitly before grammar
construction. This handles a legal incumbent state rather than forcing a
nonconstant hypothesis or bypassing bank viability/composition requirements.

LLM candidate audit scores are nullable by meaning, not by numeric cleanup:
invalid candidates use `score=null` with `invalid_candidate_no_score` and their
original rejection records; every budget-denied proposal uses `score=null`
with `not_evaluated_budget_exhausted`. Evaluated candidates require finite
scores or fail closed. The strict publisher validates before opening staging,
so invalid evidence creates neither a completed result nor a staging file.
No historical failed output is changed by this reporting repair.
