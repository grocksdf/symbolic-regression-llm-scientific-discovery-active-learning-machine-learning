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
