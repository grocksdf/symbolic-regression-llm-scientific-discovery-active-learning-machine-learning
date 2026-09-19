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

The closed-basis refit protocol now derives the explicit symbolic generation
contract `pcpi-closed-basis-v1` for both production engines. Legacy protocols
remain unrestricted. MCTS applies the adapter validator during expansion,
before accepted-generation slots and scoring, including explicit seeds.
Unsupported candidates are rejected rather than projected. Polynomial-Lasso
screens its polynomial feature library by absolute train-only standardized
correlation and deterministic index ties before its single Lasso fit. Library
admission uses the existing adapter AST and text caps with conservative signed
scalar literals; every fitted nonzero coefficient is exported and the model's
predictor uses exactly that same library. No post-fit formula trimming, extra
Lasso fits, validation-based screening, or gate relaxation is permitted.
The fixed search, engine-job, discovery, provider and wall-time ceilings remain
unchanged. The composition gate still requires actual retained contributions;
compatible generation does not establish diversity, efficacy or superiority.
Signed external scalar amplitudes have the same basis token as positive ones,
including `cos(x0)*(-2)`; changing a nonlinear argument remains unsupported.
At the closed adapter boundary, finite nonzero numeric amplitudes multiplying
an additive group are distributed into an equivalent additive form before
amplitude refit and support extraction. This covers factored canonical forms
such as `c*(x8+sin(x0))` without distributing symbolic products, changing
nonlinear arguments, merging duplicate supports, or exceeding the existing
expression/AST caps. Prediction equivalence and before/after supports are
checked algebraically; invalid or duplicate structures still fail closed.

Post-run contribution auditing is artifact-only and immutable: it reads no
receipt response values, pool labels or held-out object. It verifies completed
policy manifests and exploration evidence chains, compares structural supports,
query sequences and published development curves, and labels four-query score/
RMSE correlations descriptive only. The completed additive-boundary pilot also
exposed a reporting-only error: the selection path used the current posterior,
but its `information_audit` repeated the frozen initial class entropy. New
decisions report entropy from the same current partition used by EIG scoring.
Historical decisions and curves remain immutable; their correction scope is
metadata-only and cannot upgrade the efficacy or superiority claim.

MCTS preserves a fixed-size candidate bank from one unchanged search job.
Under the closed-basis protocol, every node is scored only after the same
external-amplitude refit used by the PCPI adapter.  A deterministic two-fold
training-only cross-fit supplies the search reward and predictive signature;
the expression exported to the scheduler is refitted on all engine-training
rows.  Consequently arbitrary literals multiplying the same closed support
cannot alter search rank, and MCTS no longer optimizes a different model from
the one consumed downstream.  The exported bank is a deterministic
Pareto/max-min selection over cross-fitted error, structural complexity and
centered predictive nonredundancy; the best-error expression is always
retained.  The scheduler validates every expression and binds separate
lineage, but engine attempts, iterations, discovery evaluation budget,
provider budget and hard wall-time remain unchanged.  Downstream source
arbitration still uses a strict leave-MCTS-out bank and accepts MCTS only
through the preregistered decision-regret or independent predictive-log-score
channels.  This objective-alignment repair is not efficacy evidence and does
not authorize measured or held-out execution.

An immutable source screen may continue after a pre-decision numerical
failure only through the zero-response continuation contract.  The contract
hash-binds the original system contract, terminal records, exploration bank,
bank-viability report and dual-channel contribution report.  It rejects any
source tree containing a decision, receipt, development curve or completed
manifest.  Candidate expressions and their evidence registries are copied to
a new non-overwriting workspace; discovery engines and the LLM are never
called again.  The nested Gauss--Jacobi class-EIG rule chooses its first look
as `max(registered_minimum, 4 * predictive_structure_count)`, because its
fine/coarse allocation requires four nodes per structure.  This adjustment
cannot exceed the registered maximum or wall-time ceiling; an infeasible
maximum is a public response-free terminal failure.

Before bank viability and source arbitration, the scientific system now
compresses each finite proposal bank to exactly twice the registered
measurement budget.  It uses only registered H0 and acquisition covariates,
never acquisition responses, reporting-validation responses or held-out data.
A deterministic greedy objective maximizes frozen operational-class entropy
while preserving at least one retained hypothesis from every executed engine
and, when present, the LLM.  Candidate identity breaks exact ties.  The frozen
bank and its capacity audit are published separately from raw exploration, so
all rejected proposals remain available for provenance.  This changes neither
the posterior model nor the class definition; it prevents redundant proposals
from consuming finite-bank probability mass and quadrature budget.
When an immutable zero-response source predates only this capacity field, the
continuation may reuse its hash-bound raw exploration.  It applies the new bank
selection, recomputes viability and both source-contribution channels, and
enters measurement only if those new reports pass.  Any other configuration
difference fails closed.  Engines and the LLM are not rerun, and the source
output remains unchanged.

The entropy-only capacity selector exposed a second, immutable negative screen:
it increased the full Gas-CO operational-class entropy from approximately
`0.00139` to `0.09336`, but it retained only two members of the four-member
MCTS frontier and the resulting MCTS leave-source-out predictive contribution
was `-5.3251` nats on the independently registered arbitration split. This is
not a measurement result and does not authorize threshold relaxation.

The replacement capacity selector aligns selection with the already registered
proper-scoring source Gate without opening that Gate's validation responses. It
enumerates the finite capacity banks, preserves all registered provenance roles,
and uses a deterministic paired two-fold posterior-predictive log score on H0
alone. A bank is eligible for entropy maximization only when every applicable
registered source has strictly positive cross-fitted contribution beyond a
floating-point error bound. If no such bank exists, the best diagnostic bank
is published but viability fails before source-arbitration, acquisition-pool,
reporting-validation or held-out access. The folds, positivity rule, capacity
and entropy tie-break are frozen independently of observed efficacy.

The first predictive-safe continuation is retained as a zero-measurement Gate
failure.  The full bank passed both registered H0 safety checks (MCTS
`+0.6891` nats and LLM `+0.00121` nats), while the `no_llm` negative-control
bank showed an MCTS contribution of `-0.1911` nats.  The implementation had
incorrectly extended the full-bank leave-source-out admission rule to every
ablation.  The registration defines source admission only for the full
production bank; `no_llm` and `single_engine` are negative controls whose
purpose is precisely to expose lost contribution or negative transfer.  They
retain identical capacity and viability checks, but do not have to outperform
additional unregistered within-control ablations.  This scope correction does
not change any score, threshold, candidate, budget or observed artifact.
