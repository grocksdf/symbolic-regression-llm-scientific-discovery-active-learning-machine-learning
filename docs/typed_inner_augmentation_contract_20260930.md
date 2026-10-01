# Typed synthesis plus inner equation proposals: exploration contract

Status: opt-in source repair and response-free correctness checks. Historical
negative runs and their frozen 48-unit protocol are unchanged. This does not
establish Full > No-LLM in prediction or measured-pool decision risk.

## Failure chain and repair

1. The old `typed_evidence_synthesis` branch passed `None` to the inner
   `ProposalRuntime`, even if the outer Scientist had a provider. A new
   `typed_inner_augmentation` registration passes the configured provider to
   both. The inner call count and provider telemetry remain separately visible
   in the discovery report; the cycle total includes outer plus inner calls.
2. `UNION_SUPPORTS` and `AUGMENT_BASE` combine additive supports and can repeat
   parents. The opt-in Scientist review can request `INTERACT_SUPPORTS`. The
   compiler crosses terms already witnessed in **different engine lineages**,
   checks the PCPI closed degree-four basis, rejects an existing engine
   support, and records the exact parent identities. It does not accept
   free-form model equations or synthesize products of unregistered functions.
   Interaction is a candidate-generating hypothesis, never evidence of fit.
3. The old seed order could discard typed candidates before evaluation. In
   the new registration they bypass the engine seed list, are normalized
   independently, and are evaluated *after* the deterministic phase with
   their own ceiling. The inner equation phase follows with its own ceiling.
   All valid states remain in `evaluated_hypothesis_bank`; downstream source
   admission remains responsible for rejecting bad candidates.

## Explicit resource contract

For an exploratory `48E + L` setup, register the agent with, for example,
`discovery_budget=60`, `synthesis_evaluation_reserve=6`,
`llm_evaluation_reserve=6`, `typed_evidence_synthesis=true`,
`scientist_orchestration=true`, and `typed_inner_augmentation=true`. The
corresponding engine-only reference gets 48 total candidate-validation work
units. The three ceilings are 48, at most 6, and at most 6; unused extra
units are not moved to a later phase. This is an **unequal-compute candidate
augmentation audit**, not a matched-budget efficacy comparison. The outer
Scientist plan/review and inner proposer also consume provider attempts; raise
and freeze the provider attempt and wall-time limits accordingly.

The engine seed bank still reserves island exploration slots, so "48E" is a
validation-work budget, not a claim that 48 distinct engine expressions enter
the bank. A common-engine-bank causal test must verify identical engine result
identities in Full and No-LLM or project an engine-only bank from the same
Full source generation; Scientist dispatch otherwise changes the engine
frontier. The new budget audit reports `compute_matched=false` and only marks
`candidate_incremental_attribution_eligible=true` if the recorded raw engine
frontiers are exactly equal; a False value exposes the engine-scheduling
confound. The canonical
measured-pool runner rejects this exploratory mode pending a separate frozen
source-projection and decision-target correctness gate.

## Before a user-executed development experiment

Freeze dataset/task set, seeds, provider identity and attempts, engine jobs,
candidate/work budgets, objective/target, data roles, stopping/failure policy,
and the independent report set. On each cycle inspect
`scientist_review.synthesis_directives`, `evidence_synthesis.rejections`,
`supplemental_synthesis_evaluations`, `supplemental_synthesis_validated`,
`inner_llm_enabled`, `llm_call_count`, `llm_rounds`, the evaluation-budget
snapshot, and the evaluated bank by `source` and `lineage_id`. Report the
entire denominator: generated, compiled, attempted, validated, posterior
admitted, and decision-relevant candidates. A compiled interaction is not
necessarily an LLM-specific discovery: code supplies the grammar; the
Scientist supplies lineage selection and the typed instruction. Compare the
same code-owned interaction generator with non-LLM lineage selection before
assigning value to the Scientist's reasoning.

Only after candidate construction and independent predictive value are
established should the common-target candidate/region posterior expansion
and action-level utility be composed into a new measured-pool protocol.
Their existing reference implementation is not yet a deployed PCPI posterior
or a real-response benefit guarantee.
