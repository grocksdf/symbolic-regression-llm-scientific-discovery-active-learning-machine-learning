# Bank-location guidance adapter (development code only)

`scripts/bank_guided_search.py` integrates the mainline predictive feedback,
existing AST editor, existing crossfit Bank scorer and independent evaluation.
It exposes `run_guided_comparison`, explicitly requiring contract
`bank-location-guided-search-v1`, modes Bank/residual/blind and a declared
quality or mechanism budget design. No executable benchmark CLI or implicit
dataset reader is installed; importing the module cannot run an experiment.

Inputs:

- frozen engine Bank E0 and frozen structurally unique candidate archive;
- fitting, left-search and right-search roles with bound source row IDs;
- unopened independent development evaluation callback returning IDs and data;
- `SearchBudget` per mode, seed, grammar operators and a new output directory.

Each component fits on the same development data. Bank location sensitivity
and MAP residual use those fitting rows only. The Bank signal affects the
actual chosen donor expression. Parent/site coverage is shared; feasibility
may differ after the changed donor, so this is not an identical-pool replay.
The generator still uses standard crossfit mixture gain for candidate selection.
It does not require learning an operator policy before finding any useful edit.

The all-arms freeze precedes evaluation. Outputs include `PLAN.json`,
`GENERATION_FEEDBACK.json`, per-arm attempt traces and generation audits,
`GENERATION_FROZEN.json`, and either `RESULT.json` or `FAILURE.json`. No old
result or freeze is overwritten. Costs include initial scorer work and shared
preparation; candidate errors remain missing outcomes, not imputed rewards.

The current adapter does not call a provider or PySR and does not implement a
matrix registration. Its callable inputs must come from a newly reviewed
development protocol before any real run. It is not permission to open final
confirmation, test/OOD or reuse historical execution authorizations.

Mainline `docs/bank_guided_generation_contract.md` documents the derivative,
LLM runtime hook, scope, budget semantics and research limitations. Ordinary
residual here means the fitted MAP component residual, not the old unfitted
raw-expression baseline. No real superiority or novel algorithm claim is made.
