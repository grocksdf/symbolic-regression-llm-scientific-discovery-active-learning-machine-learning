# Prospective repair package (not a runnable confirmation experiment)

`README.md` and the four original files remain byte-identical to the uploaded source archive. New files in this directory are explicitly prospective and do not modify or rescore the original `passed=false` run.

## New files

- `scripts/formula_recovery_contract.py`: imports the versioned mainline evaluator. It returns `literal_exact=null` for truth with unbound parameters, records topology separately, and returns not-evaluable for unsupported metadata. Explicitly registered true parameter values can support a separate instantiated exact assessment. Topology is not a fitted-parameter recovery claim.
- `scripts/provider_health_contract.py`: prompt ceiling without padding and fail-closed transport check. An empty ledger is tagged as no request attempted, not automatically an infrastructure failure.
- `scripts/run_aistats_three_arm_formula_child.py` and `configs/aistats_three_arm_formula_prospective.yaml`: prospective child/config with no padding, abort-on-provider-failure and the opt-in typed expanded grammar. A failed child records a transport or generation failure ledger before re-raising.
- `scripts/expanded_formula_admission_v2.py` and `scripts/formula_bank_materialization_v2.py`: the same mainline support parser selects candidates, screens folds and creates the frozen posterior. The stage audit keeps provider status, compiler reasons and novelty separate; it leaves `model_abstained=null` unless a semantic abstention is explicitly observed.
- `scripts/provider_transport_preflight.py`: one tiny, response-free request for a user-run quota/rate diagnosis. It never prints the provider token or raw error body; HTTP 429 without an explicit error code stays unresolved.
- `scripts/run_expanded_formula_discovery_prospective.py`: protocol draft with distinct freeze identity, an explicit task/input-symbol mapping, admission bank-contrast gate before truth opening and tri-state recovery denominators. Its source still contains an unconditional execution block after preflight because the full local benchmark branch is unavailable for integration verification here. Do not remove that block based solely on these unit fixtures.
- A new freeze must supply physically development-only metadata parquet files and `metadata_scope=physically-development-only-v1`; pointing the child at a full benchmark metadata parquet would expose sealed task metadata even if the evaluator later selects only development names.
- The child checks its non-truth input symbols against the frozen per-task mapping before generation, so the proposer and recovery evaluator cannot silently use different variable orderings.
- `tests/`: synthetic correctness fixtures. No benchmark truth, test/OOD or confirmation responses are used.

## Local checks

With SymPy, pytest, and the patched mainline on `PYTHONPATH`, run the tests in `tests/`. To run the full prospective benchmark later, first restore the complete missing local benchmark branch, verify the adapter forwarding and artifact schema, pass a response-free provider preflight, and freeze a **new** protocol. The original frozen experiment remains unchanged.
