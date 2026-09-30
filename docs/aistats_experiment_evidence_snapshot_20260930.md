# AISTATS experiment evidence snapshot

Snapshot date: 2026-09-30

This tracked document is the compact GitHub-facing index for the latest
experiment evidence.  Full run directories remain local under
`D:\01\666\outputs` and are intentionally ignored because they contain large
logs, repeated candidate banks, and intermediate checkpoints.  Every indexed
artifact is bound by an absolute path and SHA-256 identity.

## Matched-engine synthesis efficiency

Artifact:

`D:\01\666\outputs\scientific_aistats_synthesis_efficiency_audit_20260929\AISTATS_SYNTHESIS_EFFICIENCY_AUDIT.json`

SHA-256:

`db9740b9e1bf6a921090cbbce4119b5d0cbd9d1aef1294dcbc94c27b4f0d8536`

Status: `passed=true`

- 16 paired coordinates and 32 condition runs were accounted for.
- Full and No-LLM used the same 12 engine jobs per run.
- Both conditions shared the same candidate-evaluation limit.
- Full used 36 candidate evaluations on average; No-LLM used 48.
- Full made 63 logical LLM calls; No-LLM made zero.
- Mean Full-minus-No-LLM normalized AULC was `+0.026007580312139643`.
- Candidate responses and held-out remained closed.

Claim boundary: matched-engine development resource accounting only.

## Conservative BOQD family replay

The detailed immutable record is:

`docs/scientific_operational_qd_family_calibration_result_20260929.md`

Status: passed response-free replay.

- Mean BOQD effect: `+0.04082346601706559`.
- Positive families: population growth, transformed laws, materials science.
- Chemical reactions: tied.
- Negative-transfer families: zero.
- Failures: zero.

Claim boundary: response-free development replay, not realized efficacy or
held-out confirmation.

## Fresh-task MatSci acquisition confirmation

The detailed immutable record is:

`docs/scientific_aistats_matsci_acquisition_confirmation_result_20260929.md`

Status: failed registered confirmation Gate.

- Mean targeted-minus-random effect: `+0.05960731340110615`.
- Positive tasks: `MatSci6`, `MatSci8`.
- Negative tasks: `MatSci16`, `MatSci25`.
- All 16 trajectories completed without execution failure.
- No LLM, test/OOD, or held-out access occurred.

The negative result is permanent.  It does not authorize task replacement,
seed replacement, threshold relaxation, or another search for acquisition
superiority.

## Certified matched-random fallback repair

Artifact:

`D:\01\666\outputs\scientific_realized_drr_safe_fallback_gate_20260929\REALIZED_DRR_SAFE_FALLBACK_GATE.json`

SHA-256:

`ef898782421213aea3eb393f75e8dbbce0e0ce198c75cad97722acfc5fabcb50`

Status: `passed=true`

The correctness Gate proves:

- zero or unresolved utility executes the frozen matched-random action;
- positive but overlapping intervals fail closed;
- only an interval-separated positive utility can execute targeted
  acquisition;
- the historical MatSci confirmation is not repaired or relabeled.

Claim boundary: algebraic correctness only.

## External LLM-SR comparison

Immutable continuation result:

`D:\01\666\outputs\scientific_aistats_external_llmsr_continuation_20260930\AISTATS_EXTERNAL_LLMSR_RESULT.json`

SHA-256:

`1cd444e4aaf06ead9f074ce4e0eb3575ce2ca31e96cb3cd93294521489254731`

Read-only reporting correction:

`D:\01\666\outputs\scientific_aistats_external_llmsr_reporting_correction_r1_20260930\AISTATS_EXTERNAL_LLMSR_REPORTING_CORRECTION.json`

SHA-256:

`fae7c662f92fe5a14f82071199083e040c41dc7c5e3914ce6e96e70a8dfcd677`

Corrected accounting:

- Full rows preserved from the frozen source: 8.
- No-LLM rows preserved from the frozen source: 8.
- Infrastructure-corrected LLM-SR rows: 8.
- Total rows: 24.
- Total failure-inclusive failures: 5.

Aggregate results:

- Full ID NMSE: `25.000107383950247`.
- No-LLM ID NMSE: `25.000107383950247`.
- LLM-SR ID NMSE: `13.97058535594259`.
- Full OOD NMSE: `33.1713572942012`.
- No-LLM OOD NMSE: `33.1713572942012`.
- LLM-SR OOD NMSE: `75.26712414429971`.
- Full mean search time: `132.33111143112183` seconds.
- No-LLM mean search time: `75.28101289272308` seconds.
- LLM-SR mean search time: `239.55519637465477` seconds.

Task-family comparisons, lower NMSE better:

- ID: Full wins 3 families; LLM-SR wins 1; ties 0.
- OOD: Full wins 3 families; LLM-SR wins 0; ties 1.
- Full and No-LLM are equal on all four families for ID NMSE, OOD NMSE,
  ID strict accuracy, and OOD strict accuracy.

Interpretation:

- the multi-engine backbone is externally competitive and has better OOD
  performance on most registered families;
- this experiment does not demonstrate an external LLM-synthesis effect;
- the LLM admission layer is non-degrading on these tasks, but Full incurs
  additional runtime;
- the transformed-law Full and No-LLM timeouts and the single LLM-SR
  rate-limit timeout remain failure-inclusive evidence.

Claim boundary: frozen external ID/OOD comparison, not held-out confirmation,
universal superiority, or an external proof of LLM-synthesis efficacy.

## GitHub scope

The Git repositories should contain:

- source code and tests;
- protocol builders and runners;
- freeze/correction logic;
- paper source and vector figures;
- this evidence index and detailed audit records.

The Git repositories should not contain:

- API keys or `.env` files;
- full `outputs/` run directories;
- raw candidate responses;
- repeated intermediate checkpoints and provider logs;
- local paper archives.

