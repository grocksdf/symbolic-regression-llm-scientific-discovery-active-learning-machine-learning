# Offline LLM admission audit result

Date recorded: 2026-09-30

Status: **completed read-only offline audit, `passed=true`**

This document explains the observed `Full == No-LLM` identity in the frozen
external LLM-SR comparison.  It does not rerun, rescore or replace that result.

## Frozen identity

- Audit certificate:
  `D:\01\666\outputs\scientific_aistats_llm_admission_offline_audit_20260930\AISTATS_LLM_ADMISSION_AUDIT.json`
- SHA-256:
  `333c8903adc21071b1b99ef376dbb4e42ec35e617732d61d4e73e94f4b9ed1be`
- Source result:
  `D:\01\666\outputs\scientific_aistats_external_llmsr_continuation_20260930\AISTATS_EXTERNAL_LLMSR_RESULT.json`
- Source result SHA-256:
  `1cd444e4aaf06ead9f074ce4e0eb3575ce2ca31e96cb3cd93294521489254731`
- Source continuation SHA-256:
  `c49e58daa8334f78b855611019fc912a734101a0081b7127f9baff3824a07cf2`
- Full artifacts read from:
  `D:\01\666\outputs\scientific_aistats_external_llmsr_r1_20260929\runs`

## Why Full and No-LLM are identical

Two independent mechanisms remove every LLM equation from the final selection.

1. **The inner equation-proposal path is not powered.**  These runs use the
   typed evidence-synthesis profile (`typed_evidence_synthesis: true` in
   `configs/aistats_drr_full_v6.yaml`).  `DiscoveryAgent._discover` passes
   `provider_settings=None` to `discover_from_selection` whenever that profile
   is active (`hypothesis_mvp/discovery/agent.py:440-442`), so
   `ProposalRuntime.enabled` is `False` and `ScientificDiscoveryRuntime.run`
   skips `_llm_search` entirely (`scientific_runtime.py:562`).  Every
   `runtime_events` trace is therefore
   `initialize > deterministic_explore > select > done` with no
   `llm_explore` phase.
2. **The Scientist's typed synthesis produced candidates that lost.**  This is
   the mechanism that actually deserves credit for the identity, and it is
   fully recorded per cycle in `scientist_agent_synthesis_audits`.

Across all six resolved Full runs `inner_llm_call_count` is `0` while
`scientist_agent_logical_llm_call_count` totals `21`.  The outer Scientist did
run; the inner equation proposer was never alive.

## Admission accounting

| Stage | Count |
|---|---|
| Scientist synthesis directives emitted | 19 |
| Passed novelty gate | 10 |
| Rejected (`no structural novelty`) | 9 |
| Compiled into candidates | 10 |
| Entered evaluated hypothesis bank | 5 |
| Rows in evaluated banks total | 102 |
| Synthesis rows in final top-k | **0** |
| Final selected source | `engine:*` only |

`Full` and `No-LLM` produced the same `best_expression` on all six resolved
coordinates.  Two further coordinates
(`lsr_transform/II.36.38_1_0/seed91`, `seed92`) have no Full artifact because they
timed out; they are declared missing and are not imputed.

## What this licenses

The admissible statement is narrow and precise:

> On this frozen external benchmark, the LLM Scientist emitted 19 typed
> synthesis directives, 9 were rejected for lacking structural novelty, 10 were
> compiled, 5 reached the evaluated hypothesis bank, and none reached the final
> top-k.  The final selected equations came exclusively from the deterministic
> multi-engine bank, so `Full` and `No-LLM` are identical by construction on
> every resolved coordinate.

This supports three paper claims:

1. **Selective admission is observable and non-catastrophic.**  Synthesis was
   attempted, partially admitted, evaluated and then rejected on development
   evidence, with `engine_candidates_preserved: true` in every cycle.
2. **The Scientist layer is non-degrading.**  Adding the LLM plan/review policy
   did not worsen any final external metric; it only cost runtime.
3. **The multi-engine body carries the external advantage**, independent of any
   LLM equation proposal.

## What this forbids

- It is **not** evidence that "LLM synthesis cannot improve final prediction
  accuracy", because free-form LLM equation proposal was never exercised in this
  protocol.  The confound is declared in the certificate under
  `inner_llm_equation_proposal_active: false`.
- It is **not** evidence that "admission blocked LLM candidates from lowering
  results", if that is read as candidates never arriving.  Ten did arrive; they
  were evaluated and did not win.
- It is not a held-out confirmation, an efficacy demonstration, or a universal
  superiority claim.
- The earlier interpretation that Full merely "filtered out useless LLM
  candidates" must be replaced by the accounting above.

## Consequence for the experiment program

Do not open another external task set to search for an LLM equation-proposal
gain.  The gap is a protocol gap, not an effect gap.  Either (a) report the typed
synthesis profile honestly as a resource-allocation and admission result, or
(b) register a separate condition with `typed_evidence_synthesis: false` so the
inner proposer is actually powered, and only then compare final external
metrics.  Option (b) is a new registered protocol and must be frozen before any
execution.
