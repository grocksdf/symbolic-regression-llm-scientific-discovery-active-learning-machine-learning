# P3B.6 returned-result audit

## Decision

Protocol Gate: PASS. Effectiveness Gate: FAIL. Entire submission Gate: NOT YET.
P3B.6 remains valid negative/promising development evidence and is not promoted
to a superiority result.

## Evidence integrity

- 96/96 registered policy runs completed; no failed or replaced seed.
- EvidenceRegistry: 97 events, valid hash chain.
- CCPP and Gas raw hashes and split hashes match the frozen registry.
- Gas CO and NOX are aggregated as one dataset family.
- Matched initial, acquisition, validation, pool, and candidate-evaluation budgets pass.
- `heldout_opened=false`; `selection_used_heldout=false`.

## Efficacy result

PCPI has lower mean predictive nAULC than random in both registered dataset
families, but the paired 95% intervals cross zero. It is not mean-better than
every baseline in every family. Frozen-class entropy gain is not significantly
positive versus random in either family, and the preregistered class-gain
negative-transfer condition fails.

## Root cause

The failure is primarily a class-definition problem, not a candidate-ranking
threshold problem. With the fixed distance radius of one pooled predictive
standard deviation, CCPP starts with mean frozen-class entropy
`7.33e-12` and maximum class probability effectively one. Consequently 254 of
256 CCPP PCPI queries use the epistemic fallback. Gas CO/NOX also begin with
only 1.75/1.25 operational classes on average. Increasing quadrature precision
cannot create information about a class variable already degenerate at the
start of acquisition.

## Correct repair boundary

P3B.7 replaces the arbitrary fixed radius with a task-agnostic budget-resolved
definition. For future measurement budget `B`, predictive laws are operationally
equivalent only when their per-action standardized RMS distance is at most
`1/sqrt(B)`, so the aggregate root-budget separation is at most one. This
definition depends only on the registered observation budget and declared
predictive metric; it does not inspect targets, dataset names, held-out data,
formula metadata, results, or task labels.

No P4, P5, motif, VED, or held-out confirmation is allowed until the P3B.7
correctness diagnostic and a new heldout-closed real rerun are audited.
