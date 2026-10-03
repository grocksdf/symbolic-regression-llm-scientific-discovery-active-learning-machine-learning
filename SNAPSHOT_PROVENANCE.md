# LLM-SRBench experiment adapter snapshot

Snapshot date: 2026-10-04

This branch is a clean source snapshot exported from:

- upstream project: `deep-symbolic-mathematics/llm-srbench`
- local source branch: `codex/restart-experiment-adapter`
- source commit: `2ed161ad` (Register the iterative Full / frozen-engine bank predictive matrix)

The snapshot intentionally excludes the upstream Git history because that
history contains experimental ledgers larger than GitHub's 100 MB per-file
limit.  It also excludes:

- `paper_locked_outputs/`;
- local `outputs/`, logs and caches;
- raw datasets;
- API keys and `.env` files;
- generated archives and checkpoints.

The branch contains the current benchmark adapters, frozen protocol builders,
tests, LLM-SR compatibility repairs, BOQD replay support, safe acquisition
fallback logic, and external-baseline reporting correction code, and the LLM admission offline audit.

The 2026-10-02 increment brings the branch up to the full local source
branch. It adds the three-arm predictive and realized gates with their builders, the frozen
expanded-formula-discovery adapter (runner, harness, three-arm child,
correctness gate and preflight), the four files from the 2026-10-02
development run that were previously published only out-of-band, and then the v2
prospective overlay: `scripts/expanded_formula_admission_v2.py`,
`scripts/formula_bank_materialization_v2.py`,
`scripts/provider_transport_preflight.py`, the prospective child/config/driver
pair, and their contract tests. `scripts/formula_recovery_contract.py` is now a
compatibility import of the versioned mainline recovery evaluator, so this
adapter requires mainline `8229551f` or later.

The 2026-10-03 increment brings the branch up to local source commit
`4ca5bb7b` (36 commits, 53 changed files: +4098 / -43, all under `configs/`,
`scripts/` and `tests/`). It adds the Formula Generation round-two
source-first engine-support audit: the preflight and freeze builders, the
gap-routing gate, the runner and its child, the stage-wise structural support
audit, the source-first bank adapter, `configs/formula_round2_source_first.yaml`,
and their regression tests. It also carries the Formula Discovery v2.10-v2.12
continuations (engine isolation, abstention continuation, evaluator-only
recovery), the parallel-schedule work recorded as rejected history, and the
provider transport repairs. `scripts/formula_generation_support_audit.py` and
the round-two child depend on the mainline divide-by-zero screen fix, so this
adapter now requires mainline `2c4e6c00` or later.

The 2026-10-04 increment brings the branch up to local source commit
`d9713bef`. It adds the iterative expanded condition: cycle-wise partitioning
of the existing train-only Gap and admission roles
(`iterative_gap_role_indices()` and the row-count-only reconstruction
`three_arm_role_indices()` in `methods/hypothesis_mvp_pcpi/drr_adapter.py`),
last-admitted-bank posterior-MAP export and cycle-role provenance in
`methods/hypothesis_mvp_pcpi/drr_searcher.py`, admission-response opening for
this condition only in `scripts/run_aistats_three_arm_formula_child.py`, the
paired config `configs/aistats_three_arm_formula_expanded_candidate.yaml`
(byte-identical to the mainline copy), its 15 synthetic role checks in
`tests/test_iterative_formula_roles.py`, and the post-generation reporting
tool `scripts/audit_iterative_bank_reporting.py`. The nine-role historical
split is unchanged, and this adapter now requires mainline `5f22eeca` or
later for the `core_rows_before` / `bank_rows_after` /
`posterior_map_expression` trace fields and the positive-quota expanded guard.

The 2026-10-04 second increment brings the branch up to local source commit
`2ed161ad` (1 commit, 8 changed files: +1002 / -4, all under `docs/`,
`scripts/` and `tests/`). It registers the iterative Full / frozen-engine bank
predictive matrix as a new protocol beside the untouched historical three-arm
predictive gate: `scripts/iterative_matrix_contract.py` (response-free identity
and config contract), `scripts/build_iterative_expanded_matrix_freeze.py`
(hash-selected fresh tasks after every supplied exclusion freeze, symbol-only
metadata, provider-env identity), `scripts/run_iterative_expanded_matrix.py`
(all child artifacts frozen before any train-reporting response is decoded, no
resume) and `scripts/check_iterative_matrix_correctness.py`, together with 8
response-free orchestration and integrity checks in
`tests/test_iterative_expanded_matrix.py` and the protocol note
`docs/iterative_expanded_matrix_v1.md`. The child output guard now rejects any
pre-existing path, including an empty directory left by a crashed process, and
the provider preflight records model, base URL and credential-file identity
without recording credential contents. The bank-only ablation has no matched
non-LLM new-skeleton proposal attempts, so the freeze, result and docs record
`formula_recovery_attribution_authorized: false`; no formula-recovery claim is
authorized by this protocol. The provider-transport, correctness-gate and
freeze JSON artifacts are runtime inputs, so they must be regenerated per
`docs/iterative_expanded_matrix_v1.md` before any matrix execution.

Nothing here rescores the historical rows. The 36-row expanded-formula
development run keeps its `passed=false` result; see
`docs/expanded_formula_benchmark_repair_20261002.md` and the mainline note
`docs/expanded_formula_failure_repair_20261002.md`.

Canonical scientific-system code and the compact experiment-evidence index
live on the repository's `main` branch.

