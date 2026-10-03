# LLM-SRBench experiment adapter snapshot

Snapshot date: 2026-10-03

This branch is a clean source snapshot exported from:

- upstream project: `deep-symbolic-mathematics/llm-srbench`
- local source branch: `codex/restart-experiment-adapter`
- source commit: `4ca5bb7b` (Score missing single-engine support as failure)

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

Nothing here rescores the historical rows. The 36-row expanded-formula
development run keeps its `passed=false` result; see
`docs/expanded_formula_benchmark_repair_20261002.md` and the mainline note
`docs/expanded_formula_failure_repair_20261002.md`.

Canonical scientific-system code and the compact experiment-evidence index
live on the repository's `main` branch.

