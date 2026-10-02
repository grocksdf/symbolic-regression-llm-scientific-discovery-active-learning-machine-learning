# LLM-SRBench experiment adapter snapshot

Snapshot date: 2026-10-02

This branch is a clean source snapshot exported from:

- upstream project: `deep-symbolic-mathematics/llm-srbench`
- local source branch: `codex/restart-experiment-adapter`
- source commit: `d821898e` (Overlay the v2 expanded-formula admission and bank projection)

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

Nothing here rescores the historical rows. The 36-row expanded-formula
development run keeps its `passed=false` result; see
`docs/expanded_formula_benchmark_repair_20261002.md` and the mainline note
`docs/expanded_formula_failure_repair_20261002.md`.

Canonical scientific-system code and the compact experiment-evidence index
live on the repository's `main` branch.

