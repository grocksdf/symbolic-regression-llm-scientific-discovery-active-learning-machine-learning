# LLM-SRBench experiment adapter snapshot

Snapshot date: 2026-10-01

This branch is a clean source snapshot exported from:

- upstream project: `deep-symbolic-mathematics/llm-srbench`
- local source branch: `codex/restart-experiment-adapter`
- source commit: `cf786369ab58f35a6b648debfb002e1f0af491dd`

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

The 2026-10-01 increment adds the MatSci quality-first `48E + L` protocol
(builder, correctness gate, runner and tests, together with its protocol
contract), the in-system LLM-channel ablation arm configuration
`configs/aistats_drr_full_llmchannel_v7.yaml`, the registration of that arm in
the external-baseline reference evaluations, and the alignment of the
matched internal wall-clock budget with the ablation design.

Canonical scientific-system code and the compact experiment-evidence index
live on the repository's `main` branch.

