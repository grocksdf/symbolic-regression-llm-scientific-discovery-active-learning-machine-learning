# LLM-SRBench experiment adapter snapshot

Snapshot date: 2026-10-31

This branch is a clean source snapshot exported from:

- upstream project: `deep-symbolic-mathematics/llm-srbench`
- local source branch: `codex/restart-experiment-adapter`
- source commit: `f731fecf` (Fix search-rich locality filter closure)

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

The 2026-10-06 increment brings the branch up to local source commit
`d018628c` (192 commits, 28 changed files: +3532 / -10, all under `configs/`,
`scripts/` and `tests/`). It adds the matched-attribution protocol family -
the isolated matched S branch, the matched three-arm projection and
composition gate, the gap structural generator and its correctness runner, the
matched attribution freeze builder, exclusion manifest, correctness gate and
matrix runner, the common-report matched audit, the durable interruption
recovery and multi-seed reporting, the registered-family generalization, and
the proposal mechanism diagnostics with their lineage binding - together with
their regression tests. `scripts/audit_iterative_bank_reporting.py` now reads
each reporting role from HDF5 once through `_read_role_rows`, so a covariate
column is no longer materialized twice with different orderings. This adapter
requires mainline `3f659d52` or later for the candidate-validity modules
(`candidate_validity`, `mechanism_funnel`, `proposal_calibration`) and the
legacy fixed-formula token replay.

The 2026-10-06 second increment brings the branch up to local source commit
`8979f769` (1 commit, 12 changed files: +917 / -18, all under `docs/`,
`scripts/` and `tests/`). It repairs the response-free planning of the matched
S comparator: `scripts/effective_structural_generator.py` and
`scripts/effective_structural_contract.py` select parents by the true arity of
each operation (`bounded_power` is single-parent), predict the full AST node
count and source length before an operation is committed so the existing
compile budget is preserved, check compilation, support novelty and numerical
definedness on legitimate covariate points before parameter fitting, and
abstain explicitly when no feasible candidate remains or the planning budget is
exhausted - keeping the gap-activated denominator and performing no top-up,
retry or extra budget after a fit or admission rejection. The S-branch funnel
now accounts for registered slots, gap activation, candidate formation, fitting,
validation, structure preservation and independent admission, plus the point
counts of planning checks, compile calls, numerical checks and submission
checks. The new policy `effective-response-free-v1` is fixed by
`docs/effective_structural_comparator_contract.md` and is opt-in; the previous
policy remains the default, and every gate, freeze and S child must carry a
matching policy and source identity. S action covariates are read from HDF5
with the label column excluded at the bottom layer. This is a correctness and
opportunity-design upgrade only: it does not change the LLM, the independent
admission implementation or its thresholds, reporting scores or the existing
GO criteria, it does not claim an equal number of materialized or evaluable
opportunities for L and S (the registered basis is
`nominal-slots-with-response-free-feasibility`), and it does not claim any
predictive-effect improvement.

The 2026-10-06 third increment brings the branch up to local source commit
`3ba9d12b` (1 commit, 8 changed files: +1203 / -66, all under `docs/`,
`scripts/` and `tests/`). It closes two over-authorization paths in the matched
attribution runner. First, a nominal-slot comparison no longer grants an
equal-effective-opportunity causal attribution:
`equal_effective_opportunity_attribution_authorized` and
`equal_effective_opportunities_established` are fixed `false` and cannot be
turned on by post-hoc candidate counts, the legacy generic field
`matched_predictive_attribution_authorized` is disabled, and downstream readers
must address the explicit claim scope
`nominal_slot_matched_predictive_effect` instead. Second, a positive task mean
alone no longer grants GO: `passed` / `scientific_verdict` now additionally
require the task-paired bootstrap two-sided 95% CI lower bound to be strictly
greater than zero for **both** primary contrasts (`L_minus_S_log_score` and
`L_minus_E_log_score`), with NO-GO and no fallback when a bound is zero,
crosses zero, or the interval is missing, degenerate or non-finite; missing,
duplicated, unregistered or non-finite coordinates are rejected rather than
dropped from the analysis. Seeds are averaged within a task before resampling,
tasks are the resampling unit with equal weights and shared draws across
contrasts, and seeds never act as independent tasks; `S_minus_E` stays an
exploratory contrast. New files are
`scripts/attribution_claim_contract.py`, `scripts/run_attribution_claim_correctness.py`,
`docs/attribution_claim_and_ci_contract.md` and their tests; the freeze schema
advances to `matched-structural-attribution-freeze-v2`, v1 can no longer
authorize an execution, and the builder rejects an incomplete protocol, legacy
mean-only rules, an unexecuted formal dependency gate, or a fixture gate
substituted for the real one before any HDF5 task selection. The S planner, the
LLM code, admission and the reporting log-score are untouched. The statistical
method, resample count and bootstrap seed are deliberately left null and must be
registered in a full protocol before any outcome is read: this increment is a
contract supplement, not an executable experiment.

The `configs/aistats_three_arm_formula_expanded_candidate.yaml` copies are no
longer byte-identical between the two repositories: this branch freezes
`llm_model: glm-5.3-flash` for the iterative matrix (`78c5bc75`), while the
mainline copy keeps `llm_model: glm-5.3`. Every other field matches.

Nothing here rescores the historical rows. The 36-row expanded-formula
development run keeps its `passed=false` result; see
`docs/expanded_formula_benchmark_repair_20261002.md` and the mainline note
`docs/expanded_formula_failure_repair_20261002.md`.

The 2026-10-17 increment brings the three gain-localization files up to local
source commit `0dba03b1`. `methods/hypothesis_mvp_pcpi/drr_searcher.py` is the
Correction-005 complement: `DRRBenchmarkSearcher` now reads the two new
`DiscoveryConfig` transport knobs `llm_attempts` and `llm_retry_backoff_s`
through `getattr` defaults of 3 attempts / 4.0 s, so every earlier registration
keeps observing exactly the hard-coded transport behaviour it was executed
under. `configs/aistats_three_arm_formula_expanded_candidate.yaml` and the new
`configs/aistats_three_arm_formula_no_gap_block.yaml` then set those knobs to
6 attempts and 30.0 s of exponential backoff, identically on both L arms, which
spans roughly 15.5 min of provider outage before a call is declared an
infrastructure failure. The new config also carries Arm B's own legal
non-iterative `three_arm_formula_no_gap_block` condition and the explicit
`decoded_response_fraction: 0.75` control that used to be an implicit side
effect of the condition string. The `glm-5.3-flash` freeze noted above still
holds on this branch. No mainline `hypothesis_mvp` code is touched.

The 2026-10-21 increment brings the branch up to local source commit
`68a40647`. It carries three strands.

*Mechanism ablation (Experiment 2) adapter.* New files
`scripts/mechanism_ablation_contract.py`,
`scripts/build_mechanism_ablation_freeze.py`,
`scripts/run_mechanism_ablation_matrix.py`, `scripts/score_mechanism_arm.py`,
`scripts/ablation_decode_control.py` and
`scripts/ablation_equivalence_gate.py`, with their contract tests, register
the E / `L_no_feedback` / `L_full` ablation together with its freeze builder
and its three gates.

*Feedback-mechanism factorial adapter.* New files
`scripts/feedback_mechanism_factorial_contract.py`,
`scripts/build_feedback_mechanism_factorial_registration.py`,
`scripts/authorize_feedback_mechanism_factorial.py` and
`scripts/run_feedback_mechanism_factorial.py`, plus the four arm configs
`configs/feedback_mechanism_A_no_gap_no_update.yaml`,
`..._B_gap_generation_no_update.yaml`,
`..._C_masked_generation_update.yaml` and `..._D_full.yaml`, register the
96-child factorial. This protocol is published together with the
infrastructure fault it uncovered, described next.

*PySR / Julia runtime repair.* Three defects compounded into an experiment that
degraded silently. JuliaCall 0.9.36 needs `PYTHON_JULIACALL_EXE` and
`PYTHON_JULIACALL_PROJECT` bound to one validated runtime, and binding only
`PROJECT` raised an initialization error; the wrapper rewrote that error into
the generic `the requested pysr backend is unavailable`, discarding the cause;
and the scheduler then treated the engine failure as degradable, so a run
completed with a registered engine that had contributed no candidate. The
runtime is now bound from the frozen response-free Gate
(`configs/pysr_runtime_gate.json`, produced once by
`scripts/check_pysr_runtime.py`) before the output directory is created and
before any HDF5 or provider access, and `run_allocated` fails closed when a
registered engine produces nothing across all of its attempts
(`FORMULA_ENGINE_ALLOW_DEGRADED=1` restores the previous behaviour). The Gate
resolves the Juliaup binary rather than the zero-byte WindowsApps alias, and
freezes both the `julia.exe` and `libjulia.dll` SHA-256.
`methods/hypothesis_mvp_pcpi/drr_searcher.py` additionally replaces the
hard-coded condition-string tests for the gap cycle with the
`posterior_gap_directed` and `iterative_posterior_refinement` config switches,
so Arm B's non-iterative `three_arm_formula_no_gap_block` condition is legal.
No scoring rule, statistical method, resample count or bootstrap seed is
touched, and no historical row is rescored.

The 2026-10-22 increment brings the branch up to local source commit
`d8bceca7`. It is the R-v2 rerun gate: the execution-audit hotfix for the
72-child C/D/R design, in `scripts/feedback_repair_audit.py`,
`scripts/feedback_repair_contract.py`, `scripts/run_feedback_repair.py`,
`scripts/build_feedback_repair_registration.py`,
`scripts/run_feedback_repair_runtime_gate.py`,
`scripts/verify_feedback_repair_child_emitter.py` and
`tests/test_feedback_repair_audit_offline.py`, plus the no-response acceptance
harness `scripts/Invoke-FeedbackRepairNoResponseAcceptance.ps1`.

The hotfix makes four families of guarantees. Child identity: the result JSON
must agree with the artifact on task, seed (a strict JSON integer), family and
condition, the artifact carries `task_name`, and every path must resolve inside
the child's own output directory; conflicting or genuinely out-of-bounds
coordinates are rejected. Generation isolation: a successful payload states all
of `test_or_ood_accessed`, `heldout_opened`, `ground_truth_expression_opened`
and `sealed_response_values_decoded` explicitly as false, and a missing, true
or non-boolean flag fails; contradictory flags already inside the artifact are
rejected too. Provider cost: the evidence is captured from the successful
payload before the identity, artifact and engine checks run, fsynced and read
back, and retained on nonzero exit, timeout and failure artifacts, while any
capture conflict, fsync failure or illegal record ends the child as `failed`
rather than `generated`. Runtime gate: public input symbols are checked against
a fail-closed whitelist, and both synthetic engines go through the production
scheduler's isolated dispatch under `FORMULA_ENGINE_PROCESS_ISOLATION=1`
instead of an in-process `fit()` call.

*How this was applied.* The hotfix is cumulative onto an
`feedback-r-v2-complete-benchmark.patch` state that this branch has never
carried, so four of its seven files had no base here and `git apply` refused.
The delivered final files were copied in instead, and their blobs were verified
equal to the patch's own target index for all seven files, so the result is
byte-identical to a clean application. This is recorded because the inbound
package states the hotfix must not be stacked blindly.

*One known environment limitation.* In this Windows account
`Path.symlink_to()` returns without raising but produces a link that does not
exist, so `test_symlink_to_external_artifact_rejected` raises
`FileNotFoundError` at the later `resolve(strict=True)` before the symlink
guard is reached: 58 passed, 1 failed. The guard itself is correct; the test
needs an account that can create symlinks.

No R-v2 generation, skeleton quota, admission, posterior, scorer, C/D
configuration, v1/v2 protocol or statistical threshold is changed by this
increment, and the mechanistic interpretation boundary is untouched. No scoring
rule, statistical method, resample count or bootstrap seed is touched, and no
historical row is rescored.

Canonical scientific-system code and the compact experiment-evidence index
live on the repository's `main` branch.

