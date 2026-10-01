# MatSci quality-first predictive-increment protocol (2026-09-30)

Step one of the quality-first program. The previously delivered patch only
*constructed* "frozen No-LLM 48 evaluations + extra LLM proposals"; nothing
was evaluated. This protocol wires the evaluation layer and freezes it.

Authorized by this document: code, static checks, protocol freeze, and a
**development diagnostic** on previously used MatSci tasks. Not authorized:
any formal Full > No-LLM efficacy run on fresh tasks, any acquisition-benefit
claim, any test/OOD or held-out access.

## What step one answers

Does an added LLM proposal change *predictive* quality on rows that never
participated in discovery, admission, or selection?

Three arms are built from **identical matched-engine evidence**:

| arm | bank |
|---|---|
| `48E` | the frozen No-LLM production pool (48 evaluations) |
| `48E+L_LLM` | the same pool **verbatim** plus compiled typed LLM proposals |
| `48E+L_control` | the same pool **verbatim** plus the same *number* of non-LLM engine proposals |

`48` is an evaluation count, not a count of distinct engine structures.

## Arms, budget, and the report set

- Baseline: 48 evaluations, frozen from the No-LLM condition.
- `L` is **metered separately** and never comes out of the 48. Compile
  failures and abstentions are counted, not retried and not hidden.
- Data split follows the DRR adapter (sha256-stable order, 50/70/90 cuts).
  The report target is the `inference_initial` segment: rows `[:32]` are the
  initial data and rows `[32:96]` are the report target. It is disjoint from
  discovery rows, the initial data, and the action covariates, and it is used
  **only for scoring** — never for fitting, admission, or selection.

## Hard checks (fail-closed)

1. Both paired conditions complete the registered 48 baseline evaluations.
2. Engine evidence is identical between `full_scientist_v6` and `no_llm_v6`
   (any drift is a failure, recorded, never patched).
3. Development and validation fingerprints match across the pair.
4. Augmented arms keep the frozen baseline rows **verbatim** — an added
   proposal cannot displace a baseline candidate.
5. Compiled-proposal lineage identity and support identity are recomputed and
   compared; proposals with unregistered engine parents are rejected.
6. Compile failures, non-adaptable proposals, provider abstentions and
   rejected candidates are counted into the record.
7. Report predictive scores must be finite and shape-correct.

### Recorded budget asymmetry

In the frozen v6 pair the Full condition spends part of its evaluation
budget on synthesis (36 of 48 evaluations observed on the 20260929 external
pair) while No-LLM completes the registered 48. Engine evidence is identical
between the two conditions, so the shared 48-evaluation baseline is sound,
but the asymmetry is **reported, not absorbed**: every pair record carries
`full_completed_registered_baseline_evaluations` and the aggregate carries
`full_budget_shortfall_pair_count`. The control arm matches the LLM arm in
evaluation opportunity, which is the comparison this protocol pre-registers.

## Primary metric and decision rule

Primary metric: **task-mean paired report log-score difference,
`(48E+L_LLM) − 48E`**, on the identical report target for all three arms.

Pre-registered decisions (all must hold):

- zero pair failures;
- every pre-registered task×seed pair accounted for;
- global LLM log-score gain strictly positive (tolerance `1e-12`);
- at least **2 of 4 tasks** strictly positive;
- LLM gain strictly exceeds the equal-size non-LLM control gain.

The control arm is what separates "LLM proposals help" from "any extra
candidate helps".

## Selected protocol (legacy development diagnostic)

- Tasks: `MatSci6, MatSci8, MatSci16, MatSci25` — the four tasks of the frozen
  20260929 acquisition confirmation, re-used deliberately so the result can be
  read against those old records.
- Seeds: `81, 82` — repeat measurements, **not** new tasks.
- Discovery runs: 16 (4 tasks × 2 seeds × 2 conditions); pairs: 8.
- `independent_tasks: false`. This run cannot replace a fresh-task
  confirmation and cannot support an acquisition, test/OOD, or universal
  claim. `MatSci16/25` negative results are preserved as negative.

A fresh-task formal protocol is available from the same builder with
`--mode fresh` (four SHA-256-minimum MatSci tasks unused by any registered
prior freeze).

## Fresh-task formal protocol (`--mode fresh`)

Task identity is never chosen by result. The builder hashes
`sha256("aistats-matsci-quality-first-increment-v1:" + task)` and takes the
four smallest digests among the tasks that were **not excluded**. Anybody can
recompute the same four.

The exclusion set is exactly what is passed with `--exclude-freeze`; the
builder does not scan the registration tree on its own. The legacy freeze
passed 7 prior freezes (12 tasks). Reusing that same list under `--mode fresh`
selects `MatSci11, MatSci16, MatSci8, MatSci18` — `MatSci8` and `MatSci16` are
legacy tasks, so such a run would be stamped `independent_tasks: true` while
actually re-using two of the four. **The formal run must therefore pass every
prior freeze that names a MatSci task.**

As of 2026-09-30 the MatSci tasks named by any registered freeze are
`MatSci0, 1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 20, 23, 24, 25, 28` (17 tasks).
Excluding all of them leaves 8 unused tasks and the frozen pick is
`MatSci11, MatSci18, MatSci19, MatSci9`.

## Files

- `scripts/run_aistats_matsci_quality_first_increment.py` — runner
- `scripts/run_aistats_matsci_quality_first_correctness_gate.py` — response-free gate
- `scripts/build_aistats_matsci_quality_first_increment_freeze.py` — freeze builder
- `tests/test_aistats_matsci_quality_first_increment.py` — static fixtures

## Gate status (2026-09-30)

Correctness gate `passed = true`, 18/18 checks, on synthetic fixtures only:
no benchmark arrays, no provider call, no response, no held-out access.
Preflight `passed = true`, `execution_started = false`, 16 planned discovery
runs / 8 pairs.
