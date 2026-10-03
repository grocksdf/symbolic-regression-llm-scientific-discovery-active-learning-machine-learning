# Expanded iterative formula integration: source contract

Base commits: main `26ad4e6`, benchmark `83c6f9f`. This is a new opt-in
exploratory method; the earlier 36-row `passed=false` result is unchanged.

## Execution and roles

`three_arm_formula_expanded_candidate` now requires the expanded refit policy,
positive new-skeleton quota, sufficient per-cycle evaluation reserve and
`iterative_posterior_refinement=True`. For three islands, two inner rounds and
one skeleton per batch, the reserve is six. The same value is present in both
repository configs. A zero-quota expanded protocol does not request the
forbidden new-skeleton action.

The benchmark adapter retains the historical nine-role partition and splits
only its existing `gap_audit` and `gap_admission` train rows into two
deterministic subroles each. It opens admission responses for the new condition
so that each candidate is independently admitted **before** the next bank
is frozen; inference-initial, reporting, action, test/OOD and confirmation
responses stay closed during generation. Role identities and subrole indices
are recorded. The source preflight fails on overlapping covariates or small
role slices before the engine or provider runs.

Each cycle's trace stores core and admitted bank rows, frozen model and target
identities, candidate admission, proposal Gap identity and its conditional
posterior MAP expression. The benchmark export uses that last admitted-bank
MAP expression. It never exports an unadmitted LLM winner from the inner
discovery leaderboard. Absence of any admission remains a valid negative
outcome with `iterative_posterior_refinement_verified=False`.

After generation, `paired_iterative_bank_reporting` can score every cycle's
Full bank and the *same frozen engine-only bank* against the same independent
train-reporting responses. The benchmark script
`scripts/audit_iterative_bank_reporting.py` opens only registered training
rows after the artifact is complete. No-admission tasks are still included.
This is paired **bank predictive transfer**, not a second matched-search
No-LLM run or a decision/acquisition result. `run_exploration_ablations()`
continues to reject iterative mode because its old closed parser and role
contract cannot safely handle the expanded bank. Do not use it as a shortcut.

## Response-free checks before considering a new development registration

- Check both config files have the same SHA-256 and the benchmark mainline
  resolves exactly to the pinned main commit.
- Run `test_expanded_refinement_contract.py`,
  `test_iterative_posterior_feedback.py` and the benchmark
  `test_iterative_formula_roles.py` in the registered interpreter. Isolated
  sandbox dependencies passed 49 mainline and 15 benchmark synthetic and
  adjacent regression checks; the user's registered Windows interpreter and
  PySR/Julia provider stack have not been executed here.
- Run the provider transport preflight with no benchmark data. Its success
  says nothing about PySR availability or the scientific efficacy of a
  candidate. Investigate 429 responses separately; do not count them as
  model abstentions.
- Verify the actual first-cycle admission and second-cycle posterior Gap
  identities in an instrumented, response-free controlled fixture. An
  abstention must remain a recorded result, not disappear from the matrix.

The new standalone reporting contrast does not establish registered common
operational-class decision loss, a matched independent No-LLM discovery run,
continuous acquisition ranking certification or measured-executor approval.
Those require separate correctness and experimental protocols before claims
about Full > No-LLM decision performance.
