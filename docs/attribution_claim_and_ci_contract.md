# Prospective attribution claim and task-bootstrap contract

This contract fixes runner/claim authority and statistical decision logic for
**fresh** experiments. The S planner, LLM, admission and reporting log-score
implementation are unchanged. Historical artifacts and NO-GO verdicts are not
rewritten or rerun.

## Claim scopes

| Result field | Authorization |
|---|---|
| `nominal_slot_matched_predictive_effect_authorized` | May become true only under the new registered statistical gate. |
| `equal_effective_opportunity_attribution_authorized` | Always false: no prospective equal-effective matching design is implemented. |
| `equal_effective_opportunities_established` | Remains false; equal realized counts alone cannot establish a matching design. |
| `matched_predictive_attribution_authorized` | Deprecated and disabled; always false. Consumers must use an explicit scope. |

The only runnable scope is `nominal_slot_matched_predictive_effect`. A protocol
claiming equal-effective opportunity or LLM reasoning superiority is rejected
before selection/execution. This closes the overbroad authorization path; it
does not create a causal matching design or establish candidate-quality gains.

`passed` and `scientific_verdict` refer to the explicit nominal predictive-effect
scope, never to the stronger causal claim. They require the complete registered
matrix, both positive task means, both strictly positive CI lower bounds, and
at least one verified L feedback pair. With a positive mean and a CI crossing
zero, `passed=false` and `scientific_verdict=NO-GO`.

## Statistical registration

A complete new protocol must explicitly supply `statistical_analysis`,
`primary_pass_conditions`, and `claim_scope`. The helper
`statistical_contract(method=..., resamples=..., seed=...)` supplies fixed
structural fields but has **no defaults** for these three choices. Percentile,
basic, or BCa must be chosen before outcomes; no automatic method change or
extra bootstrap looks are allowed. Historical CI method, draw count or seed
cannot be recovered from the previously quoted interval and are not invented.

The task estimator averages each paired coordinate contrast over every
registered seed within each task, then equally averages tasks. Resampling draws
complete task indices with replacement, using the same draw for all contrasts.
It does not treat seed rows as independent tasks, pool reporting observations,
drop failed coordinates, or reweight tasks by successful seeds.

Both `L_minus_S_log_score` and `L_minus_E_log_score` receive the registered
two-sided 95% task-bootstrap interval. Their means and lower bounds must all be
strictly greater than zero. A zero endpoint fails, without epsilon. The
`S_minus_E_log_score` interval is exploratory and cannot change the primary gate.
An undefined/nonfinite primary interval is NO-GO; BCa degeneration does not
trigger a percentile fallback. Missing, duplicate, unknown or nonfinite
coordinates fail before any bootstrap or claim authorization. At least two
registered tasks are required; more seed rows cannot replace that requirement.

PCG64, equal task weights, the two-sided confidence level, quantile convention,
the primary contrasts, method, seed and resample count are all serialized into
the frozen contract. NumPy and SciPy versions are now required dependency
identities. Result artifacts retain the interval, ordered task identities,
condition checks, work count, and bootstrap-distribution hash. These are
approximate bootstrap intervals, not exact finite-sample coverage guarantees.

## Freeze and execution boundaries

New freezes use `matched-structural-attribution-freeze-v2`. The freeze builder
requires a complete fresh protocol preserving historical NO-GO and a passed
formal dependency end-to-end Gate that includes the claim/CI tests. Incomplete
supplements, old mean-only GO conditions, the standalone fixture Gate, and a
false formal-Gate execution marker are rejected before HDF5 selection.

The frozen analysis fields and coordinate design are compared with the original
registered protocol path and SHA256 during preflight. Changing the resampling
method/seed or GO rules only in the freeze cannot silently alter inference.
Source hashes, interpreter, provider, Julia, dependencies, generator policy,
role isolation and full failure accounting retain their existing checks.

The v1 schema is blocked for new execution. Its mean-only screen can still be
viewed as `legacy_mean_only_screen_passed`; current code cannot use that screen
to authorize any claim and does not compute a new post-hoc CI for it. Existing
historical files remain intact, including a previously saved `passed=true`.

`run_attribution_claim_correctness.py` runs only controlled source-bound
fixtures. It deliberately reports `execution_authorized=false` and
`formal_dependency_end_to_end_gate_executed=false`, and cannot substitute for
`check_matched_attribution_correctness.py` on the registered host.

No formal experiment is authorized by this patch or its fixture results.
Actual readiness requires the registered host's dependency Gate and a fresh
complete protocol with predeclared statistical settings and nominal claim scope.
The stronger equal-effective question still requires its own matching design.

The implementation follows the documented paired bootstrap/resampling and
registered interval methods: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
