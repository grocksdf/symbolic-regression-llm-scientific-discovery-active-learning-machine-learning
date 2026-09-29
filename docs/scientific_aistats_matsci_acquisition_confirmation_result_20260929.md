# MatSci acquisition confirmation result

Date recorded: 2026-09-29

Status: **FAILED registered confirmation Gate**

Paper synchronization status: **DEFERRED**

## Frozen identity

- Result:
  `D:\01\666\outputs\scientific_aistats_matsci_acquisition_confirmation_20260929\AISTATS_MATSCI_ACQUISITION_CONFIRMATION.json`
- Result SHA-256:
  `a1ac83a28ef8d6b83d8366cbde2651db9a4f8c980d2e6c9e8c46b7b048473d4a`
- Freeze:
  `D:\01\666\registrations\scientific_aistats_matsci_acquisition_confirmation_20260929\AISTATS_MATSCI_ACQUISITION_CONFIRMATION_FREEZE.json`
- Freeze SHA-256:
  `2384eb5940917a6dfdab559c098505950521732fcfcd231e8799ba571ce83719`
- Benchmark source:
  `701b679c93d2ce0649476daceb43f9c04b00a5bc`
- Mainline source:
  `b82c592d1c8c8098a795b70601cd8f16177c1745`

## Registered protocol

- Fresh tasks: `MatSci6`, `MatSci8`, `MatSci16`, `MatSci25`
- Seeds: `81`, `82`
- Condition: `no_llm_v6`
- Policies: decision-risk targeted versus matched random
- Discovery runs: 8
- Realized trajectories: 16
- Measurement budget: 2 responses per trajectory
- LLM calls: forbidden

The four tasks were selected prospectively by the frozen SHA-256 rule after
excluding every MatSci task used by the registered prior experiments.

## Result

- All 16 trajectories accounted for: `true`
- Execution failures: `0`
- LLM called: `false`
- Test or OOD accessed: `false`
- Held-out opened: `false`
- Mean targeted-minus-random effect: `+0.05960731340110615`

Task effects:

- `MatSci6`: `+0.2262198523531193`
- `MatSci8`: `+0.08388450072675376`
- `MatSci16`: `-0.05250131483440515`
- `MatSci25`: `-0.01917378464104333`

## Gate decision

Passed:

- all trajectories accounted for;
- zero failures;
- global mean strictly positive;
- no LLM calls.

Failed:

- all tasks nonnegative;
- at least three tasks strictly positive.

The terminal result is therefore `passed=false` and
`efficacy_demonstrated=false`.

## Scientific interpretation

The positive global mean shows that targeted acquisition can deliver large
benefits on some fresh MatSci tasks.  However, two of four prospectively
selected tasks exhibit negative transfer.  The earlier development
family-authorization result therefore does not replicate as a robust
task-level confirmation.

The admissible claim is:

> On four fresh MatSci tasks, decision-risk acquisition has positive average
> realized effect but heterogeneous task effects, including two negative
> transfers; the preregistered robustness confirmation fails.

The result does not support:

- robust MatSci-wide acquisition superiority;
- universal targeted-acquisition superiority;
- opening an untouched held-out confirmation;
- rerunning with replacement tasks, seeds, thresholds, or a relaxed Gate.

## Consequence for the experiment program

This confirmation is immutable terminal negative evidence.  No further
realized acquisition experiment should be used to search for a positive
result.  Remaining paper experiments should focus on claims that have
replicated without negative transfer:

1. lineage-bound LLM synthesis under matched engine execution;
2. conservative BOQD response-free improvement;
3. external symbolic-regression baselines and resource accounting.

The paper should be updated only after those remaining experiments are
complete.

