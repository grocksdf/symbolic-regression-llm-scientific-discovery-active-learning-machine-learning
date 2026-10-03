# Expanded Formula Discovery: root-cause repair and frozen follow-up protocol

## What the public main branch currently does

The expanded compiler exists and accepts numeric-inner expressions such as
exp(-0.7*x0)+log(1+Abs(x0)). The proposal path was still narrower:

1. The user payload told the provider to use the historical closed basis.
2. PROPOSE_NEW_SKELETON was not in the action enum.
3. Gap batches could return only lineage edits or closed-basis corrections.
4. Expanded amplitude fitting rejected any exact zero outer coefficient.
5. The benchmark configs still dispatched only polynomial_lasso, mcts,
   sparse_library, additive_mechanisms, all of which are closed-basis or
   explicitly forbid exp/log/division.

This explains why a successful parser did not produce expanded candidates in the bank.
The 429 records are a separate provider-infrastructure failure and remain separate
from model abstention or grammar rejection.

## Changes in expanded_formula_discovery_v2_main.patch

- Adds a code-owned PROPOSE_NEW_SKELETON action.
- Gives the expanded protocol a single, versioned payload contract. It permits only
  the bounded registered grammar, freezes inner numeric literals, and requires one
  new-skeleton proposal per Gap batch when the frozen quota is positive.
- Keeps y_hat corrections in the historical closed protocol. Expanded Gap proposals
  are complete x-only equations.
- Allows a sparse outer refit to drop zero amplitude terms while checking that the
  surviving frozen supports are a subset of the compiled supports.
- Registers PySR as an expanded-capable engine and adds the missing unary operators.
- Permits the polynomial baseline to run under the expanded compiler as a matched
  baseline.
- Adds a response-free test file and an expanded provider preflight mode.
- Adds a new benchmark config without changing the old failed config or any result
  directory.

## Engine choice

For the new formula protocol use:

| Role | Engine | Reason |
|---|---|---|
| Expanded discovery | pysr | Existing adapter can search new expression-tree topologies and fit constants; its unary/binary set is expanded by the patch. |
| Matched baseline | polynomial_lasso | Keeps the old degree-bounded baseline under the same candidate compiler. |
| Historical auxiliary evidence | mcts, sparse_library, additive_mechanisms | Keep for the old protocol or an explicitly labelled auxiliary arm; they should not be presented as expanded coverage because their registered skills forbid most of the needed operators. |

The optional dependency is already declared in the main project as
pip install -e .[symbolic]; it also requires a working Julia/PySR installation.
Do not silently fall back to a closed engine if PySR is unavailable. Fail the
preflight before opening benchmark truth.

## Provider preflight

Run the transport check first, with no benchmark data:

    D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe D:\01\666\hypothesis_mvp\scripts\preflight_llm_provider.py --config D:\01\666\hypothesis_mvp\config\bigmodel_glm_5_2.json --expanded --calls 3 --interval 10

All calls must be 2xx, strict JSON, and contain a valid PROPOSE_NEW_SKELETON
candidate. A 401/402/403/429, timeout, or invalid response is a terminal
preflight failure. Do not open benchmark truth after that failure. The 429 ledger
must retain provider status/body excerpt and be reported as transport_failed, never
as model_abstained.

## Response-free Gate before the new development run

Install the optional backend and run only correctness checks:

    D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe -m pip install -e "D:\01\666\hypothesis_mvp[symbolic]"
    D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe -m pytest D:\01\666\hypothesis_mvp\tests\test_expanded_formula_compose_gate.py -q

The fixture must prove, without benchmark responses:

- exp/log/sqrt/Abs/division parse and compile;
- a complete new skeleton survives candidate validation;
- a Gap batch without a new skeleton fails closed;
- PySR is registered and the expanded contract is accepted;
- the sparse outer-refit path does not reject a valid zero amplitude.

## New method identity and experiment

Use the added
configs/aistats_three_arm_formula_expanded_candidate.yaml.
It is a new development protocol. Keep the earlier formula_round2_source_first.yaml
output and passed=false ledger immutable.

Before running the full matrix, run one response-free/dry-run command and verify:

- engines=[pysr, polynomial_lasso];
- one worker and one provider route;
- two discovery cycles;
- pcpi-expanded-fixed-inner-v1;
- expanded_formula_synthesis=true;
- Gap calls request one new skeleton;
- no output directory already exists.

Then run the development matrix once. Keep Full and No-LLM on the same PySR jobs,
budgets, seeds, initial data, candidate cap and failure policy. For a formula recovery
claim, also give No-LLM a matched non-LLM skeleton proposal budget; otherwise a
Full-versus-No-LLM difference confounds LLM structure with candidate count.

Report separately:

- transport failure rate;
- grammar/materialization rejection rate;
- candidate-bank size and number of distinct skeletons;
- exact instantiated recovery;
- parameter-aware structural recovery;
- predictive ID/OOD score;
- fraction of tasks with at least one new-skeleton candidate;
- per-task paired Full minus No-LLM effect.

Do not use this new development run to rewrite the previous all-zero result. It tests
a new representation/engine/proposal protocol after the documented
representation-mismatch NO-GO.

