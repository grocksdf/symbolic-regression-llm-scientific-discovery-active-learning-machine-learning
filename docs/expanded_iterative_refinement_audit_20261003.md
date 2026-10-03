# Expanded formula search: posterior feedback and breadth gate

Source inspected: public `main` at `2c4e6c0065fba1d0facc730ee2804183c959df94`.
This is a response-free source audit and correctness repair, not a development
or confirmation result. The earlier 36-row `passed=false` is unchanged.

## Actual scheduling

`DiscoveryAgent.run()` calls `_orchestrate_cycle()` before `_discover()` for
each outer cycle. `_independent_gap_brief()` builds its provisional model from
the *current engine results* and generic deterministic references, and fits
that bank to the original development responses. `previous` survivors are
passed only to `_discover()` as seeds. `_screen_gap_candidates()` performs
independent regional candidate admission after the complete variant run.
The inner `ScientificDiscoveryRuntime._llm_search()` repeats proposals against
development-time equation states; it does not construct a PCPI posterior or
refresh the independent posterior Gap between its rounds.

Consequently neither `cycles=2` nor `discovery_rounds=2` establishes
`Gap0 -> LLM1 -> Admission1 -> Posterior1 -> Gap1 -> LLM2`. The current
source trace is `engine-bank-gap_t -> proposal rounds_t`, with candidate
admission after all outer cycles. The agent now marks this boundary explicitly
in the gap audit and result. `verify_iterative_refinement()` is a fail-closed
reference checker for a future production state trace; the present runner
does not emit that trace and cannot pass this gate.

## What the code repair does now

The expanded proposal path accepts the explicit `PROPOSE_NEW_SKELETON` action.
`new_skeleton_quota` is a **maximum per island batch**. A malformed,
duplicate, old-bank, or incumbent skeleton does not count as valid breadth.
Fewer than K valid proposals are reported, without forcing unsupported
equations to fill the quota. A quota larger
than `candidates_per_island` fails before provider transport. Both agent and
inner discovery configuration carry the quota. The agent requires enough
registered LLM evaluation slots for `K * islands * discovery_rounds` in each
outer cycle. This necessary capacity check does not prove every candidate
is actually scored; that needs the proposal/materialization ledger.

The supplied previous patch is a descriptive patch with bare `@@` sections;
`git apply --check` rejects it at line 5. The diff generated from this branch
is a normal unified patch against the source commit above. It addresses the
new-skeleton route and quota; it does **not** silently integrate other edits
from that patch (PySR registration, zero-amplitude refit, provider preflight,
or benchmark adapter). Those need a separate source replay before v2 runs.

## Required production loop before claiming Bayesian iterative refinement

1. Pre-register a common action domain and target weights, the outer-cycle
   count, region map, prior, likelihood, grammar, `K`, global candidate cap,
   and a fresh audit/admission response role for each cycle. Role rows must
   be pairwise disjoint across adaptive cycles and from fit/selection/report.
2. At cycle `t`, freeze the **current admitted bank**, then fit its declared
   conditional posterior on the fixed development observations. Record actual
   bank and posterior identities. Compute `Gap_t` from that posterior and
   `D_gap,t`; bind the proposal payload to the gap identity.
3. Materialize and count all attempted hypotheses before opening
   `D_admit,t`. Screen each against the current core using the independent
   admission role. Correct the evidence threshold for all regions, proposals
   and cycles using a frozen total error allocation, rather than granting
   each adaptive cycle a fresh nominal `.05` budget.
4. Freeze the next bank with independently admitted candidates and recompute
   its posterior using the declared model. Only then compute `Gap_(t+1)` from
   that bank and its fresh `D_gap,t+1`. Fail if the next gap uses the prior
   bank/posterior, a stale payload, or a reused response row.
5. Run a response-free perturbation fixture: admitting a structurally distinct
   candidate changes the next bank and posterior identity, and the next gap
   is recomputed from them. Also check an abstention case, a negative
   admission, and a case where two different posteriors yield the same rounded
   brief. Equality of printed gap text alone is not a failure; provenance of
   the full state is the test.

This would be **iterative posterior-guided discovery, conditional on each
selected bank**. Expanding the prior support after data-dependent search is
not automatically one coherent Bayesian posterior over an open grammar;
that stronger claim requires a joint proposal/selection model and posterior
target. It also does not authorize the measured PCPI executor.

## Breadth sanity before scientific efficacy

Use one registered novelty island and one inner proposal round for a clean
`K=1,2,4` breadth diagnostic, or explicitly budget all islands and rounds.
Hold PySR engine jobs, data roles, initial bank, provider identity, seed list,
and per-candidate evaluation policy fixed. Freeze the three K settings and
the accounting rules before using development truth. Report per independent
task: requests/429, parse success, valid distinct topologies, attempted and
admitted candidates, bank coverage before/after admission, structural
recovery applicability, and compute/time. For a controlled breadth comparison,
either use nested prefixes of a pre-generated candidate pool or disclose
that separate generation calls produce nonnested stochastic searches.
Do not use the opened development tasks to pick a favorable K and call it
confirmation. Give No-LLM a candidate-count-matched structural search budget
when later testing the incremental effect of LLM proposals.

E2E Transformer is not a prerequisite. `PySR` versus `PySR + Gap-LLM` is the
more interpretable first algorithm comparison, once the production loop and
transport checks are real. The previous expanded configuration with three
islands, two inner rounds, `K=1`, and only four LLM evaluation slots is
insufficient even for one outer cycle: its nominal minimum is six slots.
At `K=4` the same layout needs 24 slots per outer cycle, before any other
LLM candidate evaluations.
