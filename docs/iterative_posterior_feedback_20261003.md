# Candidate admission before the next posterior Gap

Source base: `e7cb71d`. This is an opt-in **exploratory discovery** path in
`DiscoveryAgent.run()`. No historical result or measured executor changes.

## The implemented order

With `DiscoveryAgentConfig(iterative_posterior_refinement=True,
posterior_gap_directed=True, cycles>=2)` and a fresh `(gap_audit,
gap_admission)` pair for **each** outer cycle:

1. Cycle zero freezes one engine frontier. Its bank is fitted on the fixed
   development responses; a conditional PCPI posterior and independent
   regional gap are computed from that bank and `gap_audit[0]`.
2. The actual inner proposal payload records the gap identity. All evaluated
   LLM candidates with new lineages count against the frozen per-cycle
   attempt ceiling. `gap_admission[0]` is opened only after those candidates
   have been materialized. Its region likelihood ratios either admit or
   reject each optional structure.
3. The core plus independently admitted candidates is frozen as a new bank;
   its conditional posterior is recomputed on the **same development fit
   responses**. No rejected LLM structure is passed as a cycle survivor.
4. Cycle one reuses the frozen engine frontier. Its Gap must name exactly the
   bank and posterior target from step 3. It uses fresh `gap_audit[1]`; its
   proposal payload must carry this Gap identity. A distinct
   `gap_admission[1]` handles its candidates. The same rule generalizes to
   more registered cycles.

The role preflight runs before output creation, engine jobs, or provider calls.
It rejects overlap in covariate rows even when the response value differs,
including fit, selection, every audit and every admission role. Each cycle
gets `.05/cycles` for the adequacy screen and `.05/cycles` for conditional
candidate admission; this is a fixed across-cycle allocation. The finite
bank posterior identities, candidate attempts, signed admission records,
and proposal Gap identities are retained in the feedback trace. The
new mode rejects an already populated output directory instead of replaying
provider calls. Duplicate admitted supports fail the cycle before the next
Gap is constructed. The
`iterative-posterior-refinement-gate-v1` checker must see at least one
admitted candidate reflected in a **later** Gap before setting its verified
flag true. If the new bank resolves the gap and the next cycle abstains,
the posterior update still occurred; a second LLM call is not forced.

## Correctness and claim boundary

The controlled test substitutes deterministic engine and LLM proposals but
uses the **real expanded bank constructor, regional admission, posterior
fit, adequacy screen and second-cycle gap**. It proves the execution order
and identity handoff, not predictive efficacy. A separate fixture admits
`exp(x0)` through the expanded formula contract. Reused response roles fail
before an output directory is created.

The result remains a bank-conditional Bayesian posterior. Data-dependent
proposal and admission do not turn it into a joint Bayesian posterior over
the full grammar; that requires an explicit prior/search-selection model.
The original operational-class estimand and measured acquisition runner do
not consume this exploratory bank, and its action authorization remains
false.

`run_exploration_ablations()` explicitly rejects this new mode. It currently
accepts one global gap/admission pair and does regional admission **after**
all cycles; passing this mode through it would reuse roles or double-admit.
Before a Full versus No-LLM efficacy experiment, the paired runner needs
cycle-role forwarding, common bank/target comparisons, identical frozen
engine jobs and budgets, and a separate independent reporting split. The
current mainline benchmark adapter also needs its expanded formula parser
and engine contract replay. Do not launch the new mode through the old
three-arm YAML or treat this correctness check as a successful experiment.

For a response-free source check on the canonical Windows interpreter:

```powershell
& 'D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe' -m pytest -q 'D:\01\666\hypothesis_mvp\tests\test_iterative_posterior_feedback.py'
```
