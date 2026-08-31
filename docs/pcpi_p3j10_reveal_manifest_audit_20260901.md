# P3J.10 matching reveal and run-manifest audit (2026-09-01)

P3J.10 admits a response only after loading the published P3J.9 decision and
checking the prior state, candidate identifier, and exact selected coordinates.
The response is then persisted as a no-overwrite receipt. A different response
cannot replace it.

The complete four-model state advances once from the frozen prior state and
receipt. Recovery deterministically reconstructs the same next state rather
than advancing an already advanced state. The query ledger binds prior and next
state hashes, observation/update counts, receipt hash, selected score, and
held-out-closed flags. Any advance failure terminally closes the query.

The run manifest accepts only ordered, unique, contiguous query identities with
state-linked complete ledgers and no terminal query failure. It records zero
failures and held-out closure and is itself published once.

This is correctness-only composition: no real oracle was called. P3J.11 must
replace the P3I acquisition branch in the measured-pool dataset runner with
these P3J transactions, add supervised live progress/terminal verification,
and freeze the exact execution config before real execution can be authorized.
