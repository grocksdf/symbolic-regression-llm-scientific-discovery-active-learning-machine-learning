# P3J.12 durable measured-run audit

P3J.12 closes the process-restart gap between the P3J.11 measured-pool adapter
and a formal multi-query run. Each query is bound to the source tree, config,
dataset, seed, ordered visible candidates, predictive target domain, opened
representative history, and prior operational state. A durable reveal receipt
is recovered before any new oracle access. Its identity, selected candidate,
coordinates, and target are revalidated through the same exactly-once state
advance used on first execution.

The coordinator removes a candidate and extends the representative history only
after that matching reveal has been admitted. It publishes the run manifest only
after a contiguous query lineage is complete. It contains no validation,
held-out, future-response, candidate-target, or RNG input.

This is a correctness gate, not an efficacy experiment. It reads no registered
dataset, invokes no real oracle, and does not authorize the formal dataset runner.
Integration with matched baselines, reporting, and the supervised outer process
remains a separate fail-closed stage.
