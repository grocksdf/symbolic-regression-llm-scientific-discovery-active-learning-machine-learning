# PCPI P3J.3 refinement and computational-cost audit

## Outcome

P3J.3 passed its response-free refinement-equivalence and static-cost Gate.
It accessed no real, candidate-response, validation-response, held-out or
simulated efficacy data. It changes neither the P3J joint law, robust lower
envelope, representative constraint, precision schedule nor selection rule.

The implementation now reuses the preceding fine quadrature result as the
next doubled-precision look's exact coarse result. The reuse is admitted only
when quadrature order doubles and the residual-state hash, frozen partition
hash, class count, action count and safety factor all match. A direct
correctness fixture verifies bit-identical scores and error envelopes against
a fresh fine/coarse computation. Crossed state or partition identities fail
closed.

## Static worst-case ledger

The ledger uses only frozen architectural maxima: 128 first-round candidates,
four likelihood-power models, at most seven frozen classes, at most four
Pólya-tree leaves through the registered 32-query horizon, and node orders
`[32, 64, 128, 256, 512]`. It does not inspect a dataset, response, measured
runtime or result.

Without reuse, all looks require 1488 nodes per leaf; exact reuse reduces this
to 1008. The first-query upper ledger is therefore:

- naive source-class quadrature nodes: `21,331,968`;
- reused source-class quadrature nodes: `14,450,688`;
- saved source-class nodes: `6,881,280` (`32.2581%`);
- reused cross-class density evaluations: `101,154,816`.

The last count exposes the remaining quadratic class interaction: every
source-class quadrature response must evaluate every target-class density to
form the response mixture. The refinement repair removes redundant precision
work but does not remove this mathematical interaction.

## Decision

P3J.3 is `passed-refinement-cost-audit-runner-composition-blocked`. Directly
composing the current maximum schedule into the real runner would recreate the
long, late-failing execution risk observed in earlier phases. No threshold,
budget or approximation is relaxed to hide that result.

The next admissible work is response-free computational redesign: shared
batched class-density evaluation, bounded work chunks, fsync-backed progress
and resumable deterministic scoring from committed state identities. Any
approximation that changes score intervals requires a new correctness proof.
Only after a static cost Gate and supervised-runner source Gate pass may a
separate real-development configuration be considered.
