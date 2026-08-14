# P3B.2 evidence-first pre-run audit

Audit date: 2026-08-11

## Entry condition

The source-paired P3A.2 Gauss--Jacobi class-EIG diagnostic passed all 40
scenario-budget cells, exact-reference ranking, error-envelope coverage,
zero-false-certification, and convergence Gates. P3B.2 is therefore authorized
on the registered CCPP and Gas Turbine measured pools with untouched-heldout
closed.

## Frozen statistical design

- datasets: CCPP, Gas Turbine CO, and Gas Turbine NOX;
- dataset families: CCPP and one grouped Gas Turbine family;
- policies: random, uncertainty, QBC, and PCPI class-EIG;
- eight registered paired seeds;
- 32 initial and 32 acquired measurements per policy run;
- one 128-row candidate subset and one 256-row validation subset per seed;
- identical finite-bank posterior, initial rows, candidate rows, validation
  rows, outer candidate-score budget, and failure policy across methods;
- initial-frozen operational classes for comparative structural entropy;
- dynamic operational classes only for acquisition;
- mandatory official file hashes and closed held-out.

CO and NOX are averaged within seed before family-level intervals. They are not
counted as independent dataset families.

## Evidence architecture repair

The earlier P3B runner wrote tables before appending abbreviated registry
records. P3B.2 now writes each policy summary, full learning curve, every query
decision, and every failure to the single hash-chained EvidenceRegistry. One
aggregate event stores dataset commitments, paired effects, assessment, and
the protocol decision. Tables and JSON results are reconstructed from verified
registry events, and `diagnostics/evidence_export_manifest.json` commits their
hashes.

Every seed records SHA-256 commitments for its initial, validation, and
candidate subsets. The protocol Gate also checks the 3,600 outer
candidate-score evaluations per policy, curve/query completeness, shared
subsets, shared initial class partition, official hashes, and held-out state.

## Claim boundary

Protocol completion alone is not a superiority result. Strong structural
evidence requires positive paired frozen-class entropy gain over random in
both dataset families, controlled negative transfer, non-degenerate classes,
and the frozen ranking-certificate rate. Strong joint evidence additionally
requires nonpositive mean predictive nAULC deltas against every baseline in
each family. Weak or negative results must be preserved without changing the
threshold, seeds, budgets, sample cap, or assessment rules.

P3B.2 is active measured-pool selection, not physical intervention, open-
grammar discovery, held-out confirmation, motif evidence, VED discovery, or a
new-law claim.
