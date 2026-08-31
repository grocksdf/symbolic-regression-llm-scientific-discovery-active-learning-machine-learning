# P3J.11 measured-pool adapter audit (2026-09-01)

P3J.11 creates the single measured-pool boundary that the formal dataset loop
must call. It first completes or resumes the identity-bound P3J decision
transaction. Only then does it request exactly the selected candidate identifier
from the indexed oracle.

The returned identifier is checked against the decision. The raw measured
covariates and target are transformed with the already frozen development
standardizer, and the transformed coordinates must equal the selected action
exactly. Only that matching reveal enters the P3J.10 receipt and exactly-once
advance transaction.

Correctness tests use a minimal indexed-oracle fixture and do not access a real
dataset or run an efficacy experiment. Decision failure causes zero oracle
calls; crossed identifiers or coordinates never reach state advance.

The full dataset loop remains blocked. P3J.12 must replace its P3I PCPI branch
with this one adapter, initialize the class-conditional state from frozen
initial roles and target partition, preserve baseline paths, finalize each
32-query manifest, and add the supervised execution freeze.
