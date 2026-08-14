# Frozen P2A real-result audit

Decision: **PASS WITH GENEALOGY WARNING**.

- Result: `p2a_real_smc_uci_mainline_runtime_fix_20260807_164908.zip`
- Result SHA-256: `1895593140381d35704fe6214ad0cfe41890e7a6a6320b66bfd824e5096ab09f`
- Recorded source patch SHA-256:
  `bbdab96e03717575b156bd005cdbf1422bb5a84d8c0cfcf0df92e175ee4afc8d`
- Recorded source-tree SHA-256:
  `3d3b924c62ec480b77e63dd981c63757fb6dcb799860ad3defd75c78f1858ebe`
- Completeness: 72/72 runs, 0 failures, 8 seeds, 3 particle counts,
  3 targets in 2 dataset families.
- EvidenceRegistry: 73 events; hash chain valid.
- Diagnostics: 9,216 observation steps and 8,257,536 ancestor indices audited.
- Held-out opened: false.
- Selection used held-out: false.

At 2,048 particles, mean structure TV was 0.008267 for CCPP, 0.000061
for Gas CO, and 0.006743 for Gas NOX. Mean predictive-NLL error was
0.000531, 0.003071, and 0.002178 respectively. All frozen numerical checks
passed.

Final root ancestors were 114–138 for CCPP, 1–2 for Gas CO, and 3–9 for Gas
NOX. Exact-target agreement remained valid because the finite posterior was
highly concentrated and the rejuvenation kernel was invariant. This warning
blocks extrapolation to open-grammar SMC and motivates P2A.1 robust SMC.

Supported claim: the original fixed-universe P2A implementation passed its
frozen exact-posterior numerical checks on the registered real measurements.
Unsupported: trans-dimensional correctness, discovery superiority, class-EIG,
motif safety, held-out confirmation, VED, or a new scientific law.
