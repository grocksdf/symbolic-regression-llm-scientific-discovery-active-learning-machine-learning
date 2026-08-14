# P2A decision — target-correct fixed-universe SMC

Status: **P2A.1 EXACT-REFERENCE GATE PASSED; REAL CALIBRATION PENDING**

## Implemented candidate

P2A now contains prior initialization, normalized incremental Gaussian
likelihood weights, log-sum-exp normalization, ESS-triggered systematic
resampling, parent/root genealogy, a collapsed finite-bank MH structure move,
exact conditional coefficient/noise draws, posterior predictive evaluation,
and complete per-step diagnostics.

The real runner compares SMC with the exact finite-bank posterior on official,
hash-verified CCPP and Gas Turbine CO/NOX measurements. CO and NOX share one
grouped year split and count as one dataset family. The symbolic bank depends
only on input dimension.

## Gate state

The frozen real result completed 72/72 runs with no failures on official CCPP
and grouped Gas Turbine CO/NOX measurements. Exact-agreement thresholds passed,
the EvidenceRegistry chain was valid, and held-out remained closed. The result
identity and numerical audit are recorded in
`docs/pcpi_p2a_real_result_audit.md`.

Gas CO/NOX nevertheless showed severe root-ancestor coalescence in the original
implementation. P2A.1 removed forced nonterminal resampling, added complete
parent/child/root maps, and passed its 24-run exact-reference stress Gate. The
heldout-closed real calibration remains pending and the result still cannot be
extrapolated to an open grammar.

## Allowed wording

“On an exactly enumerable inference-correctness fixture, ESS-adaptive
fixed-universe P2A.1 passed frozen posterior, genealogy, rejuvenation, and
particle-convergence checks.”

## Not supported

- open-grammar or trans-dimensional correctness;
- accurate marginal-likelihood estimation;
- class EIG;
- scientific-discovery superiority;
- motif safety, held-out confirmation, or VED discovery.

P2B is unblocked for a fresh corrected-kernel diagnostic under the new source
identity. P3/P5 scale experiments remain blocked until the subsequent method
Gates pass.
