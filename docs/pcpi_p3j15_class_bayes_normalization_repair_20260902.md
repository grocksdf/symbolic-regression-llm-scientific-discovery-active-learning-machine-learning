# P3J.15 canonical class-Bayes normalization repair

The repaired P3J.15 execution at source commit
`a9e9cddf0f0cbf38080eb5a4ef59abc83c4853f0` completed all 32 CCPP runs, then
terminated on `uci_gas_turbine_co`, seed `2026080701`, query 8 while admitting
an already revealed response. The response receipt was durably published, but
the query ledger and next state were not. The terminal output records
`P3J calibrated posterior does not match its class Bayes update`; held-out
access and validation-based selection remained false.

The former reveal path computed the same class posterior twice: once from the
current class predictive density and once by updating every structure model
and applying accumulated class calibration factors. These expressions are
algebraically identical, but their independent floating-point normalizations
can diverge after long or highly concentrated posterior sequences. A fixed
machine-epsilon comparison therefore rejected a valid update.

The repair does not widen that comparison. It defines one canonical vector of
calibrated structure log weights from the updated base posterior and cumulative
class factors. Structure probabilities, class log joints, calibrated evidence,
and the audit posterior are all normalized from this vector. The independent
check now aggregates the same canonical weights in two ways--summing normalized
structure probabilities and classwise log-sum-exp--so it still detects index,
partition, or normalization mistakes without comparing two numerically
different implementations of the same law.

Correctness coverage includes the original one-step identity, a deterministic
128-update stress chain, complete ambiguity-family admission, reveal ordering,
transactional measured-pool tests, source integrity, and the full historical
regression suite. This is an implementation-correctness repair, not an efficacy
adaptation or a result-conditioned change to the acquisition policy.

The failed output remains immutable and cannot be resumed after the source-tree
change. A subsequent formal execution requires another fresh output identity.
