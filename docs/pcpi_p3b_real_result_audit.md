# Historical P3B real measured-pool negative-result audit

Audit date: 2026-08-10  
Result ZIP SHA-256: `96c8a82b391b34302237fd538f5503dd45416f35b1279a5ae5f4d5eae903ffca`  
Registered source tree: `4b77141002c358292d89f3aed4cd9b0b1ff4acf6690573d72b31703f5b839bdd`  
Registered production code: `1105f076f3d960127c5a20be03756b371966d7410807c9196e7b9b76a13558da`

## Verdict

The artifact is a valid real-data development experiment. It is not simulated,
the protocol Gate passed, and independent recomputation reproduced the stored
statistics. Its scientific efficacy verdict is nevertheless negative:
`REAL_ADVANTAGE_NOT_DEMONSTRATED`.

It can support claims about protocol execution, provenance, budget matching,
and the identified singleton-class failure. It cannot support a claim that
PCPI outperforms random, uncertainty, or QBC.

## Identity and protocol checks

- all 139 registered source files matched their SHA-256 and size;
- the independently reconstructed source-tree hash matched the manifest;
- 96/96 policy runs, 3,072 acquisitions, and 3,168 curve rows were present;
- 32 initial plus 32 acquired observations were used for every run;
- every run used the same 128-candidate pool and 3,600 candidate-score budget;
- CCPP and all five Gas Turbine official file hashes matched registration;
- Gas Turbine CO and NOX were counted as one dataset family;
- no failed or replacement seeds occurred;
- held-out remained closed and selection did not use held-out;
- EvidenceRegistry contained 96 valid hash-chained events.

## Independent numerical recomputation

Maximum absolute discrepancies from raw per-query and per-round tables were:

- total reported class gain: \(1.33\times 10^{-15}\);
- normalized AULC: \(5.55\times 10^{-16}\);
- final validation RMSE: exactly 0;
- all 15 paired-effect rows: at most \(1.67\times 10^{-16}\) on floating metrics.

The negative result is therefore not a summary or plotting error.

## Family-level effects versus random

Negative nAULC delta favors PCPI; positive class-gain delta favors PCPI.

| Family | Mean nAULC delta | 95% CI | Mean class-gain delta | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| CCPP | -0.00894 | [-0.03428, 0.01641] | -0.14677 | [-0.41198, 0.11843] |
| Gas Turbine | -0.01888 | [-0.10433, 0.06657] | 0.15320 | [0.01467, 0.29173] |

No predictive interval excluded zero. Structural gain was not positive in
every family, and CCPP class-gain negative transfer was 0.625.

## Root-cause finding

The finite bank contained 5 structures. Every dataset, seed, policy, and round
also reported exactly 5 operational classes; the maximum absolute difference
between class entropy and structure entropy was
\(4.44\times 10^{-16}\). The P3B implementation therefore performed structure
EIG under another name rather than aggregating predictive equivalence classes.

The cause was general: exact equality of componentwise-quantized predictions
over 128 actions and 3 quantiles becomes increasingly unlikely as the action
domain grows. No dataset-name branch, answer expression, or held-out feedback
was involved.

## Decision

Preserve this artifact as development negative evidence. Replace the
high-dimensional exact-signature rule with the preregistered,
uncertainty-scaled complete-link operational partition; use an initial-frozen
partition for the comparative entropy endpoint; validate that repaired
definition in P3A.1, then rerun P3B.2 on the same real development/acquisition
roles with held-out still closed. Do not tune the threshold, sample cap, or
assessment rules after viewing the P3B.2 result.
