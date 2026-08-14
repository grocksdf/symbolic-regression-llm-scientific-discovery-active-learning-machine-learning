# P3B.7 returned-result audit

Audit date: 2026-08-12  
Decision: **protocol PASS; efficacy FAIL; stop P4/P5; repair acquisition target**

## Artifact identity and protocol

- Result ZIP SHA-256:
  `d9ad70a6653b7850f168c5650de9682db8dfc7a5cf71033984e15163d4d4a7f1`.
- Returned canonical-source ZIP SHA-256:
  `870b870846effe334cb7b04e6a5b992337fb64cc847eb34105a178478898dc09`.
- Formal source-package SHA-256 recorded by the run:
  `127725741cb7c64137ab7c23513b84bf0d761f8d75d89a60c8330c5a0414f8d7`.
- Source-tree SHA-256:
  `97b1a8ebb713a1d3a1e62edb4430a8c414aff2ddd04d3c0a06865d8dc95be0a8`.
- Production-code SHA-256:
  `b228488bf56945b48755d7570ffe23276a9e3fba64ee508fb577b9f8d898cce0`.
- EvidenceRegistry: 97 valid hash-chained events; head
  `36bec588f555d536b01918c4578bdc938a4ed83603a5bf697a712829b1581526`.

All committed read-only exports reproduce. The returned source tree verifies
against its delivery manifest and its full 131-test suite passes. The real run
contains 96/96 completed policy cells, zero failed or replaced seed, matched
initial/acquisition/validation/pool/candidate-evaluation budgets, and one
shared class partition, SafeBayes calibration, preconditioner, and split per
dataset--seed cell. Gas CO and NOX are one dataset family. Untouched-heldout
remained closed and selection never received held-out data.

## Independently reproduced effects

Negative nAULC delta favors PCPI. Positive frozen-class entropy-gain delta
favors PCPI.

| Family | Baseline | mean nAULC delta | 95% paired CI | mean class-gain delta | 95% paired CI |
| --- | --- | ---: | ---: | ---: | ---: |
| CCPP | random | -0.01731 | [-0.05689, 0.02228] | -0.28629 | [-0.47015, -0.10243] |
| CCPP | uncertainty | 0.01144 | [-0.00422, 0.02710] | -0.00195 | [-0.10598, 0.10209] |
| CCPP | QBC | 0.00540 | [-0.00696, 0.01775] | -0.04046 | [-0.11902, 0.03810] |
| Gas Turbine | random | -0.05105 | [-0.12229, 0.02020] | -0.01482 | [-0.22651, 0.19686] |
| Gas Turbine | uncertainty | -0.01781 | [-0.04137, 0.00575] | -0.11537 | [-0.22832, -0.00242] |
| Gas Turbine | QBC | -0.02617 | [-0.05006, -0.00228] | -0.11192 | [-0.23219, 0.00836] |

P3B.7 therefore does not satisfy the preregistered family-level efficacy
Gate. In particular, CCPP class gain is worse than random in all eight seeds,
and its paired interval excludes zero in the wrong direction. PCPI is
predictively better than random in both family means, but it is not better
than uncertainty and QBC in every family.

## Root-cause diagnosis

P3B.7 successfully removes the previous inactive-utility defect: class-EIG is
used for 100% of PCPI queries in both registered families and every selected
ranking is numerically certified. The remaining failure is therefore not a
class-radius or quadrature-budget problem.

The nominal expected and realized class information disagree systematically:

| Dataset | mean summed nominal class-EIG | mean realized frozen-class entropy gain | seeds with negative total gain |
| --- | ---: | ---: | ---: |
| CCPP | 0.51389 | -0.27551 | 7/8 |
| Gas CO | 1.57679 | 0.08780 | 4/8 |
| Gas NOX | 2.05892 | -0.12728 | 6/8 |

This is diagnostic of acquisition under model misspecification: PCPI selects
points at which the finite symbolic bank disagrees, but those points need not
be representative of the registered prediction domain. Surprising responses
then diffuse the frozen class posterior rather than concentrate it. Preventing
such falsification by choosing a response-direction rule would be confirmation
bias and is prohibited.

The compute premium is also unresolved. Mean PCPI policy time is 16.22 seconds
for CCPP and about 25.25 seconds for each Gas target, versus 0.31--0.65 seconds
for the baselines.

## Literature-grounded P3B.8 repair

Prediction-oriented Bayesian active learning proposes expected predictive
information gain (EPIG) because parameter-oriented information gain can select
points irrelevant to the target input distribution (Bickford Smith et al.,
AISTATS 2023). Recent BOED analysis under misspecification likewise identifies
representativeness and misspecification amplification as missing from ordinary
EIG (Forster, Ivanova, and Rainforth, 2025). Robust-EIG ambiguity-set methods
confirm the sensitivity of experiment ranking to the nominal belief (Go and
Isaac, UAI 2022).

P3B.8 keeps the posterior, structure bank, operational classes, budgets,
baselines, seeds, and held-out protocol unchanged. It replaces the acquisition
target by the joint random variable `(C, Y*)`, where `C` is the same
initial-frozen class and `Y*` is the response at a uniformly drawn point of the
registered action domain. The information chain rule gives

`I(C, Y*; Y_a) = I(C; Y_a) + I(Y*; Y_a | C)`.

The first term remains the validated class-EIG estimator. The second is a
class-conditional Gaussian-moment EPIG calculation using only visible action
covariates and the current posterior. Because finite within-class predictives
are Student-t mixtures, P3B.8 is explicitly called a joint-EIG surrogate. It
is not relabeled as exact EIG.

No dataset name, target name, answer expression, response direction, held-out
quantity, result-conditioned threshold, or regular-expression template enters
the repair. A maximin/Sibson robust-EIG replacement is not claimed here because
that would also require adopting its corresponding tilted belief update and a
preregistered ambiguity radius; changing only the score would be incoherent.

## Claim boundary

P3B.7 supports a valid negative real-development result and a fully matched,
leakage-closed protocol. It does not support acquisition superiority,
structural-discrimination superiority, held-out confirmation, open-grammar
discovery, physical intervention, motif safety, VED discovery, or a new law.
P3B.8 must pass its controlled correctness Gate before another heldout-closed
real run. P4, P5, motif, VED, and all untouched-heldout confirmation remain
blocked.
