# P3B.4 returned-result audit

Date: 2026-08-12  
Stage decision: **correctness/protocol PASS; real efficacy FAIL; P4/P5 blocked**

## Identity and integrity

- Returned result SHA-256: `8c5506dff5f82492a6463a0d63e81da852edea18bf5f2522c029c12d6d660b35`.
- Returned repacked source SHA-256: `c024327c3ee3e580563684c8a7a22039d830ad6a8b36f36f3aefc3c9eb1e2782`.
- Formal source artifact SHA-256 recorded by the run: `d62227f9daaeab1e8dce83cb5affb2e9be9f1e449a6a7fdcaea4fe00b08a829c`.
- Source tree SHA-256: `591d5d26258a89181d2f50ab31d838bf46ebaeecec87cb7358ac59ab6f3b56fb`.
- Production code SHA-256: `e32575d0096cbbc3718edade7489074ad4c280f6171302709da21600190f11b6`.
- Config SHA-256: `cd5207f92e82ae553479cc8049446f35b0c06750545aa960ae0afabdc1067dd4`.
- EvidenceRegistry: 97 valid hash-chained events; head
  `916f641bae98363f3bdba833f43def7523e884f0e7956669b6cad9a92f5960c1`.
- Evidence export manifest SHA-256:
  `24a560a82293eca80139ab6a569b5b5ba47d12a9e59abcea6d1594a42064c563`.

All registered exports, source-tree identity, production-code identity, config
identity, and EvidenceRegistry commitments were independently reproduced. The
uploaded source is a byte-level repack, but its declared source tree and
production-code identities agree with the formal source used by the run.

## Protocol audit

The run completed 96 of 96 dataset-seed-policy cells with zero failed or
replaced runs. It contains 3,168 learning-curve rows, 3,072 query rows, and 15
paired family-level comparisons. Each dataset-seed cell shared the initial,
validation, and acquisition-pool commitments, selected likelihood power,
calibration hash, initial class partition, observation budgets, candidate
evaluation budget, and seeds across all policies.

Budgets were 32 initial observations, 32 acquisitions, 256 validation
observations, 128 visible pool candidates, and 3,600 candidate evaluations per
policy. Untouched held-out remained closed and was not used for selection.
Gas Turbine CO and NOX were grouped as one dataset family.

## Independent effect reproduction

Negative nAULC deltas favor PCPI. Positive frozen-class-gain deltas favor PCPI.

| Family | Baseline | mean nAULC delta | 95% paired CI | predictive NT | mean class-gain delta | 95% paired CI | class NT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CCPP | random | -0.02548 | [-0.06001, 0.00905] | 0.125 | approximately 0 | crosses 0 | 0.625 |
| CCPP | uncertainty | -0.00034 | [-0.00214, 0.00147] | 0.250 | approximately 0 | crosses 0 | 0.375 |
| CCPP | QBC | -0.00579 | crosses 0 | 0.500 | approximately 0 | crosses 0 | 0.500 |
| Gas family | random | -0.00556 | [-0.06905, 0.05793] | 0.500 | 0.00348 | [-0.12596, 0.13291] | 0.500 |
| Gas family | uncertainty | -0.00856 | [-0.02897, 0.01185] | 0.500 | 0.03550 | [-0.10427, 0.17527] | 0.500 |
| Gas family | QBC | -0.02384 | [-0.06031, 0.01263] | 0.375 | 0.07447 | [-0.02371, 0.17265] | 0.250 |

No family-level primary effect is significant. Frozen-class gain versus random
is not positive in every family, and class negative-transfer is not controlled.
Therefore the preregistered strong-evidence Gate fails.

## Posterior and decision diagnostics

CCPP begins with mean frozen-class entropy `8.43e-12`; its mean terminal entropy
is `5.41e-16`. This is numerical collapse, not meaningful sequential class
concentration. Only 1.17% of CCPP PCPI queries use certified class EIG; 98.83%
use the epistemic fallback. The selected sets overlap uncertainty acquisition
by 99.22% on average.

The likelihood-power calibration selected eta=1 for all eight CCPP seeds. It
selected eta below one for only 3/8 Gas CO and 2/8 Gas NOX seeds. Consequently,
the P3B.4 calibration does not repair the dataset family with the strongest
initial class collapse.

PCPI costs 58.77 seconds per policy run on average, compared with 0.63--0.68
seconds for the three baselines, or about 87 times their policy cost. This
compute premium is not matched by a statistically resolved acquisition gain.

## Root cause and repair boundary

P3B.4 chooses eta by leave-one-out mixture predictive log density. Under model
misspecification, this objective can prefer a concentrated mixture while the
posterior over individual models remains unsafe. It is therefore misaligned
with structural-posterior calibration. In addition, raw square, cubic, and
interaction columns receive the same isotropic coefficient prior despite very
different scales and correlations; finite-bank marginal likelihoods can then
depend on arbitrary basis parameterization.

P3B.5 repairs these two statistical causes without changing dataset-specific
formulas, directions, thresholds, class definitions, budgets, seeds, or held-out
state:

1. learning-rate selection uses prequential posterior-expected
   posterior-randomized log loss (R-log SafeBayes);
2. every closed basis term is centered and scaled using only the frozen initial
   development covariates before the shared coefficient prior is applied.

The next allowed action is the P3B.5 controlled correctness diagnostic. A new
heldout-closed real run is allowed only if that diagnostic passes. P4, P5, motif,
and VED remain blocked.

## Claim boundary

P3B.4 supports a valid, reproducible negative real-development result and a
protocol-correct matched-budget run. It does not support acquisition
superiority, posterior-calibration success, held-out confirmation, scientific
discovery superiority, motif safety, intervention, or VED claims.
