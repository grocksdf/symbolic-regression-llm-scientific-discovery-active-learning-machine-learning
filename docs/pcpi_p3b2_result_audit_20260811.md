# P3B.2 real matched-budget result audit

Audit date: 2026-08-11  
Stage decision: **protocol PASS; acquisition-efficacy Gate FAIL; stop P4/P5**

## Artifact identity

- Result archive SHA-256:
  `4dc5516a318261db8484fa81bb0afd270146f455b4fb7da349a31a00b9629007`.
- Exact source-package SHA-256 recorded by the run:
  `54751cf1eec3ea5b340d54feea9158f3b151ff45d6a94a060c5fe8bf21326472`.
- Source-tree SHA-256:
  `1e367cfccd2e345c9495e1c2c3ad075fefb7267ebfab335e870292e230d72b71`.
- Production-code SHA-256:
  `f7d250e0038ad49d8cca2f81c7749a923256371a06eb239de0595ae0ac7a5592`.
- EvidenceRegistry: 97 valid hash-chained events, head
  `5aa4ac14f8dd053933fff0c71c8adf38d45b7df9ca237ce193b651578fb6ec58`.

The archive contains 96/96 successful policy runs, 3,072 query decisions, and
3,168 learning-curve rows. All official-file, paired-subset, matched-budget,
candidate-evaluation, and partition-sharing protocol checks passed. CCPP and
the grouped Gas Turbine CO/NOX targets count as two dataset families. No failed
seed was replaced.

Untouched-heldout remained closed. The run manifest and every evidence record
state `heldout_opened=false` and `selection_used_heldout=false`.

## Preregistered family-level effects

The signs below follow the stored convention: negative normalized-AULC RMSE
differences favor PCPI, while positive class-entropy-gain differences favor
PCPI.

| Family | Baseline | Mean delta nAULC RMSE | 95% paired CI | Mean delta frozen-class gain | 95% paired CI |
|---|---|---:|---:|---:|---:|
| CCPP | random | -0.018301 | [-0.056437, 0.019835] | -3.89e-16 | [-2.06e-15, 1.28e-15] |
| CCPP | uncertainty | 0.005151 | [-0.019877, 0.030178] | -5.00e-16 | [-2.42e-15, 1.42e-15] |
| CCPP | QBC | -0.000034 | [-0.026185, 0.026117] | -6.94e-16 | [-1.79e-15, 4.04e-16] |
| Gas Turbine | random | -0.008872 | [-0.054636, 0.036892] | -0.056962 | [-0.165457, 0.051534] |
| Gas Turbine | uncertainty | 0.001711 | [-0.037968, 0.041390] | -0.063641 | [-0.196890, 0.069608] |
| Gas Turbine | QBC | -0.017710 | [-0.058460, 0.023039] | -0.079314 | [-0.205203, 0.046575] |

No family-level structural effect is positive and significant. Every
predictive family-level interval crosses zero. The stored assessment is
`EIG_RANKING_UNCERTAIN_NO_STRONG_CLASS_CLAIM`; both
`strong_evidence` and `strong_structural_evidence` are false.

## Root-cause diagnosis

The failure is not evidence that the Gauss--Jacobi integration budget merely
needs to be increased. CCPP initial class entropy has median
(4.45\times10^{-13}), so class EIG is numerically zero before acquisition.
Its certified-ranking rate is 0.0078125. The grouped Gas family rate is
0.646484375, yet its frozen-class gain is worse than every baseline on average.

P3B.2 optimized a newly recomputed dynamic partition (C_t), but judged the
structural endpoint using the initial frozen random variable (C_0). It
therefore optimized a different decision target from the one reported. When
the current partition collapsed to one class or numerical envelopes could not
separate the leading actions, the rule also lacked a useful certified
posterior-discriminative action.

## Fundamental repair

P3B.3 fixes the structure-to-class map at (C_0) throughout each policy run.
At every posterior update, only the mass of those same classes changes. The
primary utility is therefore
[
I(C_0;Y\mid a,H_t),
]
which is aligned with the frozen structural endpoint.

If this EIG has a certified maximizer, P3B.3 selects it. If (C_0) is a
singleton or the preregistered numerical envelope remains unable to certify a
leader at the unchanged maximum quadrature budget, it selects posterior
epistemic variance of the latent mean, excluding observation noise. This
fallback is derived from the same posterior, contains no dataset-name branch,
answer expression, direction code, result-conditioned threshold, or
held-out-derived value.

All observation budgets, candidate-evaluation budgets, seeds, class distance,
quadrature settings, baselines, effect thresholds, and failure rules remain
unchanged. Dynamic classes are retained as diagnostics only.

## Claim boundary and next Gate

P3B.2 supports a valid negative real-development result and a correctly
executed matched-budget protocol. It does **not** support real acquisition
superiority, class-discrimination superiority, open-grammar discovery,
physical intervention, held-out confirmation, motif safety, VED discovery, or
a new law.

P3B.3 must first pass its controlled decision-rule diagnostic and then be run
once on the same registered real-development splits with held-out closed. The
effect and negative-transfer Gates remain unchanged. If P3B.3 also fails, the
next repair target is posterior misspecification or overconfidence, not a
dataset-specific threshold or acquisition-direction patch.
