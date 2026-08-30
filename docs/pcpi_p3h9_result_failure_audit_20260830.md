# PCPI P3H.9 real-development result and failure audit

## Immutable outcome

P3H.9 completed rather than crashing. The user-executed run recorded all
`96/96` policy runs and all `3072/3072` acquisition rounds, with zero failed
runs. The protocol Gate passed, official measured-data hashes were verified,
the source worktree was clean, the evidence registry was valid, and held-out
data remained closed. Runtime was `25464.0089` seconds (about 7 h 4 min).

The terminal effectiveness status was
`OPERATIONAL_CLASSES_DEGENERATE_NO_CLASS_CLAIM`. Both `strong_evidence` and
`strong_structural_evidence` were false. This is protocol-valid negative
development evidence, not an execution failure and not a result that may be
rerun with changed seeds, thresholds, budgets or stopping rules.

The frozen source identity was commit
`cf7b1b0ffd8b0e06fcb785453db4f8c5767291cc`, tree
`3670cb573b8a58c78e67b3062113b8afbe0c6cb2`, and config SHA-256
`946c64669fac269caea786cd4e2902e38016f286f253f6d00384746f6b893d05`.

## Registered assessment

- The total PCPI decision-valid rate was one in both dataset families.
- The effective ranking rate was one after the registered interval resolver;
  the primary-only rate was about `0.992--0.996`, and secondary resolution was
  used on only about `0.4--0.8%` of PCPI queries. P3H.8 therefore repaired the
  numerical termination failure, but it was not the cause of weak efficacy.
- Initial operational-class aggregation occurred for CCPP but not for the Gas
  Turbine family. All six Gas structures remained separate under the frozen
  quantile/noise distance.
- PCPI did not achieve lower predictive nAULC than every baseline in every
  family, did not beat random class gain in every family, and did not control
  the registered class-gain negative-transfer rate.
- In CCPP, the paired mean class-gain difference versus random was
  `-0.17644`, with 95% interval `[-0.34882, -0.00406]` and negative-transfer
  rate `0.875`. Lower nAULC favors PCPI; its paired mean nAULC difference
  versus random was `-0.01702`, but the interval `[-0.06034, 0.02630]`
  crossed zero.
- In the pooled Gas family, the paired nAULC difference versus random was
  `0.02147` with interval `[-0.03231, 0.07525]`, and the class-gain difference
  was `0.08688` with interval `[-0.08935, 0.26310]`. Neither supplied a stable
  paper claim.

## Statistical root cause

The frozen P3H.9 implementation combined three different decision objects:

1. the operational class map was still built from the legacy Student-t
   posterior-predictive quantiles and pair-specific pooled predictive scales;
2. the class-information term used the P3H.3 marginal-preserving KL-projected
   class/response coupling; and
3. the conditional prediction term still used the legacy Gaussian-moment
   Student-t EPIG surrogate.

Thus the semiparametric residual marginal, the scientific class estimand and
the second part of the acquisition utility did not arise from one predictive
joint law. The equal-weight sum also let predictive nuisance information
compete directly with the primary scientific class target. Descriptively, the
conditional EPIG term supplied about `61.8%` of selected joint utility in CCPP,
where mean realized query-local class gain was negative and `55.1%` of query
gains were below zero. These development observations diagnose the frozen
protocol; they are not used to fit a new coefficient or threshold.

P3H.3 explicitly acknowledged that a corrected response marginal does not
identify a unique class/response joint and registered an I-projection as an
additional modelling choice. P3I identifies the task-independent alternative:
push the complete base class/response joint through the common monotone PIT
transport. This preserves the base copula and, because the map is invertible,
preserves class mutual information. A marginal calibration layer alone cannot
legitimately create new information about the class.

## Evidence identities

The immutable output is
`outputs/p3h9_interval_resolved_real_acquisition_cf7b1b0f_20260829`.

| artifact | SHA-256 |
|---|---|
| `RUN_MANIFEST.json` | `e33c9edffada0612d57033b4a56255ed69428cf0299dcb6bf818b8799d3f2409` |
| `summary.json` | `82aa819e71ec3097720ff17ec67cb8ff7052345029f0f6505341fd6c527c20ab` |
| `hypotheses/effectiveness_assessment.json` | `030e9222e2ea93a6eb6e7d119c3598c892b97e83b1f8721288cb42560c61cd0c` |
| `tables/paired_effects.csv` | `845d1c0ea77f8b81fee5e9aefadf521b2678cf948a55561358faaaf1db988676` |
| `logs/run.jsonl` | `14901faa360ecd417dbb4752d79b1fa3a52eb53d4b1ae2f78c5832ea823d9322` |
| `evidence_registry.jsonl` | `b1fb15288e1c5bf36c802e87fdd7b8587ee763b413050ff28292c104e10caa9d` |

## Decision

P3H.9 is permanently closed and must not be rerun. It establishes that the
registered interval decision is operationally valid, while its class map and
joint utility do not support the desired efficacy claim. P3I may use the
development result only to state this failure; its replacement class metric
and target-only utility must be justified independently and pass response-free
correctness before any further measured-data protocol is considered.
