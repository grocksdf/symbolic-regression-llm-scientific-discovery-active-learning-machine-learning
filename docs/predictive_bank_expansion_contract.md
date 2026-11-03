# Frozen predictive Bank expansion: mechanism repair and claim boundary

Date: 2026-11-03. Status: synthetic correctness development, **not efficacy**.

## What this repairs

The historical single-candidate gate tests `q_candidate` against the complete
engine predictive law `p_E`. A candidate can lose that comparison and still
improve a mixture. For example, with probabilities `p_E=(.5,.5)`,
`q=(.9,.1)` and empirical frequencies `(.6,.4)`, the standalone average
log-score change is -0.291103, while `.75*p_E+.25*q` improves it by 0.020136.
This exact construction diagnoses objective mismatch; it is not a real-data
performance result.

There is already a joint-bank admission implementation (`admit_joint_bundle`).
It tests the expanded **BMA** law. The new adapter instead tests a calibrated
**predictive mixture**. The distinction is the tested/deployed law, not a new
name for the existing gate. Historical functions and results remain unchanged.

## Statistical object and lifecycle

1. Freeze a candidate family generated using the registered training/search
   rows. Fit every component once on the same fitting rows: `p_0=p_E` and
   `p_j=p_(E+c_j)`. This definition must appear in the experiment registration.
2. On an independent calibration role, use the existing standard stacking
   solver to fit simplex weights. Freeze `p_w=sum_j w_j*p_j`, the fitted
   posterior identities, row manifest, alpha, tested-mixture count and regions.
3. On independent admission rows, evaluate exactly that `p_w` against `p_0`.
   Keep the original global threshold `log(M/alpha)` and nonnegative evidence
   in every predeclared region. An absent region is a rejection. Do not choose
   regions, weights or candidates after seeing admission.
4. Deploy exactly `p_w` on acceptance, otherwise exactly `p_0`. There is no
   posterior refit, candidate deletion, weight recalibration or extra mixture
   assembly between admission and reporting.
5. Only after a durable admission artifact may independent reporting labels
   be loaded. Preserve negative scores. A failure terminates the coordinate;
   it is not an automatic zero score or silent retry.

The library does not certify that caller-supplied source row IDs are honest.
An experiment harness must bind them to dataset row indices and include ALL
generation/search rows in the training exclusion set. Do not compute these
identities from response values or inspect sealed data to construct them.

The threshold is a fixed-law likelihood-ratio screen under the conditional
product predictive null `p_0`. It is **not** distribution-free risk control,
a Bayesian posterior probability of a formula, evidence that the null is
true, or a guarantee of improved future prediction. Calibration reuse by
adaptive generation would invalidate the independence claim. `M` must count
all registered mixtures tested on the reused admission sample, not only winners.

## Fair baselines implemented in the adapter

All use the same fitted component family and reporting rows:

- Engine: `p_0`.
- Ordinary independent validation: calibration-best complete component.
- Predictive stacking: the calibration-frozen mixture without admission.
- Screened predictive stacking: that exact mixture or core fallback.

This comparison isolates the admission decision from standard stacking's
benefits. It does not isolate LLM reasoning from additional structural search.
The adapter publishes compute counts; a future registration still needs
source/config hashes, candidate-generation costs, seed/task allocation,
failure-inclusive aggregation and the primary contrast. It is not itself a
complete experiment registration or permission to open confirmation data.

## What is NOT repaired or established

CB-SI's current `CrossfitBankScorer` already scores `mixture(E0,E0+candidate)`.
The recent paired-policy loss is therefore not explained by this admission
repair. The audit found missing greedy-winning supports, few nonzero rewards
and a large context model relative to observations. These establish a coverage
and information bottleneck, not that one new allocation rule will fix it.
No policy prior, winning operator, task list, threshold or budget is tuned here.

## Novelty audit and research hypothesis

- [Yao et al., predictive stacking](https://arxiv.org/abs/1704.02030): optimizing
  mixtures of Bayesian predictive distributions is established.
- [Locatello et al., boosting VI](https://proceedings.mlr.press/v84/locatello18a.html):
  adding density components via functional optimization is established. Its
  reverse-KL posterior objective differs from this predictive log score.
- [IGSR official implementation](https://github.com/DrShushen/IGSR): LLM-generated
  symbolic terms, validation removal influence and propose/prune MCTS already
  exist. Merely providing formula feedback to an LLM is not a new contribution.

The testable research hypothesis is narrower: under equal structural-search
opportunity, **Bank-conditional information guides invention of complementary
symbolic structures better than single-formula loss or residual feedback**.
Novelty and efficacy of that hypothesis remain unestablished. The current
repair provides an aligned measurement/deployment path; it does not prove the
hypothesis and must not be presented as an original stacking algorithm.

Before a new efficacy run, first fix a development protocol comparing ordinary
validation, standard stacking and the frozen-mixture screen. Separately test
generation guidance using equal proposal/search opportunities and shared E0.
If ordinary stacking explains the gain, report that result rather than claiming
a novel discovery mechanism. Any final confirmation requires a new untouched
task allocation and a locked generation/scoring implementation.
