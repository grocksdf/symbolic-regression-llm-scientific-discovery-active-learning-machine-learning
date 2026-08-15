# P3E.1 posterior/decision root cause and update-coherent repair

Status: **CORRECTNESS FIXTURE PASS; REAL INTEGRATION AND RERUN BLOCKED**

## Root-cause matrix

| Layer | Evidence | Diagnosis | Can P3E.1 repair it? | Consequence |
|---|---|---|---|---|
| Evidence identity | Clean commit, verified data/splits, valid 97-event registry, 96/96 runs | Not a provenance or completeness failure | No repair needed | Retain P3D.2 as valid negative development evidence |
| Numerical ranking | Decision-valid rate 1.0; certification 0.789/0.871 by family | Intervals and reference handover executed as coded | No repair needed | Do not add another numerical guard |
| P3E.2 numerical reproducibility | Canonical fixture initially varied in strict union-orthogonality error across BLAS/NumPy environments | Near-null eigensolver modes leaked tiny components into the structure union; this is a floating-point contract issue, not posterior or utility evidence | **No: separate task-independent reprojection repair** | Reproject retained discrepancy columns, then require a clean canonical rerun before treating source identity as closed |
| Class availability | Class aggregation observed in both families | Negative result is not caused by zero class capacity | No repair needed | Preserve frozen class endpoint |
| Update/utility semantics | Ordinary class-MI score, but generalized update `p(y|z)^eta`; 13/16 Gas calibrations have `eta<1` | Score is not expected loss reduction under the actual update | **Yes: exact alignment repair** | Replace MI interpretation by update-coherent signed utility |
| Posterior adequacy | CCPP has `eta=1` yet frozen-class gain versus random is -0.2538, CI excluding zero | Separate posterior/predictive misspecification remains | **No** | A P3E.1 pass cannot authorize a real rerun |
| Score-to-realized utility | Selected score/gap has only weak association with realized local gain; targeted mean gain is negative on CCPP and NOX | Model-relative utility does not reliably transfer to measured entropy gain | Only model-relative abstention | Require positive lower bound, but do not claim realized safety |
| Cross-target stability | CO positive, CCPP/NOX negative | Repair cannot be dataset- or target-name branched | No task-specific patch permitted | Any later posterior repair must be common across tasks |

## Exact semantic mismatch

Let `q_t(z)` be the current finite-bank structure probability, `C=g(z)` the
initial-frozen operational class, and `p(y|z,a)` the nominal response model.
P3D.2 computes

\[
I_{q_t p}(C;Y_a), \qquad
m_t(y|a)=\sum_z q_t(z)p(y|z,a).
\]

The implemented observation update is instead

\[
q_{t+1}^{(\eta)}(z\mid y,a)
=\frac{q_t(z)p(y\mid z,a)^\eta}
{\sum_{z'}q_t(z')p(y\mid z',a)^\eta}.
\]

For `eta=1`, this is the conditional distribution induced by `q_t(z)p(y|z,a)`
and the mutual-information identity holds. For `eta != 1`, the generalized
update is not that conditional distribution, so ordinary mutual information
is generally not the expected frozen-class entropy reduction of the update.

P3E.1 defines the narrow, signed, loss-aligned diagnostic utility

\[
V_\eta(a)=H_{q_t}(C)-
\mathbb E_{Y\sim m_t(\cdot|a)}
\left[H_{q_{t+1}^{(\eta)}(\cdot|Y,a)}(C)\right].
\]

This is deliberately not renamed "generalized EIG." It is the expected
change in the registered frozen-class log-risk under the nominal designer
outcome distribution and the actual implemented update. It reduces exactly to
ordinary class mutual information at `eta=1`, but it is signed and may rank
actions differently when `eta != 1`.

## Task-independent decision contract

Given simultaneous containing intervals
`L_eta(a) <= V_eta(a) <= R_eta(a)` and a response-free registered reference
policy `q_ref`, define

\[
R_\eta(q_{\rm ref})=\sum_a q_{\rm ref}(a)R_\eta(a).
\]

P3E.1 authorizes targeted handover only when

\[
L_\eta(\widehat a)>
\max\{R_\eta(q_{\rm ref}),0\}+\tau_{\rm num}.
\]

Otherwise it executes the exact registered reference draw. Conditional on the
intervals and declared model, the targeted branch therefore dominates both
the reference policy and no expected frozen-class improvement. This is still
not a realized no-harm theorem.

## Correctness-only implementation

The isolated finite fixture is implemented in
`hypothesis_mvp/pcpi/reference/update_coherence.py`; it is not imported by the
real acquisition runtime. `configs/p3e_1_update_coherence_diagnostic.json`
contains no datasets, data root, validation budget, policies, held-out access,
or efficacy thresholds.

The exact Gate checks:

1. `eta=1` recovers ordinary class mutual information;
2. an `eta=0.25` finite counterexample reverses the ordinary-MI action ranking;
3. the utility equals the direct outcome-weighted realized entropy change;
4. state, action, and outcome permutations preserve the result;
5. one-class capacity gives zero utility;
6. malformed probability/update contracts fail closed;
7. targeted handover uses the aligned ranking;
8. a negative-utility leader returns to the registered reference policy;
9. the decision lower bound exceeds both reference and zero; and
10. the diagnostic exposes no real-data or held-out surface.

The repository fixture passes all ten diagnostic decisions. This validates the
finite loss/update alignment and decision implementation only.

## AISTATS claim discipline and next blocker

Generalized Bayesian experimental design and robust Bayesian design already
exist; P3E.1 must not be presented as generic generalized-EIG novelty. Relevant
recent primary sources include
[Barlas, Sloman, and Kaski (2025)](https://arxiv.org/abs/2511.07671),
[Forster et al. (2025)](https://arxiv.org/abs/2506.07805), and
[Abdulsamad et al. (2026)](https://arxiv.org/abs/2603.14094).
The defensible contribution at this stage is narrower: an auditable
counterexample and a loss/update-alignment Gate for the exact inference and
decision pipeline.

P3E.2 now states one task-independent generative repair: a response-free RBF
discrepancy basis orthogonal to the union of all candidate designs, paired with
an exact spike/null Bayes-factor e-process. Its finite correctness Gate passes,
but it has not accessed real data, calibrated an augmented real posterior, or
authorized acquisition. Thus it closes only the algebraic part of this blocker.
The CCPP `eta=1` negative transfer remains unexplained on real data. P4/P5,
held-out, motif, VED, intervention, and superiority claims remain blocked.
