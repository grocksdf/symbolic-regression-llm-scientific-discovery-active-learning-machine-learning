# PCPI P3I.4 real-development result and failure audit

## Immutable outcome

The unique user-executed P3I.4 run completed all 96 policy runs and all 3072
acquisition queries with zero failures. The protocol Gate passed, the evidence
registry independently verifies all 97 events with head hash
`94cfe1c08dc0b687deb60a68a14a88c88c6a20cd4d7ef0772db490cb61493b79`,
official measured-data hashes and all split commitments match, and held-out
data remained closed. The P3I.3 shared-H0 defect is therefore repaired.

The frozen output is
`outputs/p3i4_shared_h0_real_acquisition_376a8714_20260831`, bound to commit
`376a8714ad30cf16a793dcc76863e9c3070f750a`, tree
`598deaac3892a081b828e7125b043c61ec85b9ae` and config SHA-256
`571483bdccbeba76b94ba13cb8c9a488e623fd0f01f97f22b50a53067164e6e1`.

The terminal effectiveness status is `REAL_ADVANTAGE_NOT_DEMONSTRATED`.
`formal_protocol_evidence` is true, while `formal_efficacy_evidence`,
`strong_evidence` and `strong_structural_evidence` are false. P3I.4 is valid
negative development evidence. It must not be rerun, overwritten, presented
as an advantage, or used to tune a coefficient, threshold, seed, budget or
stopping rule.

## Registered paired assessment

Against random selection, the CCPP-family paired mean frozen-class entropy-gain
difference was `-0.2617454`, with 95% interval
`[-0.5217038, -0.0017869]` and negative-transfer rate `0.75`. Its predictive
nAULC difference was `-0.0092850`, but the interval
`[-0.0439803, 0.0254103]` crossed zero.

For the pooled Gas Turbine family, the class-gain difference versus random was
`0.1257821`, with interval `[-0.0593944, 0.3109585]` and negative-transfer
rate `0.375`, above the registered `0.25` maximum. Its predictive nAULC
difference was `0.0059268`, with interval `[-0.0608972, 0.0727507]`; lower is
better, so neither direction nor noninferiority was established. Comparisons
against uncertainty and QBC likewise did not establish an all-family class-risk
or predictive advantage.

## Failure is statistical, not numerical

Every PCPI query had an effective certified decision. The representative-safe
set was always nonempty, with minimum size 20 for CCPP, 48 for Gas CO and 63
for Gas NOx. Only one of 768 PCPI queries required a two-candidate numerical
frontier; all others had one possible maximizer. Hence quadrature termination,
the representative guard and the P3I.3 identity defect do not explain the
negative result.

In CCPP, mean registered PCPI score was `0.0014724`, but mean realized local
class-entropy gain was `-0.0077463` and `57.42%` of query-local gains were
negative. This is direct evidence that the model-relative information score
and the realized scientific class update were poorly aligned on that measured
family. These figures diagnose the frozen protocol only; they do not select a
new hyperparameter.

## Structural root cause

P3I.1 correctly proved that a residual correction applied as one common,
strictly monotone response transport preserves class mutual information.
Consequently the P3I.4 production scorer binds and validates the four P3H
residual laws, but computes the acquisition lower envelope with the original
finite Student-t class-EIG. The residual history can change reported response
marginals, yet by construction it cannot change an action ranking or repair a
misspecified class/response dependence.

This is not an implementation accident relative to the frozen P3I contract:
it is the unavoidable limitation of marginal-only calibration. Reusing the
old P3H I-projection would merely choose an unidentified copula. The admissible
next step is instead a normalized class-conditional construction. For each
frozen class `c`, let `F_c` and `f_c` be its current base conditional forecast
and let `g_c` be its own leakage-safe prequential raw-PIT law. Then

`q_c(y) = g_c(F_c(y)) f_c(y)`

is normalized class by class, and `p(c) q_c(y)` is a coherent joint law that
preserves the registered class prior while allowing residual evidence to alter
class information. The exact same `q_c` factors must update the class posterior
after a reveal; acquisition-only recalibration would create another target
mismatch. P3J begins with response-free correctness and does not authorize a
new measured-data run.

## Artifact identities

| artifact | SHA-256 |
|---|---|
| `RUN_MANIFEST.json` | `461b9a51b80c5c1191a9defc166dea721523091cd768a152fa1c89321b42bb97` |
| `summary.json` | `ee3946d0b95266d7ca1b1a457c02f4f7f4e829555228d58d0f6e312f0323347d` |
| `hypotheses/effectiveness_assessment.json` | `acffbbf1f446357500cdcf95ef660f2bccfd0393019f41774b11b3063b1f2936` |
| `tables/paired_effects.csv` | `dafdf6268092f53fbb0f4206db3dd9e9e4b2a367e1a7b915ae2d7fe7031da532` |
| `tables/acquisition_queries.csv` | `cc805b8d5a0315b19cd82276fa288500ad8dcef8f8a859b56b1922e42d64586f` |
| `logs/run.jsonl` | `520eac5fc8414b2cca61be19dd7d8f05250c9941e8f8de79953f1d624d3101da` |
| `evidence_registry.jsonl` | `bcb43b8ad140cc8703487afa13b41741a0da5ba381dd6a16ca5254a66205c5e1` |
