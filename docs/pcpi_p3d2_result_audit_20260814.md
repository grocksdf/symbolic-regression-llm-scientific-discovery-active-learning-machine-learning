# P3D.2 real result audit — 2026-08-14

Decision: **PROTOCOL PASS; EFFICACY NO-GO; P3E.1 CORRECTNESS REPAIR REQUIRED**

## Immutable evidence identity

- Returned archive: `p3d_2_reference_dominance_real_20260814_v2.zip`.
- Archive SHA-256:
  `df301fd2ec406747b2cc3437ee5fbbd3f4b2c1b324674109cb067655070c3e52`.
- Archive size: `1,687,724` bytes.
- Source commit:
  `2effdeec31faf630c31dad2e3b824774cbaf5e4a` (clean worktree).
- Source tree: `20944aa57eaf9e3ad7b4ba821eb895f73abfa195`.
- Tracked-source hash:
  `4c2c7425ece154dbcffb3e0c2d68d00e0eb22470a22db74e6e0c3f515cbb38cc`.
- Production-code hash:
  `078b2b0d2eae9940769c65a566b2b2d43204af27ecf6ea8f81ed69360bdd10d2`.
- Frozen-config hash:
  `41959e0c4ea850c6198758ae3744ebcd3558b285f937d94a840022d45b0ad498`.
- Config-file SHA-256:
  `2527483cfa2f00a6926a6da3a16a2fbf52f1f8b6d01d4eec64360876878261bf`.
- Runtime-dependency hash:
  `d8eb9c5b56333fcb3a964b7f9d18b21801773378ecfb479ae559a80edca464ad`.
- EvidenceRegistry: valid, `97` events, head
  `47b4132278685fefccf6f415b79281514ee89ff78fa18371958ace728e7f492b`.

All nine files named by the read-only export manifest reproduce their recorded
SHA-256. ZIP integrity, official dataset hashes, split identities, finite
numeric exports, and run/query/curve key uniqueness were independently checked.

## Protocol audit

The run used official measured CCPP and Gas Turbine data on Windows/Python
3.11.9. It completed `96/96` policy runs with zero failures: three registered
targets, eight paired seeds, and four policies. The archive contains 96
per-seed policy rows, 3,168 learning-curve rows (`96 x 33`), and 3,072 query
rows (`96 x 32`). Every successful policy run records 3,600 candidate
evaluations. Held-out remained closed, no LLM was called, and all protocol
decisions passed. Runtime was approximately 137 seconds; this is credible
because the experiment is finite-bank conjugate inference and pool replay, not
open-grammar symbolic search.

Therefore `formal_protocol_evidence=true` is supported.

## Registered efficacy result

The preregistered status is `REAL_ADVANTAGE_NOT_DEMONSTRATED` with
`formal_efficacy_evidence=false`, `strong_evidence=false`, and
`strong_structural_evidence=false`.

| Dataset family | PCPI minus random predictive nAULC | 95% interval | PCPI minus random frozen-class gain | 95% interval | PCPI negative-transfer rate |
|---|---:|---:|---:|---:|---:|
| CCPP | -0.024410 | [-0.062929, 0.014109] | -0.253805 | [-0.482951, -0.024660] | 0.750 |
| Gas Turbine (CO/NOX grouped) | -0.050609 | [-0.122316, 0.021099] | -0.004906 | [-0.223369, 0.213557] | 0.625 |

CCPP establishes harmful structural transfer relative to random under the
registered paired analysis. The grouped Gas interval crosses zero. Although
mean predictive nAULC is lower than random in both families, it is not
nonpositive against every baseline, and the structural conditions fail.

Dataset-level mean frozen-class entropy gains further expose heterogeneity:

| Target | PCPI | Random | Uncertainty | QBC |
|---|---:|---:|---:|---:|
| CCPP | -0.243031 | 0.010775 | -0.273565 | -0.235047 |
| Gas CO | 0.117230 | -0.028966 | 0.210310 | 0.130165 |
| Gas NOX | -0.136871 | 0.019136 | -0.019045 | 0.054191 |

## Decision/posterior diagnosis

Reference-dominance itself executed as registered: its valid-decision rate is
1.0 in both families, and ranking-certification rates are 0.7891 for CCPP and
0.8711 for grouped Gas. Class aggregation occurred in both families. Thus the
negative result cannot be relabelled as an absent-class, failed-bound, missing
fallback, or incomplete-run artifact.

The returned posterior is explicitly
`power-likelihood-generalized-bayes`. For likelihood power `eta < 1`, the
posterior engine updates structure mass as

`q_next(z | y,a) proportional to q(z) p(y | z,a)^eta`,

whereas P3D.2 scores ordinary class mutual information under the nominal
mixture `sum_z q(z)p(y | z,a)`. That mutual information equals expected
frozen-class entropy reduction only for the coherent ordinary Bayes update
`eta=1`; it is not the expected loss change of the implemented generalized
update when `eta != 1`. Thirteen of the sixteen Gas target/seed calibrations
used `eta < 1`, so this is a real decision-semantics defect, not merely a
theoretical edge case.

It is not a complete causal explanation. All eight CCPP seeds used `eta=1`,
yet CCPP still had significant negative structural transfer. CCPP therefore
retains independent evidence of posterior/predictive misspecification or
utility-to-realized-endpoint failure. No new real rerun is authorized by a
decision-only correction.

## Claim boundary

P3D.2 supports a protocol-valid negative real-development conclusion. It does
not support acquisition superiority, realized no-harm, posterior adequacy,
held-out confirmation, open-grammar discovery, motif safety, VED discovery,
physical intervention, or a new scientific law. The immutable P3D.2 manifest
is not retroactively rewritten: it correctly identifies the score it used.
Subsequent manuscript text must, however, stop interpreting that score as the
actual generalized-update expected entropy reduction when `eta < 1`.
