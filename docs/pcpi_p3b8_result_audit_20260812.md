# P3B.8 returned real-result audit

Status: **protocol valid; efficacy Gate failed; held-out remained closed**.

The returned P3B.8 bundle contains 96/96 successful runs (three registered
targets, four matched-budget policies, and eight frozen seeds), no failed seed,
a valid 97-event EvidenceRegistry chain, and read-only exports whose hashes
match the registry manifest. The returned source tree and production-code hash
match the source identity used by the run. `heldout_opened=false` and
`selection_used_heldout=false` throughout.

## Preregistered efficacy assessment

The assessment is `REAL_ADVANTAGE_NOT_DEMONSTRATED`. CCPP has a significantly
negative frozen-class gain relative to random: mean paired delta `-0.304986`,
95% CI `[-0.465243, -0.144729]`, with negative transfer in 8/8 seeds. Its mean
predictive nAULC is better than random (`-0.019438`) but worse than uncertainty
(`+0.009312`) and QBC (`+0.003268`). Gas Turbine has favorable mean predictive
nAULC versus all three baselines, but only the QBC interval excludes zero; its
frozen-class-gain intervals all cross zero.

## Decision-target diagnosis

P3B.8 used certified joint EIG in every PCPI query, so the failure is not caused
by numerical certification or the previous class-collapse fallback. On CCPP,
the pooled Spearman correlation between selected joint score and realized local
frozen-class entropy gain is only `0.057086`. Across 32 selected points per seed,
the PCPI acquisition sets overlap uncertainty by `22.375/32` and QBC by
`22.875/32` on average. The unconstrained deterministic argmax therefore remains
highly concentrated in the same high-uncertainty regions while providing no
control over selection-induced covariate shift under model misspecification.

## P3B.9 repair boundary

P3B.9 keeps the P3B.8 posterior, finite structure bank, SafeBayes calibration,
preconditioning, operational classes, joint information score, budgets, seeds,
baselines, data roles, and assessment rules. It adds one response-free decision
constraint: maximize the joint score only among candidates whose addition does
not increase empirical MMD between the observed design and the fixed registered
action domain. The MMD coordinates and RBF bandwidth use covariates only. If no
non-increasing action exists, the minimum-MMD action is selected and explicitly
recorded as a fallback.

This audit does not authorize P4/P5, untouched-heldout confirmation, motif
claims, VED discovery, or real acquisition superiority.
