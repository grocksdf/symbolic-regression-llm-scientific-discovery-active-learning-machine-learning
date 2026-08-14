# P3B.9 returned real-result audit

Status: **protocol-valid negative real-development evidence**  
Efficacy assessment: `REAL_ADVANTAGE_NOT_DEMONSTRATED`

## Identity and protocol

- 96/96 registered policy runs completed with zero failures.
- The source tree is `3b642a3f4b1bfa27e2566c149d437368d50909e26e81db20e05eb15769d2dda8`.
- The production-code hash is `d529317f30e2d893089c522f5c259e2b717e6d0196a0a6be2b276c158c81caaa`.
- Held-out remained closed and selection did not use held-out.
- All policies shared the frozen datasets, seeds, splits, initial observations,
  validation subsets, candidate pools, budgets, nominal posterior calibration,
  basis preconditioner, and efficacy rules.
- Gas Turbine CO and NOX remain one dataset family for inference.

## Representative-guard behavior

The P3B.9 guard behaved as specified. On all 256 CCPP PCPI queries the
representative safe set was nonempty, the selected action did not increase
MMD, and no minimum-MMD fallback was needed. The negative efficacy result is
therefore not explained by a missing guard or packaging failure.

## Frozen efficacy result

For CCPP, PCPI frozen-class entropy gain relative to random had mean paired
difference `-0.2509967941549854` with 95% interval
`[-0.4616616356504576, -0.04033195265951314]`. Seven of eight seeds had negative
class-gain transfer relative to random. Across the 256 selected CCPP queries,
the pooled Spearman correlation between selected joint score and realized
local frozen-class entropy gain was `-0.007906557564660103`.

These values do not satisfy the unchanged family-level efficacy Gate. They
cannot support acquisition superiority or a submission-ready C3 claim.

## P3B.10 repair boundary

P3B.10 retains the response-free representative MMD guard. It changes only the
PCPI ranking utility from one calibrated-posterior joint score to the minimum
joint score over the already frozen likelihood-power candidates
`(0.125, 0.25, 0.5, 1.0)`. The nominal calibrated posterior remains the
reporting and evaluation posterior. No dataset or target branch, result-tuned
threshold, formula hint, extra candidate budget, new split, or held-out value
enters this repair.

P3B.10 must pass its controlled 27-decision correctness Gate before any new
heldout-closed real run. Its controlled result can establish implementation
correctness only, not real efficacy.
