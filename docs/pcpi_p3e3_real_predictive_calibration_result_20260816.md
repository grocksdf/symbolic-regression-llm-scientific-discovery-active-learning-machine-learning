# P3E.3 returned real predictive-calibration audit — 2026-08-16

Decision: **PROTOCOL COMPLETE; GLOBAL PREDICTIVE-CALIBRATION ELIGIBILITY
FAILED.**

The returned CCPP validation-role audit was tied to source commit
`c0b48ab618680783011d1002ed15f4cac05fc1cb`, production-code hash
`a57bc56feeebde576af7b2204e3c3e752a00e3cffc26fb75db7854727f2c842c`,
and config hash
`2df76ba5ab577894f055244c96c819d704d136dc0e87ce193e3fdb01326fb3d9`.

All eight registered seeds completed. Sequential PIT uniformity was rejected
for seeds `2026080701`, `2026080706`, and `2026080707`. Only five of eight seeds
were eligible for the proper nominal-marginal interpretation, so
`global_predictive_calibration_eligible=false`.

The held-out role remained closed, validation responses were used only for the
registered calibration diagnostic, and no acquisition policy was run or
compared. This is negative diagnostic evidence against the frozen predictive
law; it is not acquisition evidence and cannot be repaired by changing seeds,
thresholds, budgets, or post-result regularization.

The returned archive bytes are not embedded in this source delivery. This
document records the audited decision and frozen identities; independent
archive-byte verification still requires the user's retained returned archive.
