# P3E.3 predictive-calibration correctness audit — 2026-08-15

Status: **PASS; correctness evidence only; no real-data calibration or efficacy claim**

The deterministic fixture validates the new task-independent PIT calibration
contract before any local measured-data execution. It does not access CCPP,
Gas Turbine, validation responses, held-out responses, or acquisition policies.

## Identity

| Item | Value |
|---|---|
| Stage | `P3E.3` |
| Config SHA-256 | `294d40ed30d242a146098acf679921364e7e8366e1d8b66fa99003b61412031e` |
| Correctness runner SHA-256 | `9c82ec64357870e51c9cffa016df7036fed976045803132a5d156b4861fdc1ab` |
| Production-code hash at fixture run | `a57bc56feeebde576af7b2204e3c3e752a00e3cffc26fb75db7854727f2c842c` |
| Fixture hash | `9067f73a7c53e5dc2bb07ee3495ab74ca24c2704341097d7debafe80028cc07d` |
| Correctness output | `summary.json`, `config.json` |

## Gate result

All five registered decisions pass:

| Decision | Result |
|---|---|
| Uniform PIT basis moments | pass; maximum absolute grid mean `4.998999930023238e-09` |
| Balanced PIT fixture | no threshold crossing; maximum e-value `1.0` |
| Concentrated PIT fixture | threshold crossing at round `11` |
| Predictive-CDF row-order equivariance | pass; maximum error `0.0` |
| Prequential prefix ignores future validation responses | pass; maximum error `0.0` |

The e-process has 12 fixed betting strategies and threshold `100` at
`alpha=0.01`. The concentrated fixture is intentionally a rejection control;
it is not a real-data result.

## Claim boundary

The fixture supports only the implementation statement that the fixed PIT
betting factors, mixture e-process algebra, predictive-CDF order equivariance,
and no-future-response sequencing are internally consistent. It does not
establish predictive calibration, posterior adequacy, efficacy, held-out
performance, acquisition safety, or scientific discovery. The next real step
is the separately frozen CCPP validation-role audit in
`docs/pcpi_p3e3_predictive_calibration_protocol_20260815.md`; acquisition
remains blocked regardless of its non-rejection outcome.
