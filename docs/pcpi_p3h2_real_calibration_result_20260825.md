# P3H.2 real semiparametric calibration result

## Identity and integrity

The user executed the unique frozen P3H.2 calibration-only command. The
terminal archive binds:

- source commit `e647cb79011f264db7d12e17bab7ae9756e8f128`;
- source tree `c3e3531c3153d88d8669082805b4e1161f0dec3a`;
- config SHA-256
  `becb6cc9c2190fdb5540cd2390d799022276fa3a2b1ca106779a6bd04d64a20a`;
- runtime dependency hash
  `b7bf88a64dd375e25c7d679de129c654c3346462215ffa24c425c762fb8bc4a6`.

The runtime hash is the exact P3G.5 CPython 3.11.9 identity. P3H.1 was
re-executed before data loading and all eight algebraic and leakage checks
passed. The recorded hashes of `calibration_runs.csv` and `summary.json` both
match the no-overwrite manifest. All UCI source hashes equal the P3G.5 source
hashes.

## Frozen result

All 24 registered prequential audits completed, each with 256 validation PITs.
None crossed the unchanged per-run boundary of `2400`:

- calibration rejections: `0/24`;
- global predictive-calibration eligibility: `true`;
- terminal status: `CALIBRATION_COMPATIBLE_ACQUISITION_BLOCKED`;
- largest maximum e-value: `31.856805997618764`;
- acquisition executed: `false`;
- acquisition rows: `0`;
- candidate responses used: `false` for every run;
- held-out opened: `false`;
- simulated experiment: `false`.

The family-level descriptive diagnostics are:

| Dataset | largest max e-value | geometric mean max e-value | mean PIT | mean PIT variance |
|---|---:|---:|---:|---:|
| UCI CCPP | 31.856806 | 2.108707 | 0.497246 | 0.086176 |
| UCI Gas CO | 10.448746 | 2.820402 | 0.494853 | 0.081144 |
| UCI Gas NOx | 18.794443 | 2.212871 | 0.500085 | 0.079885 |

The uniform reference variance is `1/12 = 0.083333...`. Across families, the
mean lower- and upper-decile rates lie between `0.0913` and `0.1167`, and
between `0.0933` and `0.1040`, respectively. These are diagnostics subordinate
to the preregistered e-process decision, not replacement acceptance criteria.

## Descriptive comparison with immutable P3G.5

P3G.5 had 8/24 rejections, maximum e-value `8839301.95182447`, and geometric
mean maximum e-value `347.170405146219`. P3H.2 has 0/24 rejections, maximum
e-value `31.856805997618764`, and geometric mean maximum e-value
`2.36099165481667`. The latter geometric mean is approximately `0.00680067`
times the P3G.5 value.

All eight P3G.5 rejected dataset/seed coordinates are nonrejected under P3H.2.
This comparison is descriptive development evidence. It does not erase P3G.5,
prove the innovation model, or create an efficacy claim.

## Interpretation and next authorization

The result supports the narrow conclusion that the task-independent
prequential KT dyadic Pólya-tree reconstruction is compatible with the fixed
PIT betting family on all 24 registered development audits. The simultaneous
repair across CCPP, Gas CO and Gas NOx is consistent with the P3G diagnosis
that the finite conditional residual family, rather than one isolated linear
nuisance, was the main calibration bottleneck.

Nonrejection is not a proof of conditional calibration, residual stationarity
or posterior adequacy. P3H.2 is not an efficacy experiment and supplies no
acquisition, superiority, held-out, discovery or paper-success evidence.

P3H.2 authorizes only a response-free P3H.3 correctness phase that represents
the transformed semiparametric predictive density inside the acquisition
utility. The existing Student-t component EIG code may not be reused as though
the transformed law were still a Student-t mixture. No candidate response,
real acquisition comparison or held-out execution is authorized yet.
