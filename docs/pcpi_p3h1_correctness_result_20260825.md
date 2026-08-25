# P3H.1 semiparametric residual-law correctness result

## Frozen identity

The response-free correctness runner completed with `status=passed` on source
commit `db00f91d16b6431df77c9e78300b9293e4572dab` and tree
`71f7677b6a0c909bd142a36c7d6bd07cbcc307c7`.

- config SHA-256:
  `33721e8938aac1ebf5ae0c6b532fe7485a3e9d0364861444f4319fd2bd6d5859`;
- production-code hash:
  `1c70b770a72aba098e8be8d149dc2e4be22f1d81f56de31d53ea9c709755eb03`;
- residual-law source SHA-256:
  `8417e964115d8748cd2d3515125935e23bb7b8f2872eb6796803903f5b44a93e`;
- correctness-runner source SHA-256:
  `a4719b9d818e206a02988f1cda2cf8120a57d7af1fc8ccb37cdcc1d67ed76e51`;
- runtime dependency hash:
  `a36191437d851b34aa08525adcc2fe2bf7fb88db0f82bffa9e32d9fb091ad8e6`.

The `summary.json` hash exactly matches its no-overwrite manifest. The run used
isolated CPython 3.12.13 on Windows AMD64, NumPy 2.5.2, SciPy 1.18.1 and
python-flint 0.8.0; the complete installed-distribution snapshot is embedded
in the summary. This isolated correctness runtime was used because the prior
Python 3.11 virtual-environment launcher referred to a removed base
interpreter. It does not silently redefine the runtime of a later real Gate,
which must record its own exact environment.

## Passed decisions

All eight registered decisions passed:

1. positive leaf probabilities normalize exactly;
2. CDF/inverse-CDF composition stays within the frozen tolerance;
3. the transformed density obeys the base-times-residual chain rule;
4. split-count posterior prediction is invariant to storage order;
5. active leaf count is sublinear and determined only by history count;
6. an issued forecast is unchanged by a subsequently supplied response;
7. the scientific base posterior update is unchanged;
8. production source contains no RNG, result, task, seed, threshold or
   held-out branching surface.

Observed numerical errors were:

- probability normalization: `0.0`;
- CDF/inverse composition: `2.7755575615628914e-17`;
- log-density chain rule: `2.220446049250313e-16`;
- scientific base update: `0.0`;
- future-response isolation: `0.0`.

The maximum active-leaf-to-square-root-history ratio was exactly `1.0`, as
required by the registered depth schedule. The deterministic fixture used 64
raw probability coordinates and active depth three.

## Test evidence

The P3G.1--P3G.5 and P3H targeted regression passed 38/38. The complete
repository regression passed 660/660 under an isolated `-S` import surface,
which prevents duplicate desktop and test-directory distributions from
contaminating dependency identity. After adding runtime identity to the
correctness artifact, the P3H plus integrity regression passed 19/19.

## Claim boundary and next Gate

This was an inference-correctness diagnostic fixture. It was not a simulated
experiment and accessed no real data, validation response, candidate response
or held-out information. It establishes finite-precision algebra and
prequential source isolation only. It does not establish residual stationarity,
predictive calibration, real-data adequacy, acquisition correctness, efficacy,
discovery or paper success.

P3H.2 may now be executed by the user as the separately frozen real
calibration-only Gate. Its runner re-executes P3H.1 before dependency snapshot
or real-data loading. Candidate responses and acquisition remain blocked even
if all 24 calibration runs are nonrejected.
