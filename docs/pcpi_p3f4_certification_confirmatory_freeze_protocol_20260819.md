# P3F.4-CERT.CF.1 unseen confirmatory certification freeze

Status: **FROZEN AND NOT EXECUTED**

Frozen config SHA-256:

`dc89217920cca81fb91a8d25fa2d1bea1e94086e47f886fbe30a9dbf26d6cfca`

This protocol is the only next gate authorized by the P3F.4-CERT.1
development pass. It evaluates the static semantic-envelope certification
layer on new synthetic fixtures. It does not run or modify resident-SMC.

## Frozen bank

- Four new one-dimensional irregular action grids.
- Two unique PCG64 seeds per fixture.
- Eight confirmatory runs in total.
- Cubic and quartic response generators with frozen coefficients and noise
  scales.
- No materialized target vector is present in the freeze package.

The runner materializes each target only after the config and development
dependencies pass byte-level integrity checks.

## Unchanged controls

| Control | Frozen value |
|---|---:|
| Semantic-core cutoff | 17 |
| Bridge candidate spacing | 1/32 |
| Relative-ESS lower floor | 0.8 |
| Posterior-tail upper ceiling | 0.01 |
| Mixing TV tolerance | 0.01 |
| Maximum certified mixing steps | 1 |
| Maximum bridge steps per observation | 64 |
| Required run count | 8 |
| Decision rule | Every run and every subdecision must pass |

Threshold changes and formal reruns after response materialization are
forbidden.

## Local execution

Run from a clean repository root after overlaying the package files:

```bash
python -m pip install -e '.[test]'
python -m pytest -q \
  tests/test_pcpi_p3f4_certification_layer.py \
  tests/test_pcpi_p3f4_certification_confirmatory_freeze.py
```

The freeze tests must finish with the response state still closed. Then run the
one-shot confirmatory command:

```bash
python -m scripts.run_pcpi_p3f4_certification_confirmatory \
  --config configs/p3f_4_semantic_envelope_certification_confirmatory_freeze.json \
  --output p3f4_certification_confirmatory_results \
  --archive p3f4_certification_confirmatory_results_20260819.zip
```

The correctness-first implementation may take several minutes. The runner
refuses to overwrite an existing output directory or archive.

Whether the gate passes or fails, return the generated ZIP without changing
the config, thresholds, source, result JSON, or archive. Do not rerun a failed
formal decision with a different seed or cleaned output path.

## Decision semantics

PASS requires all eight runs to pass all five subdecisions:

- semantic prior-mass conservation;
- component likelihood envelope;
- posterior-tail ceiling;
- envelope-proposal mixing bound; and
- complete fail-closed bridge path.

If every run passes, the only newly authorized action is a reviewed
resident-SMC envelope-proposal integration contract. It is not permission to
claim final paper success, predictive calibration, real-data efficacy,
acquisition, held-out, or discovery evidence.

Any failed subdecision is a formal NO-GO and returns the project to the
certification layer. Empirical particle ESS cannot replace a failed analytic
certificate.

## Freeze audit completed before delivery

Five static response-free tests passed:

1. config and dependency byte hashes are frozen;
2. no `targets` field exists in the freeze;
3. target and certification thresholds exactly equal the development contract;
4. fixture grids are new and all eight seeds are unique; and
5. import plus integrity preflight leave `responses_materialized=False`.

No confirmatory target was generated while producing this package.
