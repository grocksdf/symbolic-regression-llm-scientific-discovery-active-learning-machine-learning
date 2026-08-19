# P3F.4-CERT.1 local runbook

This package runs a deterministic certification audit. It does not run SMC,
open held-out state, or access real data.

## Environment

From the repository root, install the project and test dependencies in a clean
Python 3.11 or later environment:

```bash
python -m pip install -e '.[test]'
```

## Correctness tests

```bash
python -m pytest -q tests/test_pcpi_p3f4_certification_layer.py
```

The registered test file contains eight tests covering:

- shell-by-shell raw multiplicity conservation;
- the exact \(J=17\) count reduction;
- semantic identifier and evaluation agreement;
- equality with the raw \(L=3\) exact-reference evidence;
- the zero-likelihood-power prior identity;
- the uniform fractional-likelihood envelope;
- rank-one beta-grid agreement with independent calculations; and
- the envelope-proposal domination inequality.

## Static development certificate

```bash
python -m scripts.run_pcpi_p3f4_certification_layer \
  --config configs/p3f_4_semantic_envelope_certification_development.json \
  --output p3f4_certification_results
```

The correctness-first implementation may take several minutes. Successful
completion writes:

```text
p3f4_certification_results/summary.json
```

Required top-level result:

```json
{
  "all_certification_decisions_passed": true,
  "completed_fixture_count": 3,
  "heldout_opened": false,
  "real_data_accessed": false,
  "resident_smc_modified": false,
  "smc_executed": false,
  "next_gate": "unseen-confirmatory-certification-freeze-eligible"
}
```

Each fixture must independently pass:

```text
semantic_prior_mass
likelihood_envelope
posterior_tail
proposal_mixing
bridge_path
```

Any false decision, missing positive bridge, cutoff error, envelope violation,
or bridge-budget excess is a formal NO-GO. Do not replace a failed analytic
certificate with empirical particle ESS.

## Integrity rules

- Do not edit the cutoff, grid, thresholds, fixture responses, or priors after
  observing a result.
- Do not interpret this development output as confirmatory SMC fidelity or
  paper-level efficacy evidence.
- Do not connect the envelope proposal to resident-SMC until a separate unseen
  confirmatory certification freeze passes and the integration contract is
  reviewed.
