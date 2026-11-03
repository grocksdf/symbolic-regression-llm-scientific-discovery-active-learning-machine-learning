# Frozen predictive expansion adapter

`scripts/frozen_predictive_expansion.py` is an explicitly selected development
adapter. It does not replace historical frozen runners or open datasets on
import. No real-data run is performed by the correctness tests.

Construct `ExpansionDesign` from registered source row indices, covariate-only
admission region labels, fixed alpha and tested-mixture count. Include every
row used for adaptive generation in `training_ids`. Supply a frozen E0 and
frozen candidate rows to `FrozenBankComponents`; it fits each complete
`E0 + candidate` predictive component once. Its component IDs hash the actual
model and fitted posterior. They are revalidated whenever densities are used.

Call `run_comparison` with three authorized readers returning `(row_ids,
RoleDataset)`, one each for calibration, independent admission and reporting.
Reporting is not loaded until `ADMISSION_FROZEN.json` is fsynced. Before
admission, `CALIBRATION_FROZEN.json` is also fsynced. All outputs are created
exclusively in a new directory; exceptions preserve `FAILURE.json` and earlier
artifacts. No failed coordinate is automatically retried or discarded.

The result contains four same-family comparisons: engine, calibration-only
validation selection, standard stacking, and screened stacking. Reporting
cannot modify weights or admission. Accepted predictions use the exact tested
mixture; rejected predictions use E0. Weights are not posterior law probabilities.

This is a callable integration surface, not a complete matrix launcher or
immutable experiment freeze. A new matrix must register source identities,
candidate pool/row manifests, budgets, exclusions, primary estimand, task-level
aggregation and failure scoring before execution. Old artifacts are not
compatible input registrations for an amended experiment.

Synthetic correctness command (PowerShell):

```powershell
$env:HYPOTHESIS_MVP_ROOT = 'D:\01\666\hypothesis_mvp'
$env:PYTHONPATH = 'D:\01\666\llm-srbench;D:\01\666\hypothesis_mvp'
Set-Location -LiteralPath 'D:\01\666\llm-srbench'
& 'D:\01\666\.venvs\llm_srbench_metadata_py312\Scripts\python.exe' -B -m pytest -q -p no:cacheprovider --basetemp 'D:\01\666\outputs\_predictive_expansion_adapter_check' tests/test_frozen_predictive_expansion.py tests/test_independent_validation_comparator.py tests/test_stacking_diagnostics.py
```

Read the mainline `docs/predictive_bank_expansion_contract.md` for the fixed-law
null assumptions, mechanism counterexample, research limitations and novelty
boundaries. The method has not established real-data gain or an original
generation advantage merely by passing these tests.
