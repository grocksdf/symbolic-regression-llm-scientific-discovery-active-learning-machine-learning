# Iterative expanded bank matrix, version 1

This is a new protocol. The historical `run_aistats_three_arm_predictive_gate.py`
and its old freeze are unchanged. The new runner compares a Full iterative
bank with the frozen engine-only bank from the **same child execution** on
independent *train reporting* responses. It does not execute acquisition.

## What is registered

- A new output root and a new child path for every task and seed. Existing
  paths, including empty directories, stop the run. There is no resume.
- Clean benchmark and mainline Git source identities; an explicit
  `HYPOTHESIS_MVP_ROOT`; hashes for the configuration, HDF5, exclusions,
  symbol-only metadata, provider-env bytes and prerequisite Gates. The exact
  Python interpreter/version and dependency versions are checked again at run
  time. Credential contents are never printed.
- Task names selected by a fixed hash rule after every supplied exclusion
  freeze. Supply **all** earlier development and confirmation freezes. Names
  and input symbols are read; formula expressions and response arrays are not.
- Two seeds and two tasks per family by default. The same initial data,
  action covariates, engine jobs, engine seeds, budget, frozen engine bank,
  candidate-attempt cap and provider failure policy follow the config.
- `Full minus frozen engine` log score on the final cycle is the primary
  predictive effect; seeds are averaged within task, then tasks averaged.
  Every task, including one with no admitted structure, remains in the matrix.
  Passing requires positive task-mean gain and at least one verified
  admission-to-next-gap link. A failure is retained as a result.

The bank-only ablation has no matched non-LLM new-skeleton proposal attempts.
Thus it can quantify the increment of the admitted Full bank relative to its
common engine bank, but **cannot attribute formula recovery to LLM reasoning**.
The freeze and result both record
`formula_recovery_attribution_authorized: false`. A separate non-LLM skeleton
generator, with the same registered proposal-attempt ceiling, admission roles,
and candidate cap, is necessary for that stronger claim. The matrix also
does not establish common-class decision or measured sequential efficacy.

## Execution order

Use the Python environment that runs PySR and the benchmark adapter. Commit
the patch first: the preflight rejects dirty or untracked source. Keep all
data, gates, freezes, and run outputs outside both Git worktrees. Set paths
explicitly; the code does not assume a working directory layout.

1. Run the response-free provider transport preflight with the exact
   `llm_api_url`, `llm_model`, and provider env in the new config. Keep the
   `response-free-provider-transport-preflight-v2` JSON. A 429 stops the
   experiment; it is not an LLM abstention. The preflight now records model,
   URL and credential-file identity without recording the credential.
2. Run the correctness gate with the explicit mainline path and your frozen
   Windows interpreter. This invokes response-free iterative tests and
   checks `h5py`, `pyarrow`, `pysr`, `yaml`, and `pytest`. It must return zero.
3. Build a fresh freeze. Example (expand exclusions to **all** earlier task
   allocations):

   ```powershell
   $Python = 'D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe'
   $Mainline = 'D:\01\666\hypothesis_mvp'
   $Bench = 'D:\01\666\llm-srbench'
   $env:HYPOTHESIS_MVP_ROOT = $Mainline
   & $Python "$Bench\scripts\check_iterative_matrix_correctness.py" `
     --mainline-root $Mainline --output NEW_GATE.json
   & $Python "$Bench\scripts\build_iterative_expanded_matrix_freeze.py" `
     --mainline-root $Mainline --hdf5 DATA\lsr_bench_data.hdf5 `
     --provider-env PRIVATE\provider.env --provider-gate NEW_TRANSPORT.json `
     --correctness-gate NEW_GATE.json `
     --exclude-freeze OLD_DEVELOPMENT_FREEZE.json `
     --exclude-freeze OLD_CONFIRMATION_FREEZE.json `
     --family phys_osc `
     --metadata phys_osc=DATA\data\lsr_synth_phys_osc-00000-of-00001.parquet `
     --output-dir NEW_FREEZE_DIRECTORY
   ```

   More families require one `--family` and matching `--metadata` per family.
   If too few fresh tasks remain, the freeze stops; do not silently reuse
   confirmation tasks. Inspect task names, exclusions, source hashes and
   budgets before executing. No result-dependent edits are allowed afterward.
4. Run `& $Python "$Bench\scripts\run_iterative_expanded_matrix.py"
   --freeze NEW_FREEZE_DIRECTORY\ITERATIVE_MATRIX_FREEZE.json
   --output-root NEW_RUN_ROOT`.
   First, **all** child artifacts are generated and validated. Only after
   `GENERATION_FROZEN.json` is written does reporting read any reporting
   responses. A failed child, provider failure, missing bank trace or bad
   identity writes `MATRIX_FAILURE.json` and prevents reporting. Restarting
   requires a newly registered output root and, for a formal study, an
   explicit new run identity; do not treat a partial run as successful.

The resulting `MATRIX_RESULT.json` reports per-coordinate effects, task means,
the pass flag, provider request counts, artifact/report hashes, and the
bank-only claim boundary. The original 36-row formula-discovery NO-GO is not
modified by this protocol.
