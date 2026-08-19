# P3F.4-CERT.2 static development protocol

Status: **IMPLEMENTED FOR CORRECTNESS REVIEW; NOT A FORMAL EXPERIMENT**

## Scope

`P3F.4-CERT.2-DEV.1` implements the response-energy envelope, exact semantic-
core reuse, RE bridge lower bounds, exact conditional raw-tail prior sampling,
independence-MH correction, and dependency-aware tail/anchor decisions.

The kernel scope is fixed to:

`hybrid-state-space-envelope-anchor-only`.

The old CERT.1 module and CERT.CF.1 config/runner remain byte-identical.  The
resident raw-AST SMC and local/RJ kernels are neither modified nor executed.

## Frozen development controls

| Control | Value |
|---|---:|
| semantic cutoff schedule | `[17]` |
| bridge grid | `1/32` |
| maximum bridge steps/observation | `64` |
| relative-ESS lower floor | `0.8` |
| posterior-tail ceiling | `0.01` |
| anchor TV tolerance | `0.01` |
| anchor macro-sweep budget | `1` |
| semantic prior-mass error | `2e-12` |
| likelihood-envelope violation | `2e-12` |
| anchor-normalization error | `2e-12` |

The three AC--AE fixtures remain archived synthetic development correctness
cases.  AF--AI and their eight seeds are already-opened failed-confirmatory
postmortem diagnostics only.  Passing them under CERT.2 cannot amend or replace
the original NO-GO.

## Static Gate

A run passes only if all 11 development cases satisfy:

- exact semantic prior conservation and frozen resource ceilings;
- no registered component above the RE envelope;
- normalized anchor weights;
- posterior-tail upper at most `0.01`;
- anchor TV at most `0.01` in the one-sweep budget; and
- every observation reaches beta one through certified grid steps with
  relative-ESS lower at least `0.8` and at most 64 steps.

If the tail bound fails, anchor mixing is recorded as
`blocked_by_tail_certificate`; it is not counted as a second root failure.

## Required provenance

The runner fails before expensive computation unless the source tree is clean
and the interpreter is Python 3.11.  It writes source commit and tree, branch,
remote, dependency hashes, the exact installed-distribution snapshot and its
hash, interpreter, package versions, wall-clock, config snapshot, full per-run
certificates, bridge paths, claim boundary, and checksums.  Existing output
directories are never overwritten.

The observed development dependency snapshot is not automatically a future
confirmatory lock.  A new confirmatory freeze must commit its exact dependency
lock before any new response is materialized.

## User-side verification command

After the implementation branch is committed and available on GitHub, use the
fixed Python 3.11 environment.  The output name must be new.

```powershell
$project='D:\01\666\hypothesis_mvp'; $python='D:\01\666\.venv_hypothesis_canonical\Scripts\python.exe'; $output=Join-Path $project 'outputs\p3f4_cert2_response_energy_development_20260819'; if (Test-Path -LiteralPath $output) { throw "Output already exists: $output" }; & git -C $project status --short; if ($LASTEXITCODE) { throw 'git status failed' }; if (git -C $project status --porcelain) { throw 'Source tree is not clean' }; & $python -m pytest -q (Join-Path $project 'tests\test_pcpi_p3f4_certification_layer.py') (Join-Path $project 'tests\test_pcpi_p3f4_certification_confirmatory_freeze.py') (Join-Path $project 'tests\test_pcpi_p3f4_cert2_response_energy.py') (Join-Path $project 'tests\test_pcpi_p3f4_cert2_development_protocol.py'); if ($LASTEXITCODE) { throw 'CERT.2 tests failed' }; & $python -m scripts.run_pcpi_p3f4_cert2_response_energy_development --config (Join-Path $project 'configs\p3f_4_response_energy_certification_development.json') --output $output; if ($LASTEXITCODE) { throw 'CERT.2 development Gate returned NO-GO' }
```

This command performs correctness and postmortem development checks only.  It
does not access `D:\01\666\data`, run an acquisition policy, open held-out, or
produce formal efficacy evidence.

## Downstream boundary

A clean PASS makes a new response-free confirmatory-freeze **review** eligible.
It does not itself authorize response materialization.  Before any resident
integration, the hybrid/raw representation blocker in
`pcpi_p3f4_cert2_proof_review_closure_20260819.md` must be resolved.
