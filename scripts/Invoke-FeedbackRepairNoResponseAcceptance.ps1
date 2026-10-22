<#
Response-free intended R-v2 acceptance. NO --execute, NO 72-child launch.
Ensure full historic test suites are run under explicit provider/network deny
controls; this script cannot prove external calls were absent from those suites.
Both repositories must be CLEAN NEW commits, not the old delivered HEADs.
The emitter fixture must come from the real child writer's synthetic/no-provider
path. Check writer provenance manually; the fixture verifier cannot prove it.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$Python,
  [Parameter(Mandatory=$true)][string]$MainRepo,
  [Parameter(Mandatory=$true)][string]$BenchRepo,
  [Parameter(Mandatory=$true)][string]$GateReference,
  [Parameter(Mandatory=$true)][string]$Hdf5,
  [Parameter(Mandatory=$true)][string]$Metadata,
  [Parameter(Mandatory=$true)][string]$ProviderEnv,
  [Parameter(Mandatory=$true)][string]$InputSymbols,
  [Parameter(Mandatory=$true)][string]$EmitterFixtureDir,
  [Parameter(Mandatory=$true)][string]$EmitterFamily,
  [Parameter(Mandatory=$true)][string]$EmitterTask,
  [Parameter(Mandatory=$true)][int]$EmitterSeed,
  [Parameter(Mandatory=$true)][ValidateSet('C','D','R')][string]$EmitterArm,
  [Parameter(Mandatory=$true)][string]$EmitterCondition,
  [Parameter(Mandatory=$true)][string]$OutputRoot
)
$ErrorActionPreference = 'Stop'
function MustRun([string]$label, [scriptblock]$command) {
  & $command
  if ($LASTEXITCODE -ne 0) { throw "$label FAILED (exit=$LASTEXITCODE): PRE-RUN_NO-GO" }
}
function Require-CleanNew([string]$repo, [string]$oldHead) {
  $current = (& git -C $repo rev-parse HEAD).Trim()
  if ($LASTEXITCODE -ne 0) { throw 'Not a git repo: ' + $repo }
  if ($current -eq $oldHead) { throw 'Repository still has old HEAD; a new clean commit is required' }
  $dirty = @(& git -C $repo status --porcelain)
  if ($LASTEXITCODE -ne 0 -or $dirty.Count -gt 0) {
    throw "Repository must be clean before registration: $repo"
  }
  return $current
}
$OutputRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot)
if (Test-Path -LiteralPath $OutputRoot) {
  throw 'Use a NEW output directory for one immutable acceptance run'
}
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$MainRepo = (Resolve-Path -LiteralPath $MainRepo).Path
$BenchRepo = (Resolve-Path -LiteralPath $BenchRepo).Path
$env:HYPOTHESIS_MVP_ROOT = $MainRepo
$env:PYTHONPATH = "$MainRepo;$BenchRepo"
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTEST_ADDOPTS = '-p no:cacheprovider'
$mainHead = Require-CleanNew $MainRepo '45c246623f21b0b68a6e34a6c78e29aced55e32c'
$benchHead = Require-CleanNew $BenchRepo 'b408fdc87b2483a273f84cfbbf86e1a0adcf4a60'
if (-not (Test-Path -LiteralPath (Join-Path $BenchRepo 'scripts/run_aistats_three_arm_formula_child.py'))) {
  throw 'Actual child emitter not found: PRE-RUN_NO-GO'
}

# 1. Full historical suites. Not a selected subset and not silent skipping.
Push-Location $MainRepo
try {
  & $Python -m pytest -q tests 2>&1 | Tee-Object -FilePath (Join-Path $OutputRoot 'MAIN_FULL_PYTEST.log')
  if ($LASTEXITCODE -ne 0) { throw 'Main full regression failed: PRE-RUN_NO-GO' }
} finally { Pop-Location }
Push-Location $BenchRepo
try {
  & $Python -m pytest -q tests 2>&1 | Tee-Object -FilePath (Join-Path $OutputRoot 'BENCH_FULL_PYTEST.log')
  if ($LASTEXITCODE -ne 0) { throw 'Benchmark full regression failed: PRE-RUN_NO-GO' }
} finally { Pop-Location }
$mainHead2 = Require-CleanNew $MainRepo '45c246623f21b0b68a6e34a6c78e29aced55e32c'
$benchHead2 = Require-CleanNew $BenchRepo 'b408fdc87b2483a273f84cfbbf86e1a0adcf4a60'
if ($mainHead2 -ne $mainHead -or $benchHead2 -ne $benchHead) {
  throw 'Repository commit changed during regression: PRE-RUN_NO-GO'
}

# 2. The real emitter must have produced this fixture WITHOUT providers.
Push-Location $BenchRepo
try {
  MustRun 'Real emitter fixture contract' {
    & $Python scripts/verify_feedback_repair_child_emitter.py `
      --child-dir $EmitterFixtureDir --family $EmitterFamily --task $EmitterTask `
      --seed $EmitterSeed --arm $EmitterArm --condition $EmitterCondition `
      --output (Join-Path $OutputRoot 'EMITTER_CONTRACT.json')
  }
  # 3. Same *registered interpreter* and production scheduler isolation.
  MustRun 'Isolated Windows PySR + polynomial Gate' {
    & $Python scripts/run_feedback_repair_runtime_gate.py `
      --reference $GateReference --output (Join-Path $OutputRoot 'RUNTIME_GATE.json')
  }
  # 4. Read safe public Parquet metadata columns, hash pinned inputs, register.
  MustRun 'Frozen registration' {
    & $Python scripts/build_feedback_repair_registration.py `
      --runtime-gate (Join-Path $OutputRoot 'RUNTIME_GATE.json') `
      --mainline-root $MainRepo --hdf5 $Hdf5 --metadata $Metadata `
      --provider-env $ProviderEnv --input-symbols $InputSymbols `
      --output (Join-Path $OutputRoot 'REGISTRATION.json')
  }
  # 5. Plan-only: never pass --execute or authorized execution token.
  $planText = & $Python scripts/run_feedback_repair.py `
    --registration (Join-Path $OutputRoot 'REGISTRATION.json') `
    --output (Join-Path $OutputRoot 'MATRIX_NOT_CREATED')
  if ($LASTEXITCODE -ne 0) { throw 'Dry-run plan failed: PRE-RUN_NO-GO' }
  $planText | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $OutputRoot 'PLAN_72.json')
  $plan = ($planText | Out-String | ConvertFrom-Json)
  if ($plan.execution -ne $false -or $plan.children.Count -ne 72) {
    throw 'Dry-run is not exactly 72 non-executed children: PRE-RUN_NO-GO'
  }
  $coordinates = @($plan.children | ForEach-Object {
    '{0}/{1}/{2}/{3}' -f $_.family,$_.task,$_.seed,$_.arm
  } | Sort-Object -Unique)
  if ($coordinates.Count -ne 72) { throw 'Duplicate planned child: PRE-RUN_NO-GO' }
} finally { Pop-Location }
$mainHead3 = Require-CleanNew $MainRepo '45c246623f21b0b68a6e34a6c78e29aced55e32c'
$benchHead3 = Require-CleanNew $BenchRepo 'b408fdc87b2483a273f84cfbbf86e1a0adcf4a60'
if ($mainHead3 -ne $mainHead -or $benchHead3 -ne $benchHead) {
  throw 'Repository commit changed after registration: PRE-RUN_NO-GO'
}
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $OutputRoot 'REGISTRATION.json')).Hash.ToLowerInvariant()
$summary = [ordered]@{
  schema = 'feedback-repair-response-free-acceptance-v1'
  status = 'OFFLINE_ACCEPTANCE_COMPLETE_PENDING_MANUAL_EMITTER_PROVENANCE_REVIEW'
  main_commit = $mainHead
  benchmark_commit = $benchHead
  registration_sha256 = $hash
  children = 72
  matrix_executed = $false
  provider_calls_during_full_pytest = 'NOT INDEPENDENTLY MEASURED; run under network/provider deny controls'
  full_main_regression = 'passed'
  full_benchmark_regression = 'passed'
  real_emitter_fixture_contract = 'passed; independent writer source review still required'
  windows_isolated_engine_gate = 'passed'
}
$summary | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -LiteralPath (Join-Path $OutputRoot 'ACCEPTANCE_SUMMARY.json')
Write-Host "Response-free acceptance checks completed; manual emitter source review remains."
Write-Host "Fresh registration SHA-256: $hash"
Write-Host 'NO MATRIX WAS EXECUTED.'
