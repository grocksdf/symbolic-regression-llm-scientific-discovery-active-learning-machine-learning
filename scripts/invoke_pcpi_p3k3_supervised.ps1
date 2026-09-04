[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Project,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$Output,
    [Parameter(Mandatory = $true)][string]$EvidenceStash,
    [Parameter(Mandatory = $true)][string]$ExpectedBranch,
    [Parameter(Mandatory = $true)][string]$ExpectedCommit,
    [Parameter(Mandatory = $true)][string]$ExpectedTree,
    [Parameter(Mandatory = $true)][string]$ExpectedConfigHash,
    [Parameter(Mandatory = $true)][string]$ExpectedPythonHash,
    [ValidateRange(2, 60)][int]$HeartbeatSeconds = 5,
    [switch]$Resume,
    [switch]$PreflightOnly,
    [string]$ProtocolStage = 'P3K.3',
    [string]$RunnerRelativePath = 'scripts\run_pcpi_p3k3_formal_real_acquisition.py'
)

$ErrorActionPreference = 'Stop'
$projectPath = [IO.Path]::GetFullPath($Project).TrimEnd('\')
$pythonPath = [IO.Path]::GetFullPath($Python)
$dataRootPath = [IO.Path]::GetFullPath($DataRoot).TrimEnd('\')
$configPath = [IO.Path]::GetFullPath($Config)
$outputPath = [IO.Path]::GetFullPath($Output).TrimEnd('\')
$stashPath = [IO.Path]::GetFullPath($EvidenceStash).TrimEnd('\')
$outputsRoot = (Join-Path $projectPath 'outputs').TrimEnd('\')
$evidencePath = Join-Path $projectPath 'evidence'
$runner = Join-Path $projectPath $RunnerRelativePath
$stdoutLog = $outputPath + '.stdout.log'
$stderrLog = $outputPath + '.stderr.log'
$progressPath = Join-Path $outputPath 'logs\run.jsonl'
$process = $null
$hadEvidence = $false

function Quote-Literal([string]$Value) {
    return "'" + $Value.Replace("'", "''") + "'"
}

foreach ($directory in @($projectPath, $dataRootPath)) {
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        throw "Required directory does not exist: $directory"
    }
}
foreach ($file in @($pythonPath, $configPath, $runner)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Required file does not exist: $file"
    }
}
if ($outputPath -eq $outputsRoot -or -not $outputPath.StartsWith(
    $outputsRoot + '\', [StringComparison]::OrdinalIgnoreCase
)) { throw "Output must be one named directory below outputs/: $outputPath" }
$projectParent = [IO.Directory]::GetParent($projectPath).FullName.TrimEnd('\')
if ([IO.Directory]::GetParent($stashPath).FullName.TrimEnd('\') -ne $projectParent) {
    throw "Evidence stash must be a project sibling: $stashPath"
}
if (Test-Path -LiteralPath $stashPath) { throw "Evidence stash already exists: $stashPath" }

if ($Resume) {
    if (-not (Test-Path -LiteralPath $outputPath -PathType Container)) {
        throw "Resume output does not exist: $outputPath"
    }
    if (
        (Test-Path -LiteralPath (Join-Path $outputPath 'summary.json')) -or
        (Test-Path -LiteralPath (Join-Path $outputPath 'RUN_MANIFEST.json')) -or
        (Test-Path -LiteralPath (Join-Path $outputPath 'TERMINAL_FAILURE.json')) -or
        (Get-ChildItem -LiteralPath $outputPath -Filter 'POLICY_FAILURE.json' -Recurse -File -ErrorAction SilentlyContinue)
    ) { throw 'Resume is forbidden after a terminal success or failure' }
} else {
    foreach ($fresh in @($outputPath, $stdoutLog, $stderrLog)) {
        if (Test-Path -LiteralPath $fresh) { throw "Unique formal path already exists: $fresh" }
    }
}

$status = @(& git -C $projectPath status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect Git status' }
$unexpected = @($status | Where-Object { $_ -notmatch '^\?\? evidence/' })
if ($unexpected.Count -ne 0) { throw "Changes exist outside evidence/:`n$($unexpected -join "`n")" }
$hadEvidence = Test-Path -LiteralPath $evidencePath -PathType Container

try {
    if ($hadEvidence) { Move-Item -LiteralPath $evidencePath -Destination $stashPath }
    $remaining = @(& git -C $projectPath status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE -ne 0 -or $remaining.Count -ne 0) {
        throw "Worktree is not clean after evidence isolation:`n$($remaining -join "`n")"
    }
    $actualBranch = (& git -C $projectPath branch --show-current).Trim()
    $actualCommit = (& git -C $projectPath rev-parse HEAD).Trim().ToLowerInvariant()
    $actualTree = (& git -C $projectPath rev-parse 'HEAD^{tree}').Trim().ToLowerInvariant()
    $actualConfigHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $configPath).Hash.ToLowerInvariant()
    $actualPythonHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $pythonPath).Hash.ToLowerInvariant()
    if ($actualBranch -ne $ExpectedBranch) { throw "Branch mismatch: $actualBranch" }
    if ($actualCommit -ne $ExpectedCommit.ToLowerInvariant()) { throw "Commit mismatch: $actualCommit" }
    if ($actualTree -ne $ExpectedTree.ToLowerInvariant()) { throw "Tree mismatch: $actualTree" }
    if ($actualConfigHash -ne $ExpectedConfigHash.ToLowerInvariant()) { throw "Config mismatch: $actualConfigHash" }
    if ($actualPythonHash -ne $ExpectedPythonHash.ToLowerInvariant()) { throw "Python mismatch: $actualPythonHash" }
    if ($PreflightOnly) {
        Write-Host "$ProtocolStage preflight passed; no data process or output was created."
        return
    }

    $child = ('& {0} -B {1} --data-root {2} --output-dir {3} --config {4} ' +
        "--phase $ProtocolStage --heldout-state closed 1>> {5} 2>> {6}; exit `$LASTEXITCODE") -f @(
        (Quote-Literal $pythonPath), (Quote-Literal $runner),
        (Quote-Literal $dataRootPath), (Quote-Literal $outputPath),
        (Quote-Literal $configPath), (Quote-Literal $stdoutLog),
        (Quote-Literal $stderrLog)
    )
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($child))
    $process = Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') `
        -ArgumentList @('-NoProfile', '-NonInteractive', '-EncodedCommand', $encoded) `
        -WindowStyle Hidden -PassThru
    Write-Host "$ProtocolStage started: PID=$($process.Id) output=$outputPath resume=$Resume"

    $lastProgress = ''
    $lastCheckpoint = ''
    while (-not $process.HasExited) {
        if (Test-Path -LiteralPath $progressPath -PathType Leaf) {
            $latest = Get-Content -LiteralPath $progressPath -Tail 1
            if ($latest -and $latest -ne $lastProgress) {
                $event = $latest | ConvertFrom-Json
                Write-Host ("[{0,9:N1}s] {1}" -f $event.elapsed_seconds, $event.message)
                $lastProgress = $latest
            }
        }
        $checkpoint = Get-ChildItem -LiteralPath $outputPath -Filter 'PROGRESS.json' -Recurse -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
        if ($null -ne $checkpoint) {
            $raw = Get-Content -LiteralPath $checkpoint.FullName -Raw
            if ($raw -and $raw -ne $lastCheckpoint) {
                $item = $raw | ConvertFrom-Json
                Write-Host ("P3K checkpoint: models={0}/4 nodes={1} identity={2}" -f $item.completed_models, $item.nodes_per_leaf, $item.identity_hash)
                $lastCheckpoint = $raw
            }
        }
        Start-Sleep -Seconds $HeartbeatSeconds
        $process.Refresh()
    }
    $process.WaitForExit()
    $process.Refresh()
    if ($null -eq $process.ExitCode) { throw "$ProtocolStage child exit code is unavailable" }
    if ([int]$process.ExitCode -ne 0) {
        if (Test-Path -LiteralPath $stderrLog) { Get-Content -LiteralPath $stderrLog -Tail 80 | Write-Host }
        throw "$ProtocolStage process failed with exit code $($process.ExitCode)"
    }
    $summaryPath = Join-Path $outputPath 'summary.json'
    $manifestPath = Join-Path $outputPath 'RUN_MANIFEST.json'
    foreach ($terminal in @($summaryPath, $manifestPath)) {
        if (-not (Test-Path -LiteralPath $terminal -PathType Leaf)) { throw "Terminal artifact missing: $terminal" }
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if (-not $summary.protocol_gate_passed -or $summary.successful_runs -ne $summary.expected_runs -or $summary.failure_count -ne 0 -or $summary.heldout_opened) {
        throw "$ProtocolStage summary completeness or held-out check failed"
    }
    if ($manifest.stage -ne $ProtocolStage -or -not $manifest.protocol_gate_passed -or -not $manifest.evidence_registry.valid -or
        $manifest.source_git_commit -ne $ExpectedCommit.ToLowerInvariant() -or $manifest.source_git_tree -ne $ExpectedTree.ToLowerInvariant() -or
        $manifest.config_file_hash -ne $ExpectedConfigHash.ToLowerInvariant() -or $manifest.python_executable_hash -ne $ExpectedPythonHash.ToLowerInvariant() -or
        $manifest.heldout_opened) { throw "$ProtocolStage manifest identity or evidence check failed" }
    Write-Host ("$ProtocolStage complete: assessment={0} runs={1}/{2}" -f $summary.effectiveness_assessment.status, $summary.successful_runs, $summary.expected_runs)
}
finally {
    if ($null -ne $process) {
        $process.Refresh()
        if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force; $process.WaitForExit() }
    }
    if ($hadEvidence -and (Test-Path -LiteralPath $stashPath)) {
        if (Test-Path -LiteralPath $evidencePath) { throw "Cannot restore evidence; safe copy remains at: $stashPath" }
        Move-Item -LiteralPath $stashPath -Destination $evidencePath
    }
}
