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
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
$projectPath = [System.IO.Path]::GetFullPath($Project).TrimEnd('\')
$pythonPath = [System.IO.Path]::GetFullPath($Python)
$dataRootPath = [System.IO.Path]::GetFullPath($DataRoot).TrimEnd('\')
$configPath = [System.IO.Path]::GetFullPath($Config)
$outputPath = [System.IO.Path]::GetFullPath($Output).TrimEnd('\')
$stashPath = [System.IO.Path]::GetFullPath($EvidenceStash).TrimEnd('\')
$projectParent = [System.IO.Directory]::GetParent($projectPath).FullName.TrimEnd('\')
$outputsRoot = (Join-Path $projectPath 'outputs').TrimEnd('\')
$evidencePath = Join-Path $projectPath 'evidence'
$runner = Join-Path $projectPath 'scripts\run_pcpi_p3j15_formal_real_acquisition.py'
$stdoutLog = $outputPath + '.stdout.log'
$stderrLog = $outputPath + '.stderr.log'
$progressPath = Join-Path $outputPath 'logs\run.jsonl'
$terminalFailurePath = Join-Path $outputPath 'TERMINAL_FAILURE.json'
$process = $null
$hadEvidence = $false

function Quote-Literal {
    param([Parameter(Mandatory = $true)][string]$Value)
    return "'" + $Value.Replace("'", "''") + "'"
}

foreach ($required in @($projectPath, $dataRootPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Container)) {
        throw "Required directory does not exist: $required"
    }
}
foreach ($required in @($pythonPath, $configPath, $runner)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file does not exist: $required"
    }
}
if ($outputPath -eq $outputsRoot -or -not $outputPath.StartsWith(
    $outputsRoot + '\', [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw "Output must be one named directory below outputs/: $outputPath"
}
if (
    [System.IO.Directory]::GetParent($stashPath).FullName.TrimEnd('\') -ne
    $projectParent
) {
    throw "Evidence stash must be a project sibling: $stashPath"
}
if (Test-Path -LiteralPath $stashPath) {
    throw "Evidence stash already exists: $stashPath"
}
if ($Resume) {
    if (-not (Test-Path -LiteralPath $outputPath -PathType Container)) {
        throw "Resume output does not exist: $outputPath"
    }
    if (
        (Test-Path -LiteralPath (Join-Path $outputPath 'summary.json')) -or
        (Test-Path -LiteralPath (Join-Path $outputPath 'RUN_MANIFEST.json')) -or
        (Test-Path -LiteralPath $terminalFailurePath) -or
        (Get-ChildItem -LiteralPath $outputPath -Filter 'POLICY_FAILURE.json' -Recurse -File -ErrorAction SilentlyContinue)
    ) {
        throw 'Resume is forbidden after terminal success or policy failure'
    }
} else {
    foreach ($fresh in @($outputPath, $stdoutLog, $stderrLog)) {
        if (Test-Path -LiteralPath $fresh) {
            throw "Unique formal path already exists: $fresh"
        }
    }
}

$status = @(& git -C $projectPath status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect Git status' }
$unexpected = @($status | Where-Object { $_ -notmatch '^\?\? evidence/' })
if ($unexpected.Count -ne 0) {
    throw "Changes exist outside evidence/:`n$($unexpected -join "`n")"
}
$hadEvidence = Test-Path -LiteralPath $evidencePath -PathType Container

try {
    if ($hadEvidence) {
        Move-Item -LiteralPath $evidencePath -Destination $stashPath
    }
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
        Write-Host 'P3J.15 preflight passed; no data process or output was created.'
        return
    }

    New-Item -ItemType Directory -Path $outputsRoot -Force | Out-Null
    $child = (
        '& {0} -B {1} --data-root {2} --output-dir {3} --config {4} ' +
        '--phase P3J.15 --heldout-state closed 1>> {5} 2>> {6}; exit $LASTEXITCODE'
    ) -f @(
        (Quote-Literal $pythonPath), (Quote-Literal $runner),
        (Quote-Literal $dataRootPath), (Quote-Literal $outputPath),
        (Quote-Literal $configPath), (Quote-Literal $stdoutLog),
        (Quote-Literal $stderrLog)
    )
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($child))
    $process = Start-Process -FilePath (Join-Path $PSHOME 'powershell.exe') `
        -ArgumentList @('-NoProfile', '-NonInteractive', '-EncodedCommand', $encoded) `
        -WindowStyle Hidden -PassThru
    Write-Host "P3J.15 started: PID=$($process.Id) output=$outputPath resume=$Resume"

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
        $checkpoint = Get-ChildItem -LiteralPath $outputPath -Filter 'PROGRESS.json' `
            -Recurse -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
        if ($null -ne $checkpoint) {
            $raw = Get-Content -LiteralPath $checkpoint.FullName -Raw
            if ($raw -and $raw -ne $lastCheckpoint) {
                $item = $raw | ConvertFrom-Json
                Write-Host ("P3J checkpoint: models={0}/4 nodes={1} identity={2}" -f `
                    $item.completed_models, $item.nodes_per_leaf, $item.identity_hash)
                $lastCheckpoint = $raw
            }
        }
        if (Test-Path -LiteralPath $terminalFailurePath) {
            throw "Formal terminal failure exists: $terminalFailurePath"
        }
        Start-Sleep -Seconds $HeartbeatSeconds
        $process.Refresh()
    }
    $process.WaitForExit()
    $process.Refresh()
    $observedExitCode = $process.ExitCode
    if ($null -ne $observedExitCode -and [int]$observedExitCode -ne 0) {
        if (Test-Path -LiteralPath $stderrLog) {
            Get-Content -LiteralPath $stderrLog -Tail 60 | Write-Host
        }
        throw "P3J.15 process failed with exit code $observedExitCode"
    }
    $summaryPath = Join-Path $outputPath 'summary.json'
    $manifestPath = Join-Path $outputPath 'RUN_MANIFEST.json'
    foreach ($terminal in @($summaryPath, $manifestPath)) {
        if (-not (Test-Path -LiteralPath $terminal -PathType Leaf)) {
            throw "Terminal artifact missing: $terminal"
        }
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if (
        -not $summary.protocol_gate_passed -or
        $summary.successful_runs -ne $summary.expected_runs -or
        $summary.failure_count -ne 0 -or $summary.heldout_opened
    ) { throw 'P3J.15 summary completeness or held-out check failed' }
    if (
        $manifest.stage -ne 'P3J.15' -or -not $manifest.protocol_gate_passed -or
        -not $manifest.evidence_registry.valid -or
        $manifest.source_git_commit -ne $ExpectedCommit.ToLowerInvariant() -or
        $manifest.source_git_tree -ne $ExpectedTree.ToLowerInvariant() -or
        $manifest.config_file_hash -ne $ExpectedConfigHash.ToLowerInvariant() -or
        $manifest.python_executable_hash -ne $ExpectedPythonHash.ToLowerInvariant() -or
        $manifest.heldout_opened
    ) { throw 'P3J.15 manifest identity or evidence check failed' }
    Write-Host ("P3J.15 complete: assessment={0} runs={1}/{2}" -f `
        $summary.effectiveness_assessment.status,
        $summary.successful_runs, $summary.expected_runs)
}
finally {
    if ($null -ne $process) {
        $process.Refresh()
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force
            $process.WaitForExit()
        }
    }
    if ($hadEvidence -and (Test-Path -LiteralPath $stashPath)) {
        if (Test-Path -LiteralPath $evidencePath) {
            throw "Cannot restore evidence; safe copy remains at: $stashPath"
        }
        Move-Item -LiteralPath $stashPath -Destination $evidencePath
    }
}
