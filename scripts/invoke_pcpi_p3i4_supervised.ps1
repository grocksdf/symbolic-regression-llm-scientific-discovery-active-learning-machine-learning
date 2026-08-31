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
$runner = Join-Path $projectPath 'scripts\run_pcpi_p3i4_shared_h0_real_acquisition.py'
$stdoutLog = $outputPath + '.stdout.log'
$stderrLog = $outputPath + '.stderr.log'
$progressPath = Join-Path $outputPath 'logs\run.jsonl'
$terminalFailurePath = Join-Path $outputPath 'TERMINAL_FAILURE.json'
$process = $null
$hadEvidence = $false

function ConvertTo-SingleQuotedLiteral {
    param([Parameter(Mandatory = $true)][string]$Value)
    return "'" + $Value.Replace("'", "''") + "'"
}

if (-not (Test-Path -LiteralPath $projectPath -PathType Container)) {
    throw "Project directory does not exist: $projectPath"
}
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Frozen Python does not exist: $pythonPath"
}
if (-not (Test-Path -LiteralPath $dataRootPath -PathType Container)) {
    throw "Real-data directory does not exist: $dataRootPath"
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "P3I.4 config does not exist: $configPath"
}
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "P3I.4 runner does not exist: $runner"
}
if (
    $outputPath -ne $outputsRoot -and
    -not $outputPath.StartsWith($outputsRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)
) {
    throw "Formal output must be inside project outputs/: $outputPath"
}
if ($outputPath -eq $outputsRoot) {
    throw 'Formal output cannot be the outputs/ root itself'
}
if (
    [System.IO.Directory]::GetParent($stashPath).FullName.TrimEnd('\') -ne
    $projectParent
) {
    throw "Evidence stash must be an explicit project sibling: $stashPath"
}
if ($stashPath.StartsWith($projectPath + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Evidence stash cannot be inside the project: $stashPath"
}
foreach ($freshPath in @($outputPath, $stdoutLog, $stderrLog, $stashPath)) {
    if (Test-Path -LiteralPath $freshPath) {
        throw "Unique run path already exists; rerun/overwrite forbidden: $freshPath"
    }
}

$status = @(& git -C $projectPath status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0) {
    throw 'Cannot inspect Git worktree status'
}
$unexpected = @($status | Where-Object { $_ -notmatch '^\?\? evidence/' })
if ($unexpected.Count -ne 0) {
    throw "Worktree changes exist outside evidence/:`n$($unexpected -join "`n")"
}
$hadEvidence = Test-Path -LiteralPath $evidencePath -PathType Container

try {
    if ($hadEvidence) {
        Move-Item -LiteralPath $evidencePath -Destination $stashPath
    }
    $remaining = @(& git -C $projectPath status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE -ne 0 -or $remaining.Count -ne 0) {
        throw "Worktree remains dirty after evidence isolation:`n$($remaining -join "`n")"
    }

    $actualBranch = (& git -C $projectPath branch --show-current).Trim()
    $actualCommit = (& git -C $projectPath rev-parse HEAD).Trim().ToLowerInvariant()
    $actualTree = (& git -C $projectPath rev-parse 'HEAD^{tree}').Trim().ToLowerInvariant()
    $actualConfigHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $configPath
    ).Hash.ToLowerInvariant()
    $actualPythonHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $pythonPath
    ).Hash.ToLowerInvariant()
    if ($actualBranch -ne $ExpectedBranch) {
        throw "Branch mismatch: $actualBranch"
    }
    if ($actualCommit -ne $ExpectedCommit.ToLowerInvariant()) {
        throw "Commit mismatch: $actualCommit"
    }
    if ($actualTree -ne $ExpectedTree.ToLowerInvariant()) {
        throw "Source tree mismatch: $actualTree"
    }
    if ($actualConfigHash -ne $ExpectedConfigHash.ToLowerInvariant()) {
        throw "Config hash mismatch: $actualConfigHash"
    }
    if ($actualPythonHash -ne $ExpectedPythonHash.ToLowerInvariant()) {
        throw "Python executable hash mismatch: $actualPythonHash"
    }
    if ($PreflightOnly) {
        Write-Host 'P3I.4 supervised preflight passed; no output or real process created.'
        return
    }

    New-Item -ItemType Directory -Path $outputsRoot -Force | Out-Null
    $childCommand = (
        '& {0} -B {1} --data-root {2} --output-dir {3} --config {4} ' +
        '--phase P3I.4 --heldout-state closed 1> {5} 2> {6}; exit $LASTEXITCODE'
    ) -f @(
        (ConvertTo-SingleQuotedLiteral $pythonPath),
        (ConvertTo-SingleQuotedLiteral $runner),
        (ConvertTo-SingleQuotedLiteral $dataRootPath),
        (ConvertTo-SingleQuotedLiteral $outputPath),
        (ConvertTo-SingleQuotedLiteral $configPath),
        (ConvertTo-SingleQuotedLiteral $stdoutLog),
        (ConvertTo-SingleQuotedLiteral $stderrLog)
    )
    $encodedCommand = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($childCommand)
    )
    $hostPowerShell = Join-Path $PSHOME 'powershell.exe'
    $process = Start-Process `
        -FilePath $hostPowerShell `
        -ArgumentList @('-NoProfile', '-NonInteractive', '-EncodedCommand', $encodedCommand) `
        -WindowStyle Hidden `
        -PassThru
    Write-Host "P3I.4 started: PID=$($process.Id) output=$outputPath"

    $lastProgress = ''
    $terminalShown = $false
    while (-not $process.HasExited) {
        if (Test-Path -LiteralPath $progressPath -PathType Leaf) {
            $latest = Get-Content -LiteralPath $progressPath -Tail 1
            if ($latest -and $latest -ne $lastProgress) {
                $event = $latest | ConvertFrom-Json
                Write-Host ("[{0,9:N1}s] {1}" -f $event.elapsed_seconds, $event.message)
                $lastProgress = $latest
            }
        }
        if (
            -not $terminalShown -and
            (Test-Path -LiteralPath $terminalFailurePath -PathType Leaf)
        ) {
            $terminal = Get-Content -LiteralPath $terminalFailurePath -Raw |
                ConvertFrom-Json
            Write-Error ("P3I.4 stopped at first failure: {0}" -f $terminal.failure.failure_status)
            $terminalShown = $true
        }
        Start-Sleep -Seconds $HeartbeatSeconds
        $process.Refresh()
    }
    $process.WaitForExit()
    $process.Refresh()
    $exitCode = $process.ExitCode
    if ($null -eq $exitCode) {
        throw 'P3I.4 supervisor could not read the formal process exit code'
    }
    if ([int]$exitCode -ne 0) {
        if (Test-Path -LiteralPath $stderrLog -PathType Leaf) {
            Get-Content -LiteralPath $stderrLog -Tail 40 | Write-Host
        }
        throw "P3I.4 formal process failed with exit code $([int]$exitCode)"
    }
    if (Test-Path -LiteralPath $terminalFailurePath) {
        throw "Process returned success but terminal failure exists: $terminalFailurePath"
    }

    $summaryPath = Join-Path $outputPath 'summary.json'
    $manifestPath = Join-Path $outputPath 'RUN_MANIFEST.json'
    foreach ($terminalPath in @($summaryPath, $manifestPath)) {
        if (-not (Test-Path -LiteralPath $terminalPath -PathType Leaf)) {
            throw "Formal terminal artifact is missing: $terminalPath"
        }
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if (
        -not $summary.protocol_gate_passed -or
        $summary.successful_runs -ne $summary.expected_runs -or
        $summary.failure_count -ne 0 -or
        $summary.heldout_opened -or
        $summary.selection_used_heldout
    ) {
        throw 'P3I.4 summary failed completeness, failure, or held-out checks'
    }
    if (
        $manifest.stage -ne 'P3I.4' -or
        -not $manifest.protocol_gate_passed -or
        -not $manifest.evidence_registry.valid -or
        $manifest.source_git_commit -ne $ExpectedCommit.ToLowerInvariant() -or
        $manifest.source_git_tree -ne $ExpectedTree.ToLowerInvariant() -or
        $manifest.config_file_hash -ne $ExpectedConfigHash.ToLowerInvariant() -or
        $manifest.python_executable_hash -ne $ExpectedPythonHash.ToLowerInvariant() -or
        $manifest.heldout_opened -or
        $manifest.selection_used_heldout
    ) {
        throw 'P3I.4 manifest source/config/evidence/held-out identity mismatch'
    }
    Write-Host (
        "P3I.4 complete: protocol=PASS assessment={0} runs={1}/{2}" -f
        $summary.effectiveness_assessment.status,
        $summary.successful_runs,
        $summary.expected_runs
    )
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
            throw "Cannot restore historical evidence/; safe copy retained at: $stashPath"
        }
        Move-Item -LiteralPath $stashPath -Destination $evidencePath
    }
}
