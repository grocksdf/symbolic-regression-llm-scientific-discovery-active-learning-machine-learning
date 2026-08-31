[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Project,
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(Mandatory = $true)][string]$ExpectedBranch,
    [Parameter(Mandatory = $true)][string]$ExpectedCommit,
    [Parameter(Mandatory = $true)][string]$ExpectedTree,
    [Parameter(Mandatory = $true)][string]$ExpectedConfigHash,
    [Parameter(Mandatory = $true)][string]$ExpectedPythonHash,
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
$projectPath = [System.IO.Path]::GetFullPath($Project).TrimEnd('\')
$pythonPath = [System.IO.Path]::GetFullPath($Python)
$configPath = [System.IO.Path]::GetFullPath($Config)

if (-not (Test-Path -LiteralPath $projectPath -PathType Container)) {
    throw "Project directory does not exist: $projectPath"
}
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Frozen Python does not exist: $pythonPath"
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "P3J.8 config does not exist: $configPath"
}

$status = @(& git -C $projectPath status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0) {
    throw 'Cannot inspect Git worktree status'
}
$unexpected = @($status | Where-Object { $_ -notmatch '^\?\? evidence/' })
if ($unexpected.Count -ne 0) {
    throw "Worktree changes exist outside evidence/:`n$($unexpected -join "`n")"
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
if (-not $PreflightOnly) {
    throw 'P3J.8 formal execution remains blocked pending P3J.9 runner composition'
}
Write-Host 'P3J.8 supervised identity preflight passed; no output or real process created.'
