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

$common = Join-Path $PSScriptRoot 'invoke_pcpi_p3k3_supervised.ps1'
& $common @PSBoundParameters `
    -ProtocolStage 'P3K.7' `
    -RunnerRelativePath 'scripts\run_pcpi_p3k7_formal_real_acquisition.py'
