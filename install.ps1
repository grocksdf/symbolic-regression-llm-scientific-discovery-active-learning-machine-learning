param(
    [Parameter(Mandatory = $true)]
    [string]$WorkspaceRoot
)

$ErrorActionPreference = "Stop"
$packageRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$workspace = [System.IO.Path]::GetFullPath($WorkspaceRoot)
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $workspace "hypothesis_mvp"))
$python = (Get-Command "python" -ErrorAction Stop).Source
$installStage = [System.IO.Path]::GetFullPath((Join-Path ([System.IO.Path]::GetTempPath()) ("PCPIInstall-" + [guid]::NewGuid().ToString("N"))))
$backupRoot = Join-Path $installStage "rollback"
$managedDirectories = @(
    "hypothesis_mvp", "scripts", "tests", "docs", "config", "configs",
    "contracts", "schemas", "paper"
)
$managedFiles = @(
    "README.md", "pyproject.toml", "requirements.txt", "requirements-dev.txt",
    "pytest.ini", "LICENSE", "run_pipeline.py", "install.ps1",
    "DELIVERY_MANIFEST.json"
)
$dataPaths = @(
    "data\README.md", "data\manifests", "data\split_manifests", "data\schemas"
)
$installed = New-Object System.Collections.Generic.List[string]

function Assert-SafeChildPath {
    param([string]$Candidate, [string]$Parent)
    $candidateFull = [System.IO.Path]::GetFullPath($Candidate)
    $parentPrefix = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\') + '\'
    if (-not $candidateFull.StartsWith($parentPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe path outside project root: $candidateFull"
    }
}

function Install-ManagedPath {
    param([string]$RelativePath)
    $source = [System.IO.Path]::GetFullPath((Join-Path $packageRoot $RelativePath))
    $destination = [System.IO.Path]::GetFullPath((Join-Path $projectRoot $RelativePath))
    $backup = [System.IO.Path]::GetFullPath((Join-Path $backupRoot $RelativePath))
    Assert-SafeChildPath -Candidate $destination -Parent $projectRoot
    if (-not (Test-Path -LiteralPath $source)) { throw "Package path missing: $RelativePath" }
    if (Test-Path -LiteralPath $destination) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $backup) -Force | Out-Null
        Move-Item -LiteralPath $destination -Destination $backup
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Recurse -Force
    $installed.Add($RelativePath)
}

if (-not (Test-Path -LiteralPath $projectRoot -PathType Container)) {
    throw "Project directory does not exist: $projectRoot"
}

try {
    & $python -B (Join-Path $packageRoot "scripts\syntax_preflight.py") --root $packageRoot
    if ($LASTEXITCODE -ne 0) { throw "Package syntax preflight failed: $LASTEXITCODE" }
    Push-Location $packageRoot
    try {
        & $python -B -m scripts.verify_delivery --root $packageRoot
        if ($LASTEXITCODE -ne 0) { throw "Package delivery manifest failed: $LASTEXITCODE" }
    }
    finally { Pop-Location }
    Push-Location $packageRoot
    try {
        & $python -B "scripts\audit_final_source.py"
        if ($LASTEXITCODE -ne 0) { throw "Package integrity audit failed: $LASTEXITCODE" }
    }
    finally { Pop-Location }

    New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
    foreach ($path in $managedDirectories) { Install-ManagedPath -RelativePath $path }
    $existingProvider = Join-Path $backupRoot "config\bigmodel_glm_5_2.json"
    if (Test-Path -LiteralPath $existingProvider -PathType Leaf) {
        Copy-Item -LiteralPath $existingProvider `
            -Destination (Join-Path $projectRoot "config\bigmodel_glm_5_2.json") -Force
    }
    foreach ($path in $managedFiles) {
        if (Test-Path -LiteralPath (Join-Path $packageRoot $path)) {
            Install-ManagedPath -RelativePath $path
        }
    }
    foreach ($path in $dataPaths) { Install-ManagedPath -RelativePath $path }

    Push-Location $projectRoot
    try {
        & $python -B -c "import hypothesis_mvp; from hypothesis_mvp.pcpi.smc import FixedUniverseSMC"
        if ($LASTEXITCODE -ne 0) { throw "Import smoke failed: $LASTEXITCODE" }
        & $python -B -m scripts.verify_delivery --root $projectRoot
        if ($LASTEXITCODE -ne 0) { throw "Installed delivery manifest failed: $LASTEXITCODE" }
        & $python -B -m scripts.run_pcpi_p3b6_predictive_consistency_diagnostic --help
        if ($LASTEXITCODE -ne 0) { throw "P3B.6 diagnostic CLI smoke failed: $LASTEXITCODE" }
        & $python -B -m scripts.run_pcpi_p3b7_budget_resolved_classes_diagnostic --help
        if ($LASTEXITCODE -ne 0) { throw "P3B.7 diagnostic CLI smoke failed: $LASTEXITCODE" }
        & $python -B -m scripts.run_pcpi_p3b8_joint_eig_diagnostic --help
        if ($LASTEXITCODE -ne 0) { throw "P3B.8 diagnostic CLI smoke failed: $LASTEXITCODE" }
        & $python -B -m scripts.run_pcpi_p3b9_representative_safe_diagnostic --help
        if ($LASTEXITCODE -ne 0) { throw "P3B.9 diagnostic CLI smoke failed: $LASTEXITCODE" }
        & $python -B -m scripts.run_pcpi_p3b10_maximin_joint_eig_diagnostic --help
        if ($LASTEXITCODE -ne 0) { throw "P3B.10 diagnostic CLI smoke failed: $LASTEXITCODE" }
        & $python -B -m scripts.run_pcpi_p3b_real --help
        if ($LASTEXITCODE -ne 0) { throw "P3B.10 real CLI smoke failed: $LASTEXITCODE" }
        & $python -B -m pytest -q -p no:cacheprovider `
            "tests\test_pcpi_p2a_smc.py" `
            "tests\test_pcpi_p2a1_diagnostic.py" `
            "tests\test_pcpi_p2a1_delivery.py" `
            "tests\test_pcpi_p2a_real_protocol.py" `
            "tests\test_pcpi_leakage_boundaries.py" `
            "tests\test_pcpi_p2b_transdimensional.py" `
            "tests\test_pcpi_p2b_delivery.py" `
            "tests\test_pcpi_p3_acquisition.py"
        if ($LASTEXITCODE -ne 0) { throw "Installed minimum tests failed: $LASTEXITCODE" }
    }
    finally { Pop-Location }
    $installedManifest = Get-Content `
        -LiteralPath (Join-Path $projectRoot "DELIVERY_MANIFEST.json") `
        -Raw | ConvertFrom-Json
    Write-Host (
        "Installed canonical PCPI {0} source: {1}" -f `
            $installedManifest.stage, $projectRoot
    ) -ForegroundColor Green
    Write-Host "Raw data, outputs, provider credentials, and held-out state were preserved." -ForegroundColor Green
}
catch {
    for ($index = $installed.Count - 1; $index -ge 0; $index--) {
        $relative = $installed[$index]
        $destination = Join-Path $projectRoot $relative
        $backup = Join-Path $backupRoot $relative
        if (Test-Path -LiteralPath $destination) {
            Remove-Item -LiteralPath $destination -Recurse -Force
        }
        if (Test-Path -LiteralPath $backup) {
            New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
            Move-Item -LiteralPath $backup -Destination $destination
        }
    }
    throw
}
finally {
    if (Test-Path -LiteralPath $installStage) {
        Remove-Item -LiteralPath $installStage -Recurse -Force
    }
}
