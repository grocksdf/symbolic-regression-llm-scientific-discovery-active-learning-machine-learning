param(
    [string]$OutputPath = "dist/hypothesis_mvp_canonical_source_p3b_10_representative_safe_maximin_joint_acquisition_20260812.zip",
    [string]$PythonPath = "python",
    [int]$TestsPassed = 0,
    [string]$Stage = "P3B.10",
    [string]$Task = "representative_safe_maximin_joint_acquisition"
)

$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$destination = if ([System.IO.Path]::IsPathRooted($OutputPath)) {
    [System.IO.Path]::GetFullPath($OutputPath)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $projectRoot $OutputPath))
}
$python = if (Test-Path -LiteralPath $PythonPath -PathType Leaf) {
    (Get-Item -LiteralPath $PythonPath).FullName
} else {
    (Get-Command $PythonPath -ErrorAction Stop).Source
}
$stagingRoot = [System.IO.Path]::GetFullPath((Join-Path ([System.IO.Path]::GetTempPath()) ("PCPISource-" + [guid]::NewGuid().ToString("N"))))
$bundle = Join-Path $stagingRoot "hypothesis_mvp"

try {
    Push-Location $projectRoot
    try {
        & $python -B "scripts\audit_final_source.py"
        if ($LASTEXITCODE -ne 0) { throw "Source audit failed: $LASTEXITCODE" }
        & $python -B -m pytest -q -p no:cacheprovider
        if ($LASTEXITCODE -ne 0) { throw "Tests failed: $LASTEXITCODE" }
    }
    finally { Pop-Location }

    New-Item -ItemType Directory -Path $bundle -Force | Out-Null
    foreach ($name in @(
        "hypothesis_mvp", "scripts", "tests", "docs", "config", "configs",
        "contracts", "schemas", "paper"
    )) {
        Copy-Item -LiteralPath (Join-Path $projectRoot $name) -Destination $bundle -Recurse -Force
    }
    foreach ($name in @(
        "pyproject.toml", "requirements.txt", "requirements-dev.txt", "pytest.ini",
        "README.md", "LICENSE", "run_pipeline.py", "install.ps1"
    )) {
        Copy-Item -LiteralPath (Join-Path $projectRoot $name) -Destination $bundle -Force
    }

    $dataStage = Join-Path $bundle "data"
    New-Item -ItemType Directory -Path $dataStage -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $projectRoot "data\README.md") -Destination $dataStage -Force
    foreach ($name in @("manifests", "split_manifests", "schemas")) {
        Copy-Item -LiteralPath (Join-Path $projectRoot "data\$name") -Destination $dataStage -Recurse -Force
    }

    $providerFile = Join-Path $bundle "config\bigmodel_glm_5_2.json"
    $provider = Get-Content -LiteralPath $providerFile -Raw | ConvertFrom-Json
    $provider.api_key = "REPLACE_WITH_BIGMODEL_API_KEY"
    [System.IO.File]::WriteAllText(
        $providerFile,
        ($provider | ConvertTo-Json -Depth 20),
        [System.Text.UTF8Encoding]::new($false)
    )

    $blocked = @(
        "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv",
        ".testenv", "venv", "build", "dist", "outputs", "results", ".git",
        ".patch_backups", ".hypothesis_patch_backups", "node_modules", "pytest-of-root"
    )
    Get-ChildItem -LiteralPath $bundle -Directory -Recurse -Force |
        Where-Object { $_.Name -in $blocked -or $_.Name -like "*.egg-info" } |
        Sort-Object FullName -Descending |
        Remove-Item -Recurse -Force
    Get-ChildItem -LiteralPath $bundle -File -Recurse -Force |
        Where-Object {
            $_.Extension.ToLowerInvariant() -in @(
                ".pyc", ".pyo", ".log", ".tmp", ".zip", ".bak", ".key", ".pem",
                ".aux", ".fls", ".fdb_latexmk", ".dat", ".xlsx", ".xls", ".7z"
            ) -or $_.Name -match "^(credentials|secrets|\.env)" -or
            $_.Name -in @(".coverage", "coverage.xml")
        } |
        Remove-Item -Force

    & $python -B (Join-Path $bundle "scripts\create_delivery_manifest.py") `
        --root $bundle `
        --output (Join-Path $bundle "DELIVERY_MANIFEST.json") `
        --stage $Stage `
        --task $Task `
        --tests-passed $TestsPassed
    if ($LASTEXITCODE -ne 0) { throw "Delivery manifest generation failed" }
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    if (Test-Path -LiteralPath $destination) { throw "Output ZIP already exists: $destination" }
    Compress-Archive -LiteralPath $bundle -DestinationPath $destination -CompressionLevel Optimal
    Write-Host "Canonical source package: $destination" -ForegroundColor Green
}
finally {
    if (Test-Path -LiteralPath $stagingRoot) {
        Remove-Item -LiteralPath $stagingRoot -Recurse -Force
    }
}
