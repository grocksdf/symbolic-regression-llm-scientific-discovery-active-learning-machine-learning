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
$runner = Join-Path $projectPath 'scripts\run_pcpi_p3i3_decision_targeted_real_acquisition.py'
$stdoutLog = $outputPath + '.stdout.log'
$stderrLog = $outputPath + '.stderr.log'
$progressPath = Join-Path $outputPath 'logs\run.jsonl'
$terminalFailurePath = Join-Path $outputPath 'TERMINAL_FAILURE.json'
$process = $null
$hadEvidence = $false

if (-not (Test-Path -LiteralPath $projectPath -PathType Container)) {
    throw "项目目录不存在：$projectPath"
}
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "冻结 Python 不存在：$pythonPath"
}
if (-not (Test-Path -LiteralPath $dataRootPath -PathType Container)) {
    throw "真实数据目录不存在：$dataRootPath"
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "P3I.3 配置不存在：$configPath"
}
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "P3I.3 runner 不存在：$runner"
}
if (
    $outputPath -ne $outputsRoot -and
    -not $outputPath.StartsWith($outputsRoot + '\', [System.StringComparison]::OrdinalIgnoreCase)
) {
    throw "正式输出必须位于项目 outputs/ 内：$outputPath"
}
if ($outputPath -eq $outputsRoot) {
    throw '正式输出不能直接使用整个 outputs/ 根目录'
}
if (
    [System.IO.Directory]::GetParent($stashPath).FullName.TrimEnd('\') -ne
    $projectParent
) {
    throw "证据暂存目录必须是项目同级的显式目录：$stashPath"
}
if ($stashPath.StartsWith($projectPath + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "证据暂存目录不能位于项目内部：$stashPath"
}
foreach ($freshPath in @($outputPath, $stdoutLog, $stderrLog, $stashPath)) {
    if (Test-Path -LiteralPath $freshPath) {
        throw "唯一运行路径已经存在，禁止重跑或覆盖：$freshPath"
    }
}

$status = @(& git -C $projectPath status --porcelain=v1 --untracked-files=all)
if ($LASTEXITCODE -ne 0) {
    throw '无法检查 Git 工作树状态'
}
$unexpected = @($status | Where-Object { $_ -notmatch '^\?\? evidence/' })
if ($unexpected.Count -ne 0) {
    throw "存在 evidence/ 之外的工作树变化：`n$($unexpected -join "`n")"
}
$hadEvidence = Test-Path -LiteralPath $evidencePath -PathType Container

try {
    if ($hadEvidence) {
        Move-Item -LiteralPath $evidencePath -Destination $stashPath
    }
    $remaining = @(& git -C $projectPath status --porcelain=v1 --untracked-files=all)
    if ($LASTEXITCODE -ne 0 -or $remaining.Count -ne 0) {
        throw "证据隔离后工作树仍不洁：`n$($remaining -join "`n")"
    }

    $actualBranch = (& git -C $projectPath branch --show-current).Trim()
    $actualCommit = (& git -C $projectPath rev-parse HEAD).Trim().ToLowerInvariant()
    $actualTree = (& git -C $projectPath rev-parse 'HEAD^{tree}').Trim().ToLowerInvariant()
    $actualConfigHash = (
        Get-FileHash -Algorithm SHA256 -LiteralPath $configPath
    ).Hash.ToLowerInvariant()
    if ($actualBranch -ne $ExpectedBranch) {
        throw "分支不匹配：$actualBranch"
    }
    if ($actualCommit -ne $ExpectedCommit.ToLowerInvariant()) {
        throw "提交不匹配：$actualCommit"
    }
    if ($actualTree -ne $ExpectedTree.ToLowerInvariant()) {
        throw "源码树不匹配：$actualTree"
    }
    if ($actualConfigHash -ne $ExpectedConfigHash.ToLowerInvariant()) {
        throw "配置哈希不匹配：$actualConfigHash"
    }
    if ($PreflightOnly) {
        Write-Host 'P3I.3 监督启动预检通过；未创建输出，未启动真实进程。'
        return
    }

    New-Item -ItemType Directory -Path $outputsRoot -Force | Out-Null
    $arguments = @(
        '-B',
        ('"' + $runner + '"'),
        '--data-root', ('"' + $dataRootPath + '"'),
        '--output-dir', ('"' + $outputPath + '"'),
        '--config', ('"' + $configPath + '"'),
        '--phase', 'P3I.3',
        '--heldout-state', 'closed'
    )
    $process = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList $arguments `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -WindowStyle Hidden `
        -PassThru
    Write-Host "P3I.3 已启动，PID=$($process.Id)，输出=$outputPath"

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
            Write-Error ("P3I.3 首错终止：{0}" -f $terminal.failure.failure_status)
            $terminalShown = $true
        }
        Start-Sleep -Seconds $HeartbeatSeconds
        $process.Refresh()
    }
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) {
        if (Test-Path -LiteralPath $stderrLog -PathType Leaf) {
            Get-Content -LiteralPath $stderrLog -Tail 40 | Write-Host
        }
        throw "P3I.3 正式进程失败，退出码：$($process.ExitCode)"
    }
    if (Test-Path -LiteralPath $terminalFailurePath) {
        throw "进程返回成功但存在终止记录：$terminalFailurePath"
    }

    $summaryPath = Join-Path $outputPath 'summary.json'
    $manifestPath = Join-Path $outputPath 'RUN_MANIFEST.json'
    foreach ($terminalPath in @($summaryPath, $manifestPath)) {
        if (-not (Test-Path -LiteralPath $terminalPath -PathType Leaf)) {
            throw "正式终态文件缺失：$terminalPath"
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
        throw 'P3I.3 summary 未通过完整性、失败或 held-out 终态检查'
    }
    if (
        $manifest.stage -ne 'P3I.3' -or
        -not $manifest.protocol_gate_passed -or
        -not $manifest.evidence_registry.valid -or
        $manifest.source_git_commit -ne $ExpectedCommit.ToLowerInvariant() -or
        $manifest.source_git_tree -ne $ExpectedTree.ToLowerInvariant() -or
        $manifest.config_file_hash -ne $ExpectedConfigHash.ToLowerInvariant() -or
        $manifest.heldout_opened -or
        $manifest.selection_used_heldout
    ) {
        throw 'P3I.3 manifest 的源码、配置、证据或 held-out 身份不匹配'
    }
    Write-Host (
        "P3I.3 完成：protocol=PASS assessment={0} runs={1}/{2}" -f
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
            throw "无法恢复历史 evidence/；安全副本保留在：$stashPath"
        }
        Move-Item -LiteralPath $stashPath -Destination $evidencePath
    }
}
