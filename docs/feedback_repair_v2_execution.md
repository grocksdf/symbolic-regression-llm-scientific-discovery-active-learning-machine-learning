# Feedback repair R-v2：登记与一次开发比较

代码实现不等于效果成立。v2 正式取代归档 v1 的固定 1+1 设计，增加 gap abstention 时的第二槽 blind fallback。
不修改原 C/D YAML、admission、posterior、阈值、reporting split/scorer 或预注册统计选择门槛。

`protocols/feedback_repair_v2.json` 是干预定义；`feedback_repair_identities.json` 记录原始 bytes 的 SHA256。
v1 原样保存在 `protocols/archive/feedback_repair_v1.json`。
原 v1 raw digest 为 `5e30aea594307ff4606fe2528e458b72546e893d8e8a757fa071f5093628b2a2`；canonical-JSON digest 为 `436eb6ef875b6713045e647f411e95c7bdf575d57e1984caa3e6ddb131bfb243`。二者 hash domain 不同，均不替代尚未知晓的历史预期值。

## 离线验收与正式运行分开

源码交付已经实现 generation 隔离、候选预算、固定停止规则和 lineage synthetic tests。
本次没有实验 provider 调用、科学数据读取或真实机制/开发运行。
PySR 必须在实际 child interpreter 中通过合成 fit；不能引用其他环境的历史 passed=true 或 import 成功替代。
Gate 在读取科学数据文件 hash 或 provider-env 前执行；任一后端不可用均 PRE-RUN_NO-GO。

要求两仓库使用这次审查分支的干净 commit，`HYPOTHESIS_MVP_ROOT` 指向主仓库。
下面命令供用户在登记环境手动运行。路径参数由用户绑定到既有冻结输入；不自动找新数据或修改输入。

```powershell
$python = 'D:\01\666\.venvs\llm_srbench_metadata_py312\Scripts\python.exe'
$main = 'D:\path\to\main-repository'
$bench = 'D:\path\to\benchmark-repository'
$runRoot = 'D:\o\feedback_r_v2_unique'
$env:HYPOTHESIS_MVP_ROOT = $main
$env:PYTHONPATH = "$main;$bench"
Set-Location $bench

# 合成 y=x**2+0.5*x；不读取科学数组，不调用 provider。
& $python scripts/run_feedback_repair_runtime_gate.py `
  --reference configs/pysr_runtime_gate.json `
  --output "$runRoot\RUNTIME_GATE.json"

# Public input-symbol JSON 为 family/task -> 输入变量名列表。
# 不得含目标公式；12 tasks 必须与协议一致。
& $python scripts/build_feedback_repair_registration.py `
  --runtime-gate "$runRoot\RUNTIME_GATE.json" `
  --mainline-root $main `
  --hdf5 'D:\path\to\existing\lsr_bench_data.hdf5' `
  --metadata 'D:\path\to\existing\lsr_transform.parquet' `
  --provider-env 'D:\path\to\existing\provider.env' `
  --input-symbols 'D:\path\to\frozen\input_symbols.json' `
  --output "$runRoot\REGISTRATION.json"

# 默认只展示 72-child plan，没有 provider 调用。
& $python scripts/run_feedback_repair.py `
  --registration "$runRoot\REGISTRATION.json" --output "$runRoot\matrix"
```

明确决定运行后，由用户执行唯一一次矩阵：

```powershell
$registrationHash = (Get-FileHash "$runRoot\REGISTRATION.json" -Algorithm SHA256).Hash.ToLower()
& $python scripts/run_feedback_repair.py `
  --registration "$runRoot\REGISTRATION.json" --output "$runRoot\matrix" `
  --execute --authorized-registration-sha256 $registrationHash
```

已有输出目录不能复用，脚本无 resume/retry/replacement 路径。
2 个并行 children；provider transport 跨进程串行锁；每 child 10800 秒上限，超时终止进程树。
每 engine 的原 300 秒上限由独立进程终止机制执行。C/D/R 使用同一新 runtime 与 engine schedule。

## 冻结与统计

72 children = 12 个原冻结 tasks × seeds 2401/2402 × C/D/R；每 arm 24 个。
不额外运行旧阶段二 48 个 C/D children。

1. 校验 clean source/import origin、protocol/config/input hashes、依赖版本、Julia binaries/project、合成 Gate。
2. 完成全部 72 个 child attempts，保留每个失败、日志和 provider costs，无条件替换或二次尝试。
3. fsync 写 `GENERATION_FREEZE.json`，固定全部状态、artifact hashes 与最终 admitted rows。
4. 此后才读取 development reporting 响应，用未修改 scorer 计分。
5. 写 `REPAIR_RESULT.json`。任何缺失/失败 child 使 protocol_complete=false、development_selected=false；不做成功子集 primary 分析。

primary 为 `iterative_feedback_trace[-1].bank_rows_after`。
缺少完整两 cycle trace 或 final admitted rows 时失败；不能退回 evaluated-bank。
同一 task 先平均两个 seeds，再按 12 tasks 等权计算 D−C、R−D、R−C。
bootstrap PCG64 seed 20261021，10000 resamples，95% percentile CI。
原选择门槛保持 R−D lower > 0 且 R−C lower >= 0。
这一次开发比较可以同时复核原 D/C 和评估 R；不能把跨历史/新任务的 D−C 变化单独归因于 PySR。

候选上限匹配不代表调用成本相同。R 将候选拆为两个独立 prompt，最大 24 个 inner slot requests；
最多 6 个 Scientist review requests（每 cycle 首次 + 原有最多 2 次 format-only repair），共 30 个 logical requests。
沿用全部 arms 的 6 次 transport attempts，上限 180；R 的 inner 协议 repair 关闭。
实际 attempts、tokens、wall time、engine 成功率、生成与 admission counts 随结果保留。

这个协议不授权无人监督实验，也不代表 untouched confirmation。

后续验收修正：在 reporting 之前检查完整 72 个唯一坐标，并核对冻结 bank 与实际 final admitted trace；worker 边界失败保留为 failed child。协议及 C/D bytes 未修改，下一次登记应绑定最新干净源码。详见 `docs/feedback_repair_v2_followup_validation.json`，前一版 validation JSON 保留为历史记录。
