# Feedback R-v2：离线实现与边界修复

本次实现以 main `2cf68b778f2d3303bc6cb43ca7631e6b0aec7e3b` 为基线。
对应适配分支基线为 `b25ace147a2e5a616bffc0d8137b762fc36af247`。
这是一项待验证的 generation policy 修复，尚无优化成功或最终 bank 提升结论。

## 取证结论及干预

原阶段二 final admitted-bank 的 D−C 为 −0.075517，95% CI 跨零。
5 次 hard abstention 的对应 C 候选没有独立入库，不能把停止直接解释为最终 bank 损失。
最强负差坐标在正常生成后出现不同入库结果，支持同时保护 blind exploration，避免只修停止开关。
历史 96 个 children 的 PySR 均未正常执行；新旧差异不能单独归因于 PySR，任务、seeds 与公共正确性修复也有变化。

R-v2 入口：`DiscoveryAgentConfig.feedback_generation_policy`，值为
`protected-blind-with-blind-fallback-v2`。默认 `legacy` 保留原 C/D 路径。
核心代码为 `hypothesis_mvp/discovery/feedback_generation.py`。

| 项目 | 登记行为 |
| --- | --- |
| slot 1 | 独立 blind；new-skeleton quota 为 1 |
| slot 2 | gap 允许时 gap-directed；否则独立 blind；quota 为 0 |
| 上限 | 每 island/round 最多 2 个 materialized candidates；总新 skeleton quota 仍为 1 |
| blind 初始 parent | 当前 cycle 已评估的冻结 engine seeds，缺失时显式失败 |
| blind 后续 parent/history | 仅本 blind slot 的 winner 与 development-validation failure enums |
| 隔离 | 不提供 gap、Scientist review、共享 library、跨 slot refinements、independent-admission 结果 |
| 同轮顺序 | 两个请求都完成后再评估候选；blind 不获得另一槽当轮结果 |
| 停止 | 固定轮数或原 evaluation-budget 上限；provider/contract 错误终止；没有 winner 不提前停止 |
| lineage | cycle、round、island/parent、slot、stream、context hash 与原 provider lineage 联合绑定 |
| 可审查证据 | 每 cycle trace 记录 generation policy、slots、候选数、quota、parent、gap identity、lineage 和 telemetry |

两个 blind slot 独立维护自己的状态；新 cycle 从冻结 engine seeds 重新开始。
第二个 slot 的 fallback 标记只写 audit，不进入 blind prompt。
blind 新 skeleton 优先是明确登记的设计选择：gap 槽不能另占一个新 skeleton quota。
两次独立 prompt 可能比 D 的单批请求更贵；候选数匹配不等于计算成本匹配。

posterior-gap 计算、独立 admission、posterior 更新、prior/likelihood、threshold 与 reporting scorer 的函数体保持原样。
当原 admission 判定不允许入库时，blind fallback 也不能绕过它。
`proposal_gap_identity` 在 R 中表示 controller 消费的 gap 决策；真正 prompt 暴露由各 slot 的 gap identity/context hash 验证，blind 的 gap identity 必须为空。

## 同一新环境的公共正确性修复

这些修复共同作用于新 C/D/R；不用于重写历史结果。

| 审查问题 | 实现 |
| --- | --- |
| effectful string 在 SymPy 检查前执行 | 数学 AST allowlist 直接构建 SymPy 节点；禁用属性、下标、字符串和任意调用；engine 评分也走此边界 |
| NaN 预测被删除后可能得到完美分数 | 任意非法预测使整个指标失败；保留非法数量；shape 不匹配不截断 |
| train-only 仍加载 test/OOD 或答案元数据 | 配置先于数据加载；只选指定 task 的 train datasets；parquet 只投影允许的变量元数据列 |
| 平移/窄 Student-t 混合导致 raw-real-line 积分漏峰 | Arb cell/CDF/tail 外包围替代该 decision-risk 积分；不足以分离时保留宽区间 |
| readiness 不要求行动排序分离 | 最大 lower 必须严格超过其他 upper 加原 familywise numerical resolution |
| prefix-AULC 被当作累积收益 | 明确为 8/16/32 上 one-step expected reduction/current Bayes risk；每 prefix 重选 bank/partition；log2 横轴；没有 0 点 |
| 提前读取 action responses | adapter 先切 X 列；realized reader 仅按选定 index 解码；可用 decision_directory 在 reveal 前 fsync 写决定 |
| budget 与超时 | DRR 总预算不得小于固定逐 cycle 预算总和；engine 默认独立进程，serial/parallel 均有可终止超时 |
| QD 证据 | 缺少验证指标不算 verified；重复 task/emitter 拒绝；按真实非 intercept 支持数计 descriptor |
| import 与依赖 | 显式主仓库 root 和实际模块来源一致；NumPy 下限升至 2.0，与 np.trapezoid 一致 |
| 历史 delivery manifest | 原 P3F.2 manifest 保留为历史证据；README 不再称其覆盖当前树 |

数值外包围只覆盖给定二进制浮点数定义并归一化的有限混合。
它不证明 posterior 参数构造中的舍入、模型正确性或真实世界 no-harm；`rigorous_enclosure` 对端到端 posterior 仍为 false。
Arb 256-bit 计算每个 class 的 cell mass；`max(class mass)` 给 lower，total mass 给 upper。
有限 cell 的 Student-t 单峰性还给密度上下界；一个 class 全区间支配时可用其 CDF mass 收紧。
无限 tails 用 mass 界保留；按最大不确定 cell 分裂，达到 cap 后保留当前合法区间。
这避免把自适应 quadrature 的内部误差估计当作严格证明。

## 实验与交付边界

v1 原始 bytes 在适配仓库 `protocols/archive/feedback_repair_v1.json` 原样归档。
v2 和 hash domains 单独登记，不覆盖 v1。raw SHA256 和 canonical-JSON SHA256 不混用；未知历史预期 hash 不伪称已匹配。
C/D YAML bytes 不变，R 仅增加上述 generation policy（以及 name/condition）。

唯一开发比较仍为 12 tasks × 2 seeds × C/D/R = 72 children，无另加 48-child C/D 机制实验。
primary 只能取 `iterative_feedback_trace[-1].bank_rows_after`，不允许 evaluated-bank fallback。
比较 D−C、R−D、R−C；按 task 聚合，PCG64 seed 20261021、10000 bootstrap、95% percentile CI。
选择门槛保持 R−D lower > 0 且 R−C lower >= 0。缺失/失败 children 不从分母删除、不换 seeds、不重跑。
全部生成/admission 结果先持久冻结，再打开 development reporting 响应。
PySR 与 polynomial 的实际成功率、候选产出、admission、provider attempts/tokens/wall time 必须随主结果报告。

当前只运行合成 fixtures 和无 provider correctness tests，没有调用实验 provider、读取新科学数据或运行 72-child 实验。
本环境的 polynomial 真实后端合成 fit 与进程超时测试可执行；完整 PySR/Julia Gate 必须在登记的实际 child interpreter 上完成。
适配仓库 `docs/feedback_repair_v2_execution.md` 给出一次运行入口与阻断规则。
详细验证及受保护函数 hash 见 `docs/feedback_repair_v2_validation.json`。

后续实现修正与真实 JSON 协议验收见 `docs/feedback_repair_v2_followup.md`；当前后续验证记录为 `docs/feedback_repair_v2_followup_validation.json`。前一轮 validation JSON 保留为历史记录。
