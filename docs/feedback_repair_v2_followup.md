# R-v2 后续验收：重复 parent 与冻结边界

这是 `a5e275ee`（主系统）及 `7ad6065f`（适配分支）的实现修正。
v2 干预定义、原 C/D YAML、72-child 设计、候选/新 skeleton 配额和统计选择门槛保持原样。
归档 v1 与已登记 v2 的协议 bytes 不变；下一次登记必须绑定新的干净 source commit。
本次只使用合成 correctness fixtures，没有实验 provider 调用或新科学数据读取。

## Protected blind 的 incumbent 绑定

blind prompt 已通过 `current_equation_state.expression` 提供本槽 parent，但协议验证器用于检查
`PROPOSE_NEW_SKELETON` 与 incumbent 重复的字段是 `current_expression`。
缺少后者时，同一个 parent 可以被解析器接受为新 skeleton，后续才在 evaluation 阶段发现重复。

现在从同一个本槽 parent 填入 `current_expression`。
它没有增加新信息，也不含 gap、共享 winner、admission 或额外响应。
真实协议解析器因此在 materialization 之前拒绝“新 skeleton = 本槽 parent”。
拒绝不触发第二次请求，不改变 slot 2，也不增加候选上限。

上一轮核心 runtime fixtures 替换了 `_request_validated`，没有覆盖这处真实解析路径。
本轮新增 fixtures 只替换网络 `_request`，原始 JSON 通过实际 `_request_validated` / `_validate_batch`：

- 原样返回 parent：修正前复现错误接收；修正后拒绝，logical request 仍为 1。
- gap=true：blind + gap 两个合法槽均可通过真实解析，lineage 正确。
- gap=false：blind + blind fallback 两个合法槽均可通过真实解析，logical requests 仍为 2。

## 逐槽 provenance

每条 slot audit 显式增加 cycle、round_id 和 island，便于将候选与 admission lineage 关联。
验证器要求两个槽属于同一 batch，candidate_count 为严格整数 0/1、quota 正确、lineage 数量匹配且不复用。
空 audit、重复 batch、错误 fallback 标记或不一致的 round 均失败。
这些是已登记预算和 lineage 的验收约束，不更改 generation 选择策略。

## 唯一一次开发矩阵的 reporting 入口

原 CLI 正常路径计划了全部 72 children，但独立调用 reporting helper 时，矩阵完整性检查发生得过晚。
本轮用 71-child 合成 fixture 复现：不完整矩阵可以触发 reporting callback，之后统计函数才拒绝。
现在共用矩阵校验，先检查 12×2×3 个唯一坐标及 family，再写 generation freeze；缺失、重复或未登记坐标不会触发 reporting。

冻结后，主指标的 bank_rows 还必须与 artifact 的
`iterative_feedback_trace[-1].bank_rows_after` 完全一致，之后才打开 reporting 响应。
不一致记录为 reporting_failed，保持原“不删除失败、不做成功子集 primary 分析”的规则。

worker 外层保证 log 创建或 failure-artifact 解码异常仍成为一个终止的 failed child；不重试，不让其他已完成 children 的结果消失。

## 证据边界

本轮验收报告在 `feedback_repair_v2_followup_validation.json`。
前一轮 `feedback_repair_v2_validation.json` 原样保留，代表前一版源码的历史验收。
PySR 的实际 registered-runtime Gate 与正式 72-child 开发对照仍由用户在实际环境执行。
这些修正不构成最终 admitted-bank 收益、PySR 机制效应或确认性效果证据。
