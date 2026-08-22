# KIR Pilot v2 候选数据卡（Activity Grounding 校准）

> Status: `160 candidates preserved / adjudicated pre-split Gold materialized / not frozen`
>
> 数据文件：[candidates.jsonl](candidates.jsonl)

## 为什么有 v2

v1 的 intent 名称与 Kindred Activity 同名，但边界主要来自需求讨论和通用语义，没有逐个绑定实际运行环境中的
Activity/Action 合同。v2 以 `kindred-activity-grounding-v1` 为权威快照，并由
`kindred-activity-intents-v2` 表达可执行的 inclusion/exclusion 边界。

这次校准确认 `play_xiaohongshu` 不只是浏览：运行时还支持公开搜索、公开互动、创作和发布公开图文；私信明确
不在能力范围。其余 7 个 Activity 的近边界也都能由运行时合同裁决，不需要新增产品策略。

## 数据迁移

- 仍为 160 条：`in_scope=80 / oos=32 / no_intent=24 / ambiguous=24`；8 个 intent 各 10 条正例；
- v1 的 160 条候选全部经过 grounding impact audit；159 条草稿分层标签保持，公开发布样本由 OOS 改为
  `in_scope/play_xiaohongshu`；
- 158 条上下文逐字不变，复用原人工标签并保留原 reviewer/timestamp；迁移映射写入
  [migration receipt](reviews/initial/migrations/kir-pilot-v1-to-v2-grounding-v1.json)；
- 新增 2 条上下文用于补齐公开互动正例和私信 near-OOS，均已由原 reviewer 完成盲标并通过工作簿导入 gate；
- 原始候选记录仍是 `llm_assisted_pending_human_review + draft`，用于保持审计输入不变；物化 Gold 另存于
  [adjudicated/cases.jsonl](adjudicated/cases.jsonl)。

完整运行时审计见 [Activity Grounding Audit](../../docs/activity-grounding-audit.md)，逐候选影响见
[v1 grounding impact](../kir-pilot-v1/grounding-impact.yaml)。

## 当前门禁

Review、grounded 仲裁、双通道 policy resolution、semantic tags/关系复核和 provenance materialization 均已
完成。物化产物仍是 pre-split `AdjudicatedCase`；下一道门是 IE1.3 group-safe split/freeze。
