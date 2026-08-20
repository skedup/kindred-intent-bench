# KIR Pilot 标注规范（IE0 骨架）

> Status: `skeleton / do not author or freeze IE1 cases yet`
>
> 本文件只固定 IE1 标注前必须接受的边界，不包含 160-case 数据，也不授权生成 test prediction。

## 1. 标注对象

标注“输入中已经可观察到的、现在或下一段生活的主要行动意图”。不要根据天气、时间、疲劳、历史
Activity 或标注者偏好推断 Kindred 应该选择什么。

允许的 evidence carrier：当前 Thought/note 中已明确表达的想法、对话中 Kindred 已说出的欲望、人工合成
的第一人称意图表达、中性表达，以及明确的“没有具体想法/无法区分”。生产 State、Thought、真实用户消息
不得复制进本仓库。

## 2. 一级决策树

```text
是否表达现在或近期的行动倾向？
  否 -> no_intent
  是 -> 动作与对象是否具体到足以区分？
    否 -> ambiguous
    是 -> taxonomy 中是否有一个 Activity 能完整承接？
      是 -> in_scope + 唯一 target_intent
      否 -> oos
```

- `later` 愿望但没有当前行动倾向：`no_intent`，slot 保留 `horizon=later`；
- 同时出现多个行动时，有明确先后则标下一主要行动，否则 `ambiguous`；
- `oos` 是“具体但目录外”，不是“不确定”或“没有想法”；
- `ambiguous` 不能被当成困难样本的兜底标签。

## 3. Evidence 与 Gold

- `gold.evidence_quote` 必须是 NFKC 规范化后 `state_summary` 或某条 conversation content 的连续子串；
- quote 只用于 Gold 审计，绝不传给 baseline；
- `in_scope` 必须有 taxonomy 内唯一 target，其余三类 target 必须为 `null`；
- near-OOS 必须带 `near_oos` tag、至少一个 sibling intent，并通过相同 `contrast_group_id` 关联对应
  sibling-ID case；
- `annotation_note` 解释标签边界，不写模型应模仿的长推理。

## 4. 关系与泄漏控制

每条 case 都必须填写 `scenario_family_id / contrast_group_id / paraphrase_cluster_id / source`。共享任一前三个
关系 ID 的 case 在图上连边；完整连通分量生成同一个 `split_group_id`，并令
`bootstrap_cluster_id=split_group_id`。`source` 只作 provenance/分层，不能作为 group key。

dev/test 只能按完整 group 分配。test 不得进入 Prompt、few-shot、prototype 或阈值选择；任何 frozen test
修订都必须增加 dataset version。

## 5. IE1 开始前待补示例

IE1 必须先补齐并人工审阅以下内容，之后才能批量编写 Gold：

- 四类各至少 3 个正例、边界反例与仲裁例；
- 8 个 intent 各自 inclusion/exclusion 对照；
- near-OOS 与 sibling-ID 配对示例；
- multi-turn、context-distractor、brandless XHS、rest/eat confusion 示例；
- 标注者分歧记录、仲裁流程和 dataset card 限制声明。
