# KIR Pilot v1 候选数据卡（IE1.2）

> Status: `160 candidates generated / pending human review / not split / not frozen`
>
> 数据文件：[candidates.jsonl](candidates.jsonl)

## 用途与非用途

本文件描述 IE1.2 的人工复核队列。160 条候选用于逐条审查 Gold decision、target、evidence、slots、tags
和关系 ID；它们还不是可运行实验的正式 Gold。

候选记录有意缺少 `split / split_group_id / bootstrap_cluster_id`，统一使用：

- `source=llm_assisted_pending_human_review`；
- `annotator_id=codex-draft`；
- `adjudication_status=draft`；
- `draft-kir-pilot-*` 临时 ID。

因此，抽出单条候选也不能通过正式 `Case` schema。不得把本文件传给 runner，不得据此生成 prediction，
也不得把 validator 的 `valid_pending_human_review` 误写成“人工复核通过”。

## 当前结构覆盖

| 项目 | 当前数量 |
|---|---:|
| in_scope | 80（8 intents × 10） |
| OOS | 32（near 16 + far 16） |
| no_intent | 24 |
| ambiguous | 24 |
| multi-turn | 44 |
| hard-negative | 40 |
| context-distractor | 24 |
| weak-model trap | 40 |
| brandless XHS | 8 |
| rest/eat confusion | 16 |

每个 intent 有两个 near-OOS/sibling 对照；关系通过共同 `contrast_group_id` 连接。第 9/10 个 in-scope
候选组成小型 paraphrase pair，其他无关候选使用独立关系 ID，避免 IE1.3 产生超大连通分量。

## 来源

候选文本由 Codex 根据冻结 taxonomy 和 IE1.1 annotation pack 辅助起草，再由仓库内确定性脚本序列化。
生成过程不调用 Gemini、DeepSeek、OpenAI、Grok 或 embedding Provider，也没有使用 Kindred 生产 State、
Thought 或真实用户消息。

构建脚本只保证内容可复现和结构覆盖，不等于人工语义复核：

```bash
uv run python scripts/build_ie12_candidates.py
uv run intentbench dataset candidates validate
```

## 人工复核要求

复核者必须逐条执行以下动作：

1. 不参考未来模型 prediction，独立判断 decision 与唯一 target；
2. 确认 `evidence_quote` 是足以支持 Gold 的输入内连续片段；
3. 检查 near/far OOS、brandless、multi-turn、context distractor 和 rest/eat tags 是否名副其实；
4. 检查文本说话者确实表达 Kindred 的下一行动，而不是误标用户自己的行动；
5. 对同意的记录填写真实 reviewer/annotator provenance；有分歧时进入 IE1.1 仲裁流程；
6. 排除或修订任何无法形成唯一可审计 Gold 的候选，并补齐相同分层位置；
7. 全部完成后才允许 materialize 为正式 `Case`，再进入 IE1.3 group split。

若只有仓库维护者一人复核，仍需至少间隔三天盲重标，并在最终 dataset card 中声明这是
intra-annotator test-retest，而不是独立双标或 blind test。
