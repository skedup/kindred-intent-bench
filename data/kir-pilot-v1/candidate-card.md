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
| multi-turn | 40 |
| multi-turn patterns | override/recency/resolution/reconsideration 各 10 |
| hard-negative | 40 |
| context-distractor | 24 |
| context-control | 24 |
| hypothesized weak-model probe | 40 |
| brandless XHS | 8 |
| rest/eat confusion | 16 |

每个 intent 有两个 near-OOS/sibling 最小对照；24 条 context-distractor 各有一条只改变一个背景事实的
context-control。两类对照均共享 `scenario_family_id / contrast_group_id`。第 9/10 个 in-scope 候选组成
小型 paraphrase pair，避免 IE1.3 把相关样本拆到不同 split 或当作独立 bootstrap 证据。

每组 context pair 还共享显式 `context_perturbation`，登记背景前缀与新增 recent activity；validator 要求
distractor 精确等于向 control 应用该 delta，禁止混入第二个事实或覆盖当前行动。

`hard_negative` 固定为 near-OOS 16 + no-intent 16 + ambiguous 8，每条通过 `hard_negative_against` 声明
竞争 intent。`hypothesized_weak_model_probe` 是 near-OOS、brandless XHS 与 rest/eat confusion 的冻结并集，
不是实验后挑出的错题，也不能作为三份切片之外的第四份独立证据。

主要交集固定如下；最终报告必须同时展示该交集，不能把同一 case 重复计算为多份独立证据：

| 交集 | 数量 |
|---|---:|
| near-OOS ∩ hard-negative ∩ hypothesized probe | 16 |
| brandless XHS ∩ hypothesized probe | 8 |
| rest/eat confusion ∩ hypothesized probe | 16 |
| hard-negative ∩ hypothesized probe | 24 |
| multi-turn ∩ context-distractor | 12 |
| multi-turn ∩ context-control | 12 |

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
3. 检查 near/far OOS、hard-negative、hypothesized weak-model probe、brandless、multi-turn、context pair 和
   rest/eat tags 是否名副其实；
4. 对 hard-negative 核对 `hard_negative_against`；对 context pair 核对登记的 `context_perturbation` 确实只是
   单一背景事实；
5. 检查文本说话者确实表达 Kindred 的下一行动，而不是误标用户自己的行动；
6. 对同意的记录填写真实 reviewer/annotator provenance；有分歧时进入 IE1.1 仲裁流程；
7. 排除或修订任何无法形成唯一可审计 Gold 的候选，并补齐相同分层位置；
8. 全部完成后才允许 materialize 为正式 `Case`，再进入 IE1.3 group split。

若只有仓库维护者一人复核，仍需至少间隔三天盲重标，并在最终 dataset card 中声明这是
intra-annotator test-retest，而不是独立双标或 blind test。
