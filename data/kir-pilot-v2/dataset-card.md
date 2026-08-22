# KIR Pilot v2 Dataset Card

## 概要

KIR Pilot v2 是 Kindred activity routing 的离线、开放集意图识别评测集。它包含 160 条中文合成场景：dev 48 条、frozen test 112 条。所有条目均经过人工盲审或仲裁，正式 Case 保留逐条 review provenance。

本数据集只用于离线求职作品与方法验证，不接入 Kindred 生产路径，也不代表线上自然流量分布。

## 来源与标注

- 数据由 LLM 辅助起草，再由一名人工 reviewer 完成标签盲审；有分歧的条目经过显式仲裁。
- routing Gold 为 `in_scope / oos / no_intent / ambiguous`；开放目录外想法另存为非路由 `open_intent_candidate`，不改写 routing Gold。
- 数据不含线上用户日志、真实用户 PII 或生产会话。
- 当前仅有一名人工标注者，不能把本数据集的一致性视为多人共识；主观边界是已知限制。

## Split 方法

共享 scenario、paraphrase 或 contrast 关系的条目先合并为连通分量。dev/test 只按完整分量分配，且 `bootstrap_cluster_id = split_group_id`。主标签采用精确配额；诊断 tags 仅做确定性的近似分层。稳定选择 namespace 为 `kir-pilot-v2-group-split-v1`。

该 frozen test 是仓库内可见的 non-blind test。冻结后不得依据 test Gold、test 错误或 test prediction 调整 taxonomy、prompt、阈值、模型选择或数据；后续正式运行由独立 experiment lock 再绑定。冻结之前没有生成 test prediction。

## 分布

### Decision

| decision | dev | test |
| --- | --- | --- |
| in_scope | 24 | 56 |
| oos | 10 | 22 |
| no_intent | 7 | 17 |
| ambiguous | 7 | 17 |

### In-scope intent

| intent | dev | test |
| --- | --- | --- |
| create_picture | 3 | 7 |
| dine_out | 3 | 7 |
| eat_at_home | 3 | 7 |
| play_xiaohongshu | 3 | 7 |
| reach_out_to_user | 3 | 7 |
| rest | 3 | 7 |
| take_a_walk | 3 | 7 |
| visit_cultural_place | 3 | 7 |

### 诊断切片

| tag | dev cases | test cases | dev clusters | test clusters |
| --- | --- | --- | --- | --- |
| single_turn | 36 | 84 | 29 | 63 |
| multi_turn | 12 | 28 | 6 | 14 |
| multi_turn_override | 4 | 6 | 2 | 3 |
| multi_turn_recency | 2 | 8 | 1 | 4 |
| multi_turn_resolution | 2 | 8 | 1 | 4 |
| multi_turn_reconsideration | 4 | 6 | 2 | 3 |
| near_oos | 5 | 11 | 5 | 11 |
| far_oos | 5 | 11 | 3 | 9 |
| hard_negative | 11 | 29 | 8 | 20 |
| hypothesized_weak_model_probe | 12 | 28 | 8 | 19 |
| context_distractor | 7 | 17 | 7 | 17 |
| context_control | 7 | 17 | 7 | 17 |
| brandless_xhs | 1 | 7 | 1 | 6 |
| rest_eat_confusion | 6 | 10 | 4 | 6 |
| slot_required | 38 | 90 | 27 | 61 |
| quiet_control | 7 | 17 | 5 | 11 |

必需 bootstrap test cluster 数：`{"full":77,"gold_in_scope":43,"near_oos":11,"no_intent":11}`。预登记下限为每个切片 8 个 cluster。

## Semantic preservation audit

冻结前按 `risk-ranked-distinct-cluster-v1` 固定 24 个 test Case：in_scope 12（八个 intent 各至少一条）、OOS 4、no_intent 4、ambiguous 4。它们只用于 B3 primary/weak 的 action、object、horizon 语义保真诊断，不进入主 verdict，也不得反向用于调参。精确 ID 记录在 split 与 freeze manifest。

## 完整性绑定

- adjudicated pre-split Gold: `2b5d9424172a47e0045a36674b1cff6a01f589fad5c3773991f77f91cc331e66`
- dev.jsonl: `c4807ed66def5fe04402c8ee5c7d1a5460965dfc32f063a3f272b522fceb03c5`
- test.jsonl: `648ebce7fd1fe9d9ce9699fd96f27948a1aa675fa4f8db1d73aa07bf4ef57fa5`
- split-manifest.json: `cc68976c1c6eb4de664d05e529d31db1d5208073a4553c630e437e416f587902`

## 适用范围与限制

- 这是人为平衡的小型合成集，不估计生产 prevalence、真实准确率或商业收益。
- 同一套 taxonomy 和 authoring process 可能造成构造偏差；诊断 slice 结果只支持定位假设，不自动证明因果。
- near-OOS、开放意图和低性能模型退化是定向覆盖，不代表所有小众 activity 或自然语言变体。
- test 在私有仓库中对开发者可见，因此防泄漏依赖 hash freeze、操作纪律与完整实验记录，不是密码学盲测。
- dataset freeze 只承诺数据层不再变化；模型、prompt、provider 参数、预算和 verdict 必须在首次正式 test 前由 experiment lock 单独冻结。
