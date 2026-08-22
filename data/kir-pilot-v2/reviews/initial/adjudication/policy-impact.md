# IE1.2 仲裁后的分类合同影响

> 状态：`resolved with routing-plus-open-intent-v1 / materialized`
>
> 工作簿 SHA-256：`41ce0cb3b4f543673f434afe082e9a31d3264848439634e7d168b7525cc85255`
>
> Resolution SHA-256：`ac8aa06c2e3cd921376ed0133b13c5ec087b49f9489991e50dfc11be403ef9a2`

32 条仲裁结果已通过 candidate、taxonomy、grounding、blind-review response 四重 hash gate 导入。14 条分歧
保留草稿标签，15 条一致抽查确认共同标签；以下 3 条选择接受人工标签，但它们不只是修正单个样本，而会改变
当前 routing taxonomy 或一级 decision 规则，因此在修改 Gold 前单独复核。

| candidate | 已选择的标签 | 当前合同冲突 | 直接物化的结构影响 |
|---|---|---|---|
| `draft-kir-pilot-v2-0094` | `in_scope/take_a_walk/now` | runtime `take_a_walk` 只注册 `walk`；taxonomy 明确排除慢跑 | 失去 `take_a_walk` near-OOS 对照；`near_oos`、`hard_negative` 与 in-scope 真值冲突 |
| `draft-kir-pilot-v2-0131` | `oos/later` | 当前任务只路由当前/下一行动；“哪天可能……但不是现在”按书面规则是 `no_intent/later` | `quiet_control` 与 OOS 冲突；OOS 又缺少 near/far OOS 归属 |
| `draft-kir-pilot-v2-0151` | `in_scope/take_a_walk/now` | “离开屋子”没有承诺步行，不能证明 `walk` 是唯一可执行行动 | 无单条 tag 真值冲突，但扩大 `take_a_walk` 的语义边界 |

若原样采用，一级标签分布从 `in_scope/oos/no_intent/ambiguous = 80/32/24/24` 变为
`82/32/23/23`，`take_a_walk` 从 10 条变成 12 条，near-OOS 从 16 条变成 15 条。当前 IE1.2 的精确分布、
每 intent 10 条、每 intent 2 条 near-OOS sibling 以及 hard-negative 组成合同都会失效；不能只改三行 Gold 和 tags。

## 推荐处理

把“下一行动路由”与“目录外/未成形想法发现”拆成两个通道：

1. 这三条的 routing Gold 暂时保留草稿标签，继续遵守已绑定的 runtime Activity 合同；
2. 另增非路由字段 `open_intent_candidate`，分别记录 `slow_run/now`、`learn_music/later`、
   `go_out_unspecified/now`，允许进入日志、taxonomy backlog 或后续用户可见提示；
3. `open_intent_candidate` 不参与当前 HEM，也不被强行映射到最相近 Activity，避免为了捕捉开放想法而污染
   `take_a_walk` 与 `no_intent` 的路由含义；
4. 若产品明确希望 `take_a_walk` 承接“所有出门”，则应先同步修改 Kindred runtime Activity/Action 合同，
   再版本化 taxonomy、替换失效的 near-OOS/平衡样本并补做人审，而不是把三条作为例外。

该策略已由 `policy-resolution.json` 绑定 resolution hash 后批准。`resolutions.jsonl` 继续保留用户的原始
仲裁选择作为不可变审计证据；物化器不覆写它，而是在 pre-split Gold 中增加 `open_intent_candidate`。

物化结果位于 `data/kir-pilot-v2/adjudicated/cases.jsonl`，SHA-256 为
`2b5d9424172a47e0045a36674b1cff6a01f589fad5c3773991f77f91cc331e66`。一级标签、semantic tags 与关系合同
全部有效；下一步进入 IE1.3 group split，而不是再次修改这三条 routing Gold。
