# KIR Pilot v2 Review 与 Grounded 仲裁

Grounding 校准后的 review workspace 已从 v1 按**完全相同的规范化 context**迁移 158 条人工标签；没有用
候选 ID 或语义相似度自动迁移。2 条新增上下文已由原 reviewer 独立填写，当前 review 为 160/160 complete。

补审工作簿 SHA-256 为 `a41658ad2bdac42dcdc0046f7fb3fbcad0f12c8cd9e4c084ad2cfaa2988aeee5`；导入回执
保存在 `initial/imports/`。以下导入命令保留为复现记录。

回传工作簿后先做不落盘校验：

```bash
uv run intentbench dataset reviews import-workbook \
  --cases data/kir-pilot-v2/candidates.jsonl \
  --taxonomy configs/kindred-activity-intents-v2.yaml \
  --review-dir data/kir-pilot-v2/reviews/initial \
  --workbook <returned.xlsx> \
  --dry-run
```

校验通过后去掉 `--dry-run` 导入。当前两步均已完成：

```bash
uv run intentbench dataset reviews validate
uv run intentbench dataset reviews build-adjudication-packet
```

## 仲裁导入结果

v2 comparison 得到 17 条 hierarchical disagreement；另抽取 15 条 agreement audit。填写后的
[grounded adjudication workbook](../../../outputs/2026-08-22-ie12-grounded-adjudication/kir-pilot-v2-grounded-adjudication.xlsx)，
已完成 32/32 条并导入：14 条分歧保留草稿标签、3 条接受人工标签、15 条一致抽查确认共同标签。

导入工作簿 SHA-256 为 `41ce0cb3b4f543673f434afe082e9a31d3264848439634e7d168b7525cc85255`；
机器 resolution 与回执保存在 `initial/adjudication/`。复现入口为：

```bash
uv run intentbench dataset reviews import-adjudication-workbook \
  --cases data/kir-pilot-v2/candidates.jsonl \
  --taxonomy configs/kindred-activity-intents-v2.yaml \
  --grounding configs/kindred-activity-grounding-v1.yaml \
  --packet-dir data/kir-pilot-v2/reviews/initial/adjudication \
  --workbook <returned.xlsx> \
  --adjudicator-id <stable-id> \
  --dry-run
```

3 条“接受人工标签”会改变当前 routing taxonomy 或 decision rule，不能作为单 case 例外静默写入 Gold。
已批准的 [policy resolution](initial/adjudication/policy-resolution.json) 保留原 routing Gold，并分别新增
`slow_run`、`learn_music`、`go_out_unspecified` 三个 `open_intent_candidate`；原仲裁选择仍保留在
`resolutions.jsonl` 中。

物化命令为：

```bash
uv run intentbench dataset gold materialize
```

输出 [adjudicated pre-split Gold](../adjudicated/cases.jsonl) 共 160 条；一级分布保持
`80/32/24/24`，semantic tags、16 组 near-OOS、24 组 context pair 和 112 个关系连通分量均通过审计。
该产物下一步只进入 group-safe split，不再回写原始 candidates 或仲裁工作簿。

v1 workspace 和仲裁包保留为历史审计证据，但它们绑定旧 candidate/taxonomy hash，不能用于 v2 freeze。
