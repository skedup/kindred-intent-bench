# KIR Pilot v1 人工盲审工作区

> Historical：本工作区绑定未 grounding 的 v1 taxonomy。其人工输入与原始回执保留用于审计；旧仲裁包中的
> `product_policy_required` 是校准前建议，不能作为当前结论。当前工作区见
> [KIR Pilot v2 人工补审](../../kir-pilot-v2/reviews/README.md)。

本目录用于 IE1.2 的人工独立标签复核。`initial` pass 已完成 160/160 并进入分歧仲裁；在仲裁完成前它仍
不是正式 Gold，也不允许生成模型 prediction。

## 盲审边界

- `initial/blind/*.jsonl` 只暴露随机化的 `review_item_id` 和模型实际可见的 `context`；不包含候选 ID、Gold、
  tags、关系 ID 或 provenance。
- `initial/responses/*.jsonl` 是已完成的 ReviewRecord；`initial/imports/*.json` 用工作簿 SHA-256、reviewer、
  file mtime 和导入时间记录来源。
- `manifest.json` 绑定候选文件和 taxonomy 的 SHA-256；候选内容发生变化后，旧 review workspace 会失效。
- review 顺序由 `review_pass + case_id` 的 SHA-256 稳定打乱。可选 `blind_retest` 使用不同顺序和 item ID，
  但在草稿揭示后只能衡量同一标注者稳定性。
- 仓库维护者能够访问原始候选，所以这是过程上的 label-blind review，不宣称外部 double-blind annotation。

## 开始人工复核

选择一个稳定、非敏感的 reviewer ID，然后按 batch 运行：

```bash
uv run intentbench dataset reviews run \
  --review-dir data/kir-pilot-v1/reviews/initial \
  --batch 1 \
  --reviewer-id <reviewer-id>
```

可用 `--limit 5` 先完成少量样本。每条提交后立即写回对应 response 文件，再次执行会自动跳过已完成项。
Reviewer 必须只根据显示的 context 判断：

1. `decision`；
2. `target_intent`（仅 `in_scope`）；
3. context 内连续的 `evidence_quote`；
4. `desired_experience / object / horizon`，未知字段留空。

`decision` 可输入完整名称，也可使用快捷键：`1=in_scope`、`2=oos`、`3=no_intent`、
`4=ambiguous`。响应文件始终保存完整名称，数字只用于交互录入。

如果已保存后发现填错，不要直接编辑 JSONL。使用相同 reviewer ID 定向重填；新答案完整通过校验前，旧答案
不会被覆盖：

```bash
uv run intentbench dataset reviews revise \
  --review-dir data/kir-pilot-v1/reviews/initial \
  --review-item-id review-initial-0001 \
  --reviewer-id <reviewer-id>
```

查看进度：

```bash
uv run intentbench dataset reviews status
```

Pass 未完成时，`status` 只显示 completed/pending 进度；候选一致率和 disagreement item IDs 保持锁定，避免
reviewer 根据即时反馈调整后续标签。只有 160 条全部写入后才揭示字段级一致性。

只有 160 条全部完成后，以下完成门禁才会通过：

```bash
uv run intentbench dataset reviews validate
```

## 后续阶段

完成版 Excel 通过下面的命令先全量 dry-run，再写入四个 response batch；任何 metadata/hash/context/label
错误都会在写入前拒绝，已有 response 冲突也不会被覆盖：

```bash
uv run intentbench dataset reviews import-workbook \
  --workbook <completed-review.xlsx> \
  --dry-run

uv run intentbench dataset reviews import-workbook \
  --workbook <completed-review.xlsx>
```

当前 dataset-QA 比较为：`decision=144/160`、`hierarchical=143/160`、`horizon=160/160`。Reference 是
`llm_assisted_draft_label`，不是模型 prediction；人工 initial label 也不是自动 Gold。自由文本只报告
exact-match 诊断，不用来决定分歧集合。

确定性仲裁包包含全部 17 条 hierarchical disagreement 和 15 条分层 agreement audit：

```bash
uv run intentbench dataset reviews build-adjudication-packet
```

历史填写入口是
[仲裁工作簿](../../../outputs/2026-08-22-ie12-adjudication/kir-pilot-v1-adjudication.xlsx)。其中 15 条一致项已
预填供抽查。运行时 grounding 审计已证明其中 7 条 near-OOS 均可由 Activity/Action 合同裁决，
`product_policy_required=0`；因此不要继续填写或复用这个旧包。v2 补审完成后重新生成当前仲裁包。

同一 reviewer 的 retest 现在只能作为可选的 intra-annotator stability 检查，不构成 IAA。真正的独立一致性
需要第二名未见候选草稿的 reviewer；当前 pilot 只需如实披露这一限制。
