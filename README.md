# Kindred Intent Bench

Kindred Intent Bench 是一个离线优先的 open-set intent recognition Workbench，用于评测 stateful Agent
在输入已经包含可观察意图证据时，能否把下一行动正确路由到 Activity taxonomy，同时显式处理
`oos / no_intent / ambiguous`。

它不替 Kindred 决定“应该想做什么”，也不把自主选择分布是否均匀当作正确性指标。当前已完成 IE0 工程
合同和 IE1.1 标注合同。IE1.2 已从 Kindred 实际运行环境捕获 Activity/Action grounding，并完成 taxonomy
v2 与 160 条候选校准；158 条原人工标签按完全相同 context 迁移，2 条新增边界样本也已完成人工补审。
32 条 grounded 仲裁结果已通过 hash gate 导入；3 条 policy impact 已用 routing + open-intent 双通道解决。
160 条带 reviewer/adjudicator provenance 的 Gold 已完成 IE1.3 group-safe split：48 条 dev、112 条 frozen
test，dataset card、split manifest、SHA-256 与 24 条 semantic-audit IDs 均已冻结。仓库尚未运行 frozen
test，experiment lock 仍为 draft，也没有接入 Kindred 生产路径。

## IE0 已实现

- 绑定真实 Kindred 运行环境的 8-intent 版本化 taxonomy：包括 `play_xiaohongshu` 等小众 Activity；
- Pydantic Gold / Prediction / RunManifest 合同与四类输出真值表；
- 完整 Gold universe evaluator：missing、Provider failure、schema invalid 均计错，duplicate/extra 使 run 无效；
- HEM、固定标签 Macro-F1、OOS/no-intent/ambiguous、near-OOS、统一 slice metrics 与成对 context 指标；
- 关系图连通分量、paired cluster bootstrap、固定 Type-7 percentile CI；
- `promising / inconclusive / negative` 预注册确定函数；
- dataset freeze + experiment lock 双冻结 guard；
- B1 embedding prototype/actionability/OOS/ambiguity 公式与确定性阈值 tie-break；
- synthetic-only Provider readiness，不保存密钥或原始响应。

## IE1.1 已实现

- 四类 decision 各 3 个正例、3 个反例、3 个仲裁例，共 36 条非数据集 guideline fixtures；
- 8 个 intent 的 inclusion/exclusion 对照，以及每个 intent 一组 near-OOS/sibling 配对；
- multi-turn、context pair、brandless XHS、rest/eat、hard-negative 和假设性弱模型探针的明确标注边界；
- 标注分歧、仲裁、单标注者重检和 dataset-card 七项强制披露合同；
- 带防误用 wrapper 的 case 编写模板，以及 taxonomy 对齐的离线 validator。

这些资产只用于指导 IE1.2 人工编写与复核，不是正式数据集，也没有 case/split 身份。

## IE1 Gold 数据

[候选数据卡](data/kir-pilot-v2/candidate-card.md)和 `candidates.jsonl` 已提供 grounding 校准后的 160 条审阅队列：
`in_scope=80 / oos=32 / no_intent=24 / ambiguous=24`，并满足 8-intent、near/far OOS、24组
context control/distractor 对照、显式 background perturbation、hard-negative竞争intent和横切标签精确覆盖。

原始候选仍使用 `llm_assisted_pending_human_review + draft` 并作为不可变审计输入。人工复核后的
[adjudicated pre-split Gold](data/kir-pilot-v2/adjudicated/cases.jsonl) 也是不可变上游证据；IE1.3 从它
确定性生成正式 [dev](data/kir-pilot-v2/dev.jsonl) 与 [frozen test](data/kir-pilot-v2/test.jsonl)。共享
scenario/paraphrase/contrast 关系的 Case 不跨 split，`bootstrap_cluster_id=split_group_id`。

[dataset card](data/kir-pilot-v2/dataset-card.md) 披露了合成、单标注者和 non-blind test 限制；
[split manifest](data/kir-pilot-v2/split-manifest.json) 记录 35/77 个 dev/test group、精确主标签配额、诊断
切片分布及 24 条 semantic-preservation audit IDs；[freeze manifest](data/kir-pilot-v2/freeze-manifest.json)
绑定 taxonomy/dev/test/split hashes。四个 required test domain 的 cluster 数为 77/43/11/11，均满足下限 8。

IE1.2 review workspace 由 candidate/taxonomy hash、稳定乱序响应表和 complete gate 约束。v1 已完成的 158 条
完全相同 context 保留原 reviewer/timestamp 并写入显式 migration receipt；2 条新 context 不能暗中继承标签。
补审已通过完整工作簿和 hash gate 导入，入口与仲裁说明见
[v2 review workspace](data/kir-pilot-v2/reviews/README.md)。Grounding 还纠正了旧设计把公开发布误判为 OOS
的问题。仲裁导入证据与 3 条已用双通道解决的合同影响见
[policy-impact audit](data/kir-pilot-v2/reviews/initial/adjudication/policy-impact.md)。

预注册采用三 Provider、七运行产物矩阵：`gemini-3.6-flash` 是 primary，完整运行
`B2a/B2b/B3`，且是唯一全局 verdict authority；`deepseek-v4-flash` 是低成本弱模型复现，
`gpt-5.6-luna` 是跨 Provider 参考复现，二者只运行 `B2b/B3` 并各自比较 `B3-B2b`，不跨模型平均、
不在看到 test 后替换 primary。B1 embedding 固定为 `gemini-embedding-001`。

## 本地门检

需要 Python 3.10+ 与 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv run intentbench grounding validate
uv run intentbench taxonomy validate configs/kindred-activity-intents-v2.yaml
uv run intentbench annotations validate
uv run intentbench dataset candidates validate
uv run intentbench dataset reviews status
uv run intentbench dataset gold split-freeze
```

以上命令不访问 LLM/embedding Provider；最后一条在已冻结仓库上执行幂等再验证并返回 `unchanged`。正式
test runner 后续必须通过双冻结 guard；dataset 已 frozen，但 experiment 仍是 `draft`，因此 test 仍不可运行。

## 显式 Provider smoke

下面的命令是 IE0 唯一允许访问 Provider 的路径，只能使用合成 fixture。各环境先生成 partial record：

```bash
uv run intentbench providers check \
  --role primary_decision --role embedding \
  --fixture tests/fixtures/provider-smoke.json \
  --config configs/kir-pilot-v1-experiment.yaml \
  --output .provider-cache/google.json \
  --environment-label us-linux-kindred-runtime

uv run intentbench providers check \
  --role weak_decision --role cross_provider_reference \
  --fixture tests/fixtures/provider-smoke.json \
  --config configs/kir-pilot-v1-experiment.yaml \
  --output .provider-cache/cross-provider.json \
  --environment-label mac-kindred-runtime-config

uv run intentbench providers merge \
  --fixture tests/fixtures/provider-smoke.json \
  --config configs/kir-pilot-v1-experiment.yaml \
  --input .provider-cache/google.json \
  --input .provider-cache/cross-provider.json \
  --output configs/provider-readiness.json
```

运行前由操作者在对应环境安全加载配置中的环境变量；命令行和仓库都不写 key。已提交的合并记录中，
Gemini/embedding 来自用户授权的美国 Linux，DeepSeek/OpenAI 来自 Kindred 的 Mac 运行配置，仅包含
identity、usage、延迟、环境标签、向量维度与 raw-response SHA-256；不包含 key 或响应正文。合并器重新计算
当前 experiment/fixture hash，并逐角色核对 model、adapter、arms、verdict authority、结构化输出和 usage
证据；四个角色必须恰好各出现一次。单条 fixture 命中只表示接口就绪，不是效果结果。

完整任务定义见 [设计文档](docs/design.md)，阶段与 Exit Gate 见
[实施计划](docs/implementation-plan.md)，IE1 标注规则与示例见
[标注规范](docs/annotation-guideline.md)。
