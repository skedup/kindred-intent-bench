# Kindred Intent Bench

Kindred Intent Bench 是一个离线优先的 open-set intent recognition Workbench，用于评测 stateful Agent
在输入已经包含可观察意图证据时，能否把下一行动正确路由到 Activity taxonomy，同时显式处理
`oos / no_intent / ambiguous`。

它不替 Kindred 决定“应该想做什么”，也不把自主选择分布是否均匀当作正确性指标。当前已完成 IE0、IE1、
IE2，以及 IE3 的实现和 Gemini dev Prompt 选择；experiment lock 仍未冻结，正式 test 尚未运行。IE1.2
已从 Kindred 实际运行环境捕获 Activity/Action grounding，并完成 taxonomy
v2 与 160 条候选校准；158 条原人工标签按完全相同 context 迁移，2 条新增边界样本也已完成人工补审。
32 条 grounded 仲裁结果已通过 hash gate 导入；3 条 policy impact 已用 routing + open-intent 双通道解决。
160 条带 reviewer/adjudicator provenance 的 Gold 已完成 IE1.3 group-safe split：48 条 dev、112 条 frozen
test，dataset card、split manifest、SHA-256 与 24 条 semantic-audit IDs 均已冻结。仓库尚未运行 frozen
test；B0/B1/B2a 只运行了 frozen dev，experiment lock 仍为 draft，也没有接入 Kindred 生产路径。

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

## IE2.1 离线 evaluator

`intentbench evaluate dev` 只接受与 dataset freeze 中 hash/path 完全一致的 dev 和 taxonomy，并拒绝任何
`split=test` Case。它读取本地缓存的 Prediction JSONL，原子生成 `metrics.json`、行归一化
`confusion.csv` 和按固定错误 taxonomy 全量输出的 `badcases.jsonl`；missing/provider/schema failure 始终
留在完整 Gold universe 中计错。重复执行只允许字节完全相同并返回 `unchanged`。

`intentbench evaluate compare-dev` 对两组缓存 prediction 生成固定 `right-left` 的 paired
cluster-bootstrap artifact。实际 dev 中 full HEM 与 Gold-in-scope intent Macro-F1 分别有 35/21 个
cluster，可生成 CI；near-OOS/no-intent 都只有 5 个 cluster，只保留 point estimate 并明确标记
`insufficient_clusters`，不会降级成 case-level bootstrap。两个命令都不会调用 Provider。

```bash
uv run intentbench evaluate dev \
  --predictions experiments/<run-id>/predictions.jsonl \
  --output-dir experiments/<run-id>

uv run intentbench evaluate compare-dev \
  --left-predictions experiments/<left>/predictions.jsonl \
  --right-predictions experiments/<right>/predictions.jsonl \
  --left-id <left> --right-id <right> \
  --output experiments/comparisons/<right>-minus-<left>/bootstrap.json
```

IE1.2 review workspace 由 candidate/taxonomy hash、稳定乱序响应表和 complete gate 约束。v1 已完成的 158 条
完全相同 context 保留原 reviewer/timestamp 并写入显式 migration receipt；2 条新 context 不能暗中继承标签。
补审已通过完整工作簿和 hash gate 导入，入口与仲裁说明见
[v2 review workspace](data/kir-pilot-v2/reviews/README.md)。Grounding 还纠正了旧设计把公开发布误判为 OOS
的问题。仲裁导入证据与 3 条已用双通道解决的合同影响见
[policy-impact audit](data/kir-pilot-v2/reviews/initial/adjudication/policy-impact.md)。

## IE2.2/IE2.3 dev baselines

[experiment v2 draft](configs/kir-pilot-v2-experiment.yaml) 已登记完整 B0 lexical rules、B1 prototype 来源与
去重算法、256 组阈值网格，以及 `gemini-embedding-001 / SEMANTIC_SIMILARITY / 3072` 的 embedding 身份。
两个 runner 都只接收 hash-bound frozen dev，且 adapter 函数只取得 `Case.context`；Gold 只在 dev 内用于
B1 拒识 prototype 选择和阈值评分，`evidence_quote`、review 字段与 test 不会进入 adapter。

```bash
uv run intentbench baseline b0-dev \
  --output-dir experiments/ie2-b0-dev-final

uv run intentbench baseline b1-dev \
  --cache-dir .provider-cache/b1-embeddings-v2 \
  --output-dir experiments/ie2-b1-dev-final

uv run intentbench evaluate dev \
  --predictions experiments/ie2-b1-dev-final/predictions.jsonl \
  --output-dir experiments/ie2-b1-dev-final/evaluation
```

B1 activity centroid 只使用 taxonomy canonical examples；OOS/no-intent 先按 tag 分层，每个
`bootstrap_cluster_id` 只保留最小 case ID，再做 round-robin，最终为 8/5 条。首次空缓存运行需要 81 个
唯一文本的 Provider 调用；完整二次运行已实测 `provider_calls=0`、输出 `unchanged`。选定配置的脱敏
Prediction、manifest、B1 raw scores、prototype IDs、阈值与 response hash 已进入
[IE2 reduced reference bundle](experiments/reference/kir-pilot-v2-ie2-final/README.md)；其他临时实验仍不提交。

当前 dev 结果如下。B0 规则是在 dev 上收窄后得到的词法上界，`1.0` 只说明这套合成 dev 文字可被规则覆盖，
不是 frozen-test 或生产准确率；不得据此宣称 B0 优于语义模型。

| arm | dev HEM | dev decision Macro-F1 | 说明 |
|---|---:|---:|---|
| B0 | 1.0000 | 1.0000 | 冻结 lexical 四门控；0 Provider 调用 |
| B1 | 0.7708 | 0.6039 | 选中阈值 `0.00 / 0.00 / 0.30 / 0.02` |
| B2a | 1.0000 | 1.0000 | Gemini 3.6 Flash；one-stage/one-call；选定 dev Prompt v2 |

paired cluster bootstrap 的 `B1-B0` full-HEM point 为 `-0.2292`，dev 95% CI
`[-0.3636, -0.1111]`；Gold-in-scope intent Macro-F1 point 为 `-0.0500`，CI
`[-0.1750, 0.0000]`。near-OOS/no-intent 各只有 5 个 dev cluster，因此只报告 point，不作显著性结论。
B1 的主要缺口是 rejection/ambiguity gate，而不是 8-intent centroid 的普遍失效；这会作为 IE2.3 LLM
baseline 的对照，不通过扩大阈值网格或读取 test 继续追分。

B2a 固定 12 条类别平衡且 bootstrap-cluster 去重的 dev few-shot，使用
`thinkingLevel=minimal`、512 max output、无 retry、无 schema repair。完整 dev、只排除 12 个示例 ID 的
36 条子集，以及排除所有示例 cluster 的 30 条子集均为 `1.0000/1.0000`；实际 input/output 为
`156556/3104` tokens，估算成本 `$0.129057`，P50/P95
时延为 `1235.9/1879.6 ms`。一次 dev Prompt 修订只澄清 OOS/no-intent/ambiguous 的一般边界，随后停止
调优；[few-shot receipt](configs/kir-pilot-v2-b2a-few-shot-selection.yaml)与
[Prompt-selection receipt](configs/kir-pilot-v2-b2a-prompt-selection.yaml)登记了选择边界与停止规则。
B2a-B1 full HEM point 为 `+0.2292`，selection-set descriptive paired-cluster 95% interval 为
`[+0.1111,+0.3636]`。它没有校正同一 dev 上的阈值/Prompt selection bias，不是泛化置信区间，也不代表
frozen test 或生产准确率。

B2a generation cache 由真实 context、完整 rendered Prompt、taxonomy、Prompt template、few-shot、response
schema、model、adapter revision、structured-output mode、512 上限和 generation parameters 自动计算不可变
合同；不再接受调用方提供的独立 request hash。48 条离线重跑实测 48 hits、0 Provider calls、产物
`unchanged`，这证明流水线和 artifact 可复现，不证明模型重复采样稳定。B2b call-1 后续必须通过只读
`require_existing` 复用 reference bundle；缺失或身份漂移时直接失败，不能回退到网络调用。

## IE3 dev 选择与正式运行框架

IE3 已实现 B2b 同通道 verifier、B3 taxonomy-free `FormedIntention` → taxonomy grounding、正式 test 的双冻结
与 clean-commit 门禁，以及七格 LLM matrix checker。B3 阶段 A 的自由文本意图会写入
`formed-intentions.jsonl`；即使暂时不能映射到 Activity，用户仍可通过该日志观察模型表达的行动、对象、
体验、限定条件、备选行动与 horizon。

Gemini dev 按一次预登记结构修订后停止调 Prompt，选择结果如下；完整脱敏证据见
[IE3 dev selection bundle](experiments/ie3-dev/README.md)。

| arm | selected Prompt | dev HEM | schema invalid | total tokens |
|---|---:|---:|---:|---:|
| B2b | v2 | 0.9792 | 0 | 309,138 |
| B3 | v2 | 1.0000 | 0 | 245,110 |

两臂 token 绝对差约 20.7%，超过预登记的 10% 严格比较门。该事实已在 test 前披露：如果 formal test 仍有
同类差异，matrix 必须标记 `budget-confounded`，不能把 B3 的差值解释为已完成等算力因果消融。以上仍只是
48 条 synthetic、non-blind dev 的选择结果，不是泛化准确率。

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
uv run intentbench evaluate --help
```

以上命令不访问 LLM/embedding Provider；最后一条在已冻结仓库上执行幂等再验证并返回 `unchanged`。正式
test runner 后续必须通过双冻结 guard；dataset 已 frozen，但 experiment 仍是 `draft`，因此 test 仍不可运行。

## 显式 Provider smoke

IE0 readiness 只能使用合成 fixture；IE2.2 另有显式的 dev-only B1 runner。两者之外的 evaluator、数据与
冻结命令都不访问 Provider。各环境先生成 partial record：

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
