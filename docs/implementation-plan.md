# Kindred Intent Bench 实施计划

> Status: `IE0 complete / IE1 not started`
>
> 日期：2026-08-21
>
> 权威设计：[design.md](design.md)

## 0. 目标与阶段边界

本计划把设计中的 IE0～IE4 转成一个**目标 9 个有效工作日、另留 2 天风险缓冲**的可执行 Pilot。首轮完成时，
仓库应能：

1. 校验版本化 taxonomy、Gold case、prediction 和 run manifest；
2. 在同一 160-case 数据集上运行 B0、B1、B2a、B2b、B3；
3. 离线重算指标、paired cluster bootstrap、混淆矩阵和 badcase；
4. 用冻结的确定函数输出 `promising / inconclusive / negative`；
5. 产出面向面试的 README、实验报告和可复现命令。

Pilot 不修改 Kindred，不读取真实 State、Thought 或用户消息，不执行 Activity/Capability，也不建设通用训练
平台、在线服务或 dashboard。IE5 的 240-case 扩展不进入首轮承诺。

## 1. 已冻结的工程决策

| 项目 | 决策 |
|---|---|
| 语言与包管理 | Python `>=3.10`、`uv`、`src` layout |
| 包与 CLI 名称 | Python package `intentbench`；命令 `intentbench` |
| 数据模型 | Pydantic；taxonomy 使用 YAML，case/prediction 使用 JSONL，结果使用 JSON/CSV/Markdown |
| 工程门检 | pytest、ruff、mypy；GitHub Actions 执行离线门检 |
| 仓库关系 | 不 import Kindred；只维护人工审核后的 taxonomy/context 静态快照 |
| 网络边界 | schema、evaluator、统计和报告完全离线；只有 LLM/embedding runner 可以访问 Provider |
| Provider 边界 | IE0 固定 primary/weak/cross-provider-reference/embedding 四个角色并做 dev-fixture smoke；只实现实际需要的窄 adapter |
| Secret | 只从环境变量读取；manifest 只记录配置名/hash，不记录 key、cookie 或原始环境变量 |
| 测试集边界 | 默认是 non-blind frozen test；正式 test 必须同时通过 dataset freeze 与 experiment lock hash 门禁 |
| 结果版本化 | 提交一组脱敏 reference run；临时 probe 和含 secret 的原始响应不进入 Git |

这些选择优先服务“短周期、可复现、方便面试讲解”，不提前抽象成通用 benchmark framework。

## 2. 目标目录

```text
kindred-intent-bench/
├── .github/workflows/ci.yml
├── .gitignore
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   ├── kindred-activity-intents-v1.yaml
│   ├── kir-pilot-v1-experiment.yaml
│   └── provider-readiness.json
├── data/
│   └── kir-pilot-v1/
│       ├── dev.jsonl
│       ├── test.jsonl
│       ├── dataset-card.md
│       ├── freeze-manifest.json
│       └── SHA256SUMS
├── prompts/
│   ├── b2a-one-stage/
│   ├── b2b-same-channel-verifier/
│   └── b3-evidence-then-grounding/
├── src/intentbench/
│   ├── cli.py
│   ├── schemas.py
│   ├── taxonomy.py
│   ├── dataset.py
│   ├── runner.py
│   ├── evaluator.py
│   ├── metrics.py
│   ├── bootstrap.py
│   ├── verdict.py
│   ├── freeze.py
│   ├── reporting.py
│   └── adapters/
│       ├── base.py
│       ├── rule.py
│       ├── embedding.py
│       └── llm.py
├── experiments/
│   └── <run-id>/
│       ├── manifest.json
│       ├── predictions.jsonl
│       ├── metrics.json
│       ├── confusion.csv
│       ├── badcases.jsonl
│       ├── semantic-audit.csv
│       └── report.md
├── tests/
│   ├── fixtures/
│   ├── unit/
│   └── integration/
└── docs/
    ├── design.md
    ├── implementation-plan.md
    └── annotation-guideline.md
```

## 3. 实施顺序与阶段门

```mermaid
flowchart LR
    A["IE0 合同、脚手架与单元测试"] --> B["IE1 160-case Gold 与冻结 split"]
    A --> C["IE2 evaluator 与 adapter 骨架"]
    B --> D["IE2 dev-only baseline 调优"]
    C --> D
    D --> E["Freeze manifest / Prompt / thresholds"]
    E --> F["IE3 B2b/B3 与正式 test run"]
    F --> G["IE4 报告与求职材料"]
    G --> H{"promising?"}
    H -->|"是"| I["可选 IE5：扩到 240"]
    H -->|"否"| J["保留结论与评测资产"]
```

任何阶段没有通过 exit gate，不进入下一阶段。尤其不能以“先看看效果”为理由提前运行 frozen test。

## 4. Pilot 任务清单

### IE0：仓库脚手架、评测合同与 Provider readiness（第 1 天～第 2 天上午）

#### IE0.1 最小工程脚手架

- 建立 `pyproject.toml`、`src/intentbench`、pytest/ruff/mypy 配置与 `uv.lock`；
- 建立 `intentbench --help` 和 GitHub Actions 离线门检；
- 默认忽略 `.env`、临时响应、Provider cache 和未选定的实验输出。

#### IE0.2 Schema 与 taxonomy

- 实现 `Taxonomy`、`IntentDefinition`、`Case`、`Gold`、`Prediction`、`RunManifest`；
- 按设计中的 Gold/Prediction 真值表固定四类字段组合、nullable/forbidden 条件和 `reason_short` 长度；
- 只有 `in_scope` 允许唯一 `target_intent`；near-OOS 必须声明 sibling intents 和关联 contrast group；
- 校验 `evidence_quote` 是规范化 context 字段中的连续子串；
- 写入 8 个 Activity、定义、inclusion/exclusion、canonical examples 和 sibling intents；
- 校验 taxonomy/version、case id、受控枚举、candidate 和 target 一致性。

#### IE0.3 Evaluator 合同先行

- 先用小型 fixture 固定 `hierarchical_exact_match` 的定义；
- 用 Gold IDs 作为评测全集：missing/provider/schema-invalid 计错误并进入 `invalid` confusion 列；
  duplicate/extra prediction 使 run invalid；paired comparison 禁止取成功样本交集；
- 实现 HEM、最小 `metrics.py/evaluator.py`、三态 verdict 边界和单元测试，不等待真实模型结果；
- 固定关系图连通分量生成 `split_group_id`，并令 `bootstrap_cluster_id=split_group_id`；
- 实现最小 paired cluster bootstrap，固定 seed、迭代数、最少 8 cluster 和 `[L, U]` 格式；
- 固定 ID Macro-F1 样本域、`zero_division=0`、context/schema error count 和 token 差异公式；
- 实现 dataset freeze + experiment lock guard 的接口和拒绝路径测试。

#### IE0.4 Provider 与模型前置验证

- 在 experiment draft 中登记 primary、weak、cross-provider-reference 三个 LLM 角色，以及 embedding
  model、各自 adapter、结构化输出模式和环境变量名；
- 每个模型只使用单条 synthetic dev fixture 验证连接、超时、结构化输出和 identity；
- LLM smoke 必须确认 input/output token usage 可得；缺失时在进入 IE1 前更换 adapter 或明确预算合同不可行；
- 凭据跨环境时输出带相同 experiment/fixture hash 的 partial records，合并器校验四个角色无缺失、无重复；
- 输出不含 secret 的 `configs/provider-readiness.json`；不生成 test prediction，不做效果判断。

**Exit gate IE0**

```text
uv sync
uv run ruff check .
uv run mypy src
uv run pytest
uv run intentbench taxonomy validate configs/kindred-activity-intents-v1.yaml
uv run intentbench providers check --role <role> --fixture tests/fixtures/provider-smoke.json
uv run intentbench providers merge --input <partial> --input <partial>
```

前四项完全离线并全部通过；Provider check 是显式手动 smoke，成功结果只记录 capability/usage metadata。
schema、invalid policy、cluster/bootstrap、双 freeze guard 与 verdict 边界已有自动测试；没有 Kindred import。

### IE1：160-case Gold 数据（第 2 天下午～第 4 天）

#### IE1.1 标注规范与模板

- 完成 `docs/annotation-guideline.md`；
- 固定 evidence carrier、四类决策树、near/far OOS、ambiguous 与 no-intent 边界；
- 固定 `scenario_family_id / contrast_group_id / paraphrase_cluster_id / source` 规则，以及 near-OOS 到
  sibling-ID 的关联规则；
- 建立正例、反例和需要仲裁的示例。

#### IE1.2 编写与复核数据

- `in_scope=80`：8 intents × 10；
- `oos=32`：far 16 + near 16；
- `no_intent=24`；
- `ambiguous=24`；
- 满足 multi-turn、hard-negative、context-distractor、weak-model trap、brandless XHS、rest/eat confusion
  的最低覆盖。

每条 Gold 必须能从输入内审计 decision evidence；`gold.evidence_quote` 不传给 baseline。所有内容使用合成
或人工脱敏场景，不拷贝生产 State/Thought。

#### IE1.3 Group split 与冻结

- 对共享 scenario/paraphrase/contrast id 的 case 建边，以连通分量生成唯一 `split_group_id`，并令
  `bootstrap_cluster_id=split_group_id`；`source` 只用于 provenance/stratification，不作为 group key；
- 以完整 group 为原子，近似分层得到 dev 48 / test 112；
- 自动检测相同或归一化文本、关系边和 group 的跨 split 泄漏；
- 输出分布和切片覆盖，生成 dataset card；
- 记录 taxonomy、dev、test 的 SHA-256；
- 在 freeze manifest 预选 24 个 semantic-preservation slice IDs；
- 冻结后任何修订都创建新 dataset version，不覆盖 `kir-pilot-v1`。

**Exit gate IE1**

- 160 条全部通过 schema、唯一 id、taxonomy target 和 evidence audit；
- 四类数量、8 个 intent 数量和横切标签覆盖符合设计；
- group leakage 检查为 0；
- Pilot verdict 所需的 full-test HEM、Gold-in-scope、near-OOS、no-intent 四个样本域各至少有 8 个 test
  bootstrap clusters，否则 IE1 不得冻结；
- dataset card 明确声明 non-blind frozen-test 限制；
- test hash 已写入 `freeze-manifest.json`，此前没有生成 test prediction。

### IE2：Evaluator、B0/B1/B2a（第 4～6 天）

#### IE2.1 确定性 evaluator

- 实现 HEM、decision Macro-F1、ID intent Macro-F1、OOS/no-intent/ambiguous 指标和 per-intent recall；
- 实现 near-OOS、sibling false reject、context distractor、schema invalid 等切片；
- 实现按固定 label set、`zero_division=0` 和完整 Gold 全集计算的指标；
- 实现按 `bootstrap_cluster_id` 的 paired draws、normalized confusion matrix 和确定性 badcase 选择；
- evaluator 只消费缓存 prediction，重复计算不得调用 Provider。

#### IE2.2 B0 与 B1

- B0 按冻结的 actionability → OOS → ambiguity → in-scope 顺序输出四类；
- B1 使用 frozen embedding、taxonomy examples、受限 dev prototypes 和固定 gate 顺序；
- Activity prototype 使用 normalize 后的 canonical-example centroid 与 cosine similarity；
- 阈值在冻结网格上按 dev HEM、decision Macro-F1 依次最大化，仍相同时用参数 tuple 字典序打破平局；
- 记录 prototype case id、阈值、embedding identity/version 和 adapter config hash；
- 所有选择只在 dev 完成，禁止根据 test 补规则或替换 prototype。

#### IE2.3 B2a 与通用 runner

- 定义窄 Provider Protocol、超时、重试、schema repair 与错误分类；
- runner 统一记录 latency、input/output token、费用快照和 raw-response hash；
- B2a 实现 one-stage/one-call，并只用 dev 选择 few-shot 和 Prompt；
- 缓存 key 使用规范化 case 内容、taxonomy、model、Prompt、adapter config 和参数 hash，不能只依赖 case ID；
- B2a 缓存保留为 B2b call-1 的唯一输入来源。

**Exit gate IE2**

- metric golden tests、cluster bootstrap seed test、confusion/badcase snapshot tests 通过；
- B0/B1/B2a 能完整跑 dev 并输出统一 prediction schema；
- 重跑缓存结果不产生网络调用且 metrics 相同；
- test 仍未用于阈值、Prompt、few-shot 或 prototype 选择。

### IE3：B2b/B3 公平消融与正式运行（第 6～7 天）

#### IE3.1 两个双调用实验臂

- B2a/B2b/B3 使用相同 few-shot case IDs、顺序和数量；Prompt 表达可以适配各阶段，监督 case 不变；
- B2b：call-1 必须直接复用相同 case/model/Prompt/config 的 B2a 缓存，call-2 才做同一闭集语义中的
  审查/修正；
- B3-A：只恢复输入中已有的自由文本 evidence representation，不看 taxonomy；
- B3-B：读取阶段 A 与完整 taxonomy，输出四类 routing decision；
- 对齐模型、调用次数、每次 output budget、输入事实、解码参数、候选顺序、重试和 repair policy。

#### IE3.2 Freeze 与运行

- 在 test 前冻结 B0/B1 配置、三个 LLM Prompt、Provider 配置和 dependency lock；
- 把模型角色、Prompt/config hash、预算和三态门写入 `kir-pilot-v1-experiment.yaml`；
- 预登记 Gemini primary、DeepSeek weak 与 OpenAI cross-provider-reference；只有 primary 有全局 verdict
  authority，后两者只做各自模型内的 `B3-B2b` 复现；
- 先完整运行 dev sanity check，再对 frozen test 执行版本化 run；
- `run --split test` 必须先验证 `freeze-manifest.json` 与 experiment lock 的全部 hash，任一缺失或不匹配就拒绝；
- 报告 B2b/B3 实际总 token 差异；超过 10% 时自动标记 `budget-confounded`。

**Exit gate IE3**

- B0/B1 各有一组 manifest；primary 的 B2a/B2b/B3 与两个复现角色的 B2b/B3 共七格都有 manifest 与
  prediction；
- 每个 case 的失败都显式记录，不因 parse/provider failure 静默丢弃；
- seven-run matrix checker、所有 B2b call-1 的 one-stage-compatible cache provenance 和双 freeze guard
  均通过；secondary 的 call-1 cache 不另计正式 B2a scoring arm；
- B2b/B3 预算合同可机器校验；
- 原始 test、Gold、Prompt 和 primary model 在看到结果后没有被修改。

### IE4：统计结论、报告与求职交付（第 7～9 天）

#### IE4.1 自动报告

- 输出 metrics、95% paired cluster-bootstrap CI、confusion、badcases、latency/token/cost；
- 对预冻结的 24 条 slice，在 B3 三个 LLM 角色上按 `action/object/horizon` rubric 记录
  `preserved/partial/lost`，共 72 行，生成
  `semantic-audit.csv`；该结果只作诊断，不进入三态 verdict，也不用于回调 Prompt；
- primary、weak 与 cross-provider-reference 分别报告，不平均、不投票；
- evaluator 按有效性 → negative → statistical inconclusive → promising 的冻结顺序生成结论；
- 所有表格由结构化结果生成，README 不手填与缓存不一致的数字。

#### IE4.2 面试材料

- README 首屏说明问题、边界、方法、关键数字、复现命令和真实限制；
- 生成 5 分钟与 15 分钟项目讲法；
- 只有完成实验后才填写简历 bullet 中的数字；
- negative/inconclusive 也如实解释其工程价值，不重采样或改 test 追求正结果。

**Exit gate IE4 / Pilot Definition of Done**

- fresh clone 能安装并执行离线门检；
- 使用已缓存 prediction 能重算完全相同的结构化指标和 verdict；
- 160-case 数据、五个 baseline、三 LLM 角色、CI、badcase、延迟与成本均有可审计产物；
- 24-case × 三个 LLM 角色的 semantic-preservation audit 有冻结 IDs、rubric 和 72 行记录；
- 仓库不包含 secret、生产数据或对 Kindred 的运行时依赖；
- 报告明确说明 synthetic、balanced、non-blind portfolio pilot 的外推限制。

## 5. 测试策略

| 测试层 | 必测内容 |
|---|---|
| Schema unit | 四类字段真值表、evidence 子串、near-OOS sibling、未知 target、重复 id、版本错误 |
| Metric unit | invalid/missing、固定 label、缺类、HEM、Macro-F1、OOS、ambiguous、切片指标 |
| Verdict unit | 每个 CI 边界、token 10% 分母/失败调用、最少 cluster、缺字段和合同失效 |
| Split/bootstrap unit | 关系图连通分量、source 非 group key、paired draws、单 cluster 与跨 split 泄漏 |
| Freeze guard | dataset/experiment lock 缺失、hash mismatch 和合法 test run |
| Adapter contract | 调用次数、预算、重试、repair、candidate order 与错误映射 |
| Cache integration | 内容/taxonomy/Prompt/config 任一 hash 变化导致 miss；B2b call-1 provenance 指向 B2a |
| Matrix integration | primary 三臂 + 两个复现角色各两臂，共七格齐全；缺一格不得生成正式报告 |
| Report snapshot | 指标、confusion、badcase 和 manifest 引用一致 |
| Privacy check | fixtures/results 不包含已知 secret pattern 或 Kindred 生产路径内容 |

Provider live tests 默认不进入普通 CI；使用显式 marker 手动运行，避免 CI 消耗费用或因外部波动失败。

## 6. 提交与交付节奏

建议每个提交都可独立通过离线门检：

1. `chore: bootstrap python workbench`
2. `feat: define taxonomy and evaluation schemas`
3. `data: add annotated pilot dataset`
4. `feat: implement deterministic evaluator`
5. `feat: add rule and embedding baselines`
6. `feat: add llm runner and one-stage baseline`
7. `feat: add matched two-call ablation`
8. `report: publish frozen pilot results`

不要把 160 条数据、全部 adapter 和最终报告压成一个不可审查的大提交。只有第 8 个提交允许引用正式 test
结果；前七个提交的调试输出只能来自 fixtures 或 dev。

## 7. 风险与应对

| 风险 | 预防与处理 |
|---|---|
| 单人编写 test 导致记忆泄漏 | 明确 non-blind；先冻结 hash；结果不包装成无偏 benchmark |
| 160 条使 CI 太宽 | 按合同判 inconclusive；不临时改门槛；另开 power-analysis proposal 才能扩样 |
| B1 得到更多人工监督 | 限制 prototype 来源/数量并报告 case id；所有方法只用同一 dev pool |
| 双阶段收益来自更多 token | B2b 双调用对照；记录真实 token；差异超过 10% 判 budget-confounded |
| Provider 响应不可复现 | 固定各 Provider 版本/参数/hash并缓存 normalized response；只声称 evaluator 可复现 |
| 生成数据过于模板化 | 使用 scenario/contrast/paraphrase cluster，做泄漏和 slice coverage 检查 |
| 项目膨胀成平台 | Pilot 只实现五个 baseline 与离线报告；IE5、训练、Shadow、dashboard 全部后置 |
| 9 天排期没有余量 | 对外目标 9 天，计划预留 2 天风险缓冲；不以删除公平对照换进度 |

仓库目前是 private，因此 IE0 不仓促选择 LICENSE。准备公开作品集前必须完成数据/依赖许可复核，再选择
明确 LICENSE；未完成前保持 private，不把“没有许可证”误解成默认允许复用。

## 8. IE5 触发条件

只有 primary model 的 Pilot verdict 为 `promising`，才进入 IE5：

- 新增 80 条至 240；
- 至少 25% 独立双标，或诚实报告 test-retest；
- 扩大 cluster 后继续 bootstrap；
- 增加重复稳定性和更完整弱模型矩阵；
- 重新应用 IE-V1 的生产讨论门。

IE5 仍不修改 Kindred。sampled Shadow 必须另写生产 proposal，并单独获得授权。

## 9. 第一实施切片

计划获批后，只启动 **IE0**，不同时开始造 160 条数据或运行效果实验。只允许每个预选模型各一次
dev-fixture readiness smoke。第一实施切片指“第一批必须一起落地并通过 Exit Gate 的文件集合”，完成物是：

```text
pyproject.toml
uv.lock
.gitignore
.github/workflows/ci.yml
README.md
src/intentbench/{cli,schemas,taxonomy,b1,evaluator,metrics,bootstrap,verdict,freeze,providers}.py
src/intentbench/adapters/{base,google,deepseek,openai}.py
configs/{kindred-activity-intents-v1.yaml,kir-pilot-v1-experiment.yaml,provider-readiness.json}
tests/unit/
tests/fixtures/provider-smoke.json
docs/annotation-guideline.md（骨架）
GitHub Actions 离线门检
```

IE0 通过后再进入 IE1。这样可以在 Gold 编写前发现 schema、标签边界和 verdict 合同问题，避免后续批量返工。
