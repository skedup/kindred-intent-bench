# Kindred Intent Bench

Kindred Intent Bench 是一个离线优先的 open-set intent recognition Workbench，用于评测 stateful Agent
在输入已经包含可观察意图证据时，能否把下一行动正确路由到 Activity taxonomy，同时显式处理
`oos / no_intent / ambiguous`。

它不替 Kindred 决定“应该想做什么”，也不把自主选择分布是否均匀当作正确性指标。当前 IE0 只建立评测
合同与工程门禁：没有编写 160-case Gold、没有运行 frozen test，也没有接入 Kindred 生产路径。

## IE0 已实现

- 8-intent 版本化 taxonomy：包括 `play_xiaohongshu` 等小众 Activity；
- Pydantic Gold / Prediction / RunManifest 合同与四类输出真值表；
- 完整 Gold universe evaluator：missing、Provider failure、schema invalid 均计错，duplicate/extra 使 run 无效；
- HEM、固定标签 Macro-F1、OOS/no-intent/ambiguous、near-OOS 与诊断计数；
- 关系图连通分量、paired cluster bootstrap、固定 Type-7 percentile CI；
- `promising / inconclusive / negative` 预注册确定函数；
- dataset freeze + experiment lock 双冻结 guard；
- B1 embedding prototype/actionability/OOS/ambiguity 公式与确定性阈值 tie-break；
- synthetic-only Provider readiness，不保存密钥或原始响应。

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
uv run intentbench taxonomy validate configs/kindred-activity-intents-v1.yaml
```

以上命令不访问 LLM/embedding Provider。正式 test runner 后续必须通过双冻结 guard；当前 experiment
仍是 `draft`，这是 IE0 的预期状态。

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
[实施计划](docs/implementation-plan.md)，IE1 标注前置规则见
[标注规范骨架](docs/annotation-guideline.md)。
