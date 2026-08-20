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

预注册模型以 Kindred/OpenClaw `agent:main:main` 运行时元数据为权威：primary 是
`gemini-3.6-flash`，weak 是同 Provider 的 `gemini-3.5-flash`，embedding 是
`gemini-embedding-001`。Kindred 可用的 Grok、DeepSeek、OpenAI 候选只登记、不在 IE0 扩成模型横评。

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

下面是唯一会访问网络的 IE0 命令，只能使用合成 fixture：

```bash
GEMINI_API_KEY=... uv run intentbench providers check \
  --fixture tests/fixtures/provider-smoke.json \
  --config configs/kir-pilot-v1-experiment.yaml \
  --output configs/provider-readiness.json \
  --environment-label user-provided-us-linux
```

已提交的 readiness 记录来自用户授权的美国 Linux 环境，仅包含 identity、usage、延迟、向量维度与 raw
response SHA-256；不包含 key 或响应正文。单条 fixture 命中只表示接口就绪，不是效果结果。

完整任务定义见 [设计文档](docs/design.md)，阶段与 Exit Gate 见
[实施计划](docs/implementation-plan.md)，IE1 标注前置规则见
[标注规范骨架](docs/annotation-guideline.md)。
