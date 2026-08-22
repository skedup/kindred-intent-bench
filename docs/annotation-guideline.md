# KIR Pilot 标注规范（IE1.1）

> Status: `approved_for_ie1_authoring / not a frozen dataset`
>
> 适用版本：`kir-pilot-v2` 数据设计、`kindred-activity-intents-v2` taxonomy
>
> 机器可校验示例：`kir-pilot-annotation-v3`
> [kir-pilot-v2-annotation-pack.yaml](../configs/kir-pilot-v2-annotation-pack.yaml)
>
> Activity 边界绑定 [kindred-activity-grounding-v1.yaml](../configs/kindred-activity-grounding-v1.yaml)；
> 先查运行时 Activity/Action 合同，只有合同缺失或冲突才升级产品策略。
>
> 本规范和示例包不是 160 条正式 Gold，不包含 test split，也不授权生成 prediction 或调用 Provider。

## 1. 标注目标与边界

标注输入中**已经可观察到的、现在或下一段生活的主要行动意图**，判断它能否由当前 Activity taxonomy
承接。标注者不是替 Kindred 决定“应该做什么”，也不以 Activity 分布更均匀为目标。

只使用以下证据：`state_summary` 的明确表达、`conversation` 中已说出的内容，以及人工合成的中性上下文。
天气、时间、疲劳、饥饿、最近 Activity 和标注者偏好只能作为上下文，不能单独推出行动。严禁复制生产
State、Thought 或真实用户消息；正式数据只能是人工创作或人工审核后的合成场景。

标注顺序固定为：先独立读输入并选 `decision`，再按真值表填 `target_intent`、`evidence_quote` 和 slots，
最后补关系 ID、来源与边界 tags。不得查看或借用任何模型 prediction、置信度或实验结论来决定 Gold。

## 2. 四类 decision 的可执行决策树

这棵树是给 **Gold 标注者、独立复核者和仲裁者** 使用的人工 SOP，不是运行时 router，也不会由离线
validator 自动执行。`oos` 最终仍由人依据输入 evidence 和冻结 taxonomy 标记；机器只检查字段组合与覆盖合同。

```text
输入是否表达现在或下一段生活的行动倾向？
  否 -> no_intent
  是 -> 能否从输入确定唯一的下一主要行动？
    否（动作不具体，或多个候选无主次） -> ambiguous
    是 -> taxonomy 中是否有一个 Activity 能完整承接这个单一具体行动？
      是 -> in_scope + 唯一 target_intent
      否 -> oos
```

| decision | 必要条件 | 不能误用为 |
|---|---|---|
| `in_scope` | 行动明确，且一个现有 intent 能完整承接下一主要行动 | “大概相似”或为了避免 OOS 的强行映射 |
| `oos` | 行动具体、可区分，但冻结 taxonomy 没有完整承接者 | 没有行动、动作含糊或多个候选难分 |
| `no_intent` | 没有当前/下一行动倾向，或明确不打算行动 | 低行动的 `rest`，或具体但目录外的行动 |
| `ambiguous` | 已有行动倾向，但动作不够具体，或多个候选没有主次 | 所有困难样本的兜底标签 |

补充规则：

- 有明确的“先 A、再 B”，只标下一主要行动 A；没有主次且两个以上候选都成立，标 `ambiguous`。
- 远期愿望若没有当前/下一行动安排，标 `no_intent`，slots 可保留 `horizon=later`。
- 一个明确动作即使和目录内 Activity 很相似，只要关键动作或对象不满足定义，仍标 `oos`。
- 宽泛的“做点轻松的事”“出去一下”说明有行动倾向但不够具体，通常是 `ambiguous`，不是
  `no_intent` 或 `oos`。

### 2.1 `in_scope`：正例、反例、仲裁例

| 示例 ID | 类型 | 关键输入 | Gold | 判定依据 |
|---|---|---|---|---|
| `guide-in-scope-positive-01` | 正例 | 现在想去河边慢慢走一圈 | `in_scope/take_a_walk` | 动作和目的唯一，taxonomy 完整承接 |
| `guide-in-scope-positive-02` | 正例 | 随手看看别人最近分享的日常 | `in_scope/play_xiaohongshu` | 不依赖品牌词，浏览公开生活分享即可命中 |
| `guide-in-scope-positive-03` | 正例 | 背景是下雨；随后说在家煮面 | `in_scope/eat_at_home` | 明确行动覆盖天气干扰，地点排除外出用餐 |
| `guide-in-scope-negative-01` | 反例 | 今天天气很舒服 | `no_intent` | 状态事实不是散步行动 |
| `guide-in-scope-negative-02` | 反例 | 戴耳机听一张专辑 | `oos` | 行动具体，但目录没有音乐播放 |
| `guide-in-scope-negative-03` | 反例 | 散步或找家店坐坐都可以 | `ambiguous` | 两个可行 intent 无主次 |
| `guide-in-scope-adjudication-01` | 仲裁 | 先下楼走十分钟，回来再吃饭 | `in_scope/take_a_walk` | “先”给出唯一下一行动 |
| `guide-in-scope-adjudication-02` | 仲裁 | 有点累，现在先靠着休息十分钟 | `in_scope/rest` | 疲惫只是背景，后半句是明确选择 |
| `guide-in-scope-adjudication-03` | 仲裁 | 刚画完画；随后说给当前用户发消息 | `in_scope/reach_out_to_user` | 最近 Activity 不可复制，当前动作优先 |

### 2.2 `oos`：正例、反例、仲裁例

| 示例 ID | 类型 | 关键输入 | Gold | 判定依据 |
|---|---|---|---|---|
| `guide-oos-positive-01` | 正例 | 打开音乐安静听一会儿 | `oos` | 具体动作不在目录 |
| `guide-oos-positive-02` | 正例 | 骑自行车沿湖转一圈 | `oos` | 骑行与散步相邻，但不是步行 |
| `guide-oos-positive-03` | 正例 | 打开小红书私信查看照片 | `oos` | 运行时合同明确排除私信 |
| `guide-oos-negative-01` | 反例 | 下楼散散步 | `in_scope/take_a_walk` | 已有完整承接者 |
| `guide-oos-negative-02` | 反例 | 还没有具体想做的事 | `no_intent` | 不存在目录外的具体行动 |
| `guide-oos-negative-03` | 反例 | 想做轻松的事但没决定是什么 | `ambiguous` | 有倾向但不具体 |
| `guide-oos-adjudication-01` | 仲裁 | 裁剪、调色已有照片，不创作新图 | `oos` | 编辑既有内容不满足“创作新视觉作品” |
| `guide-oos-adjudication-02` | 仲裁 | 给一位老朋友发消息 | `oos` | `reach_out_to_user` 不泛化为任意第三方 |
| `guide-oos-adjudication-03` | 仲裁 | 在家找个线上展览，具体哪个都行 | `oos` | 行动类型足够具体；未指定展览不改变目录外结论 |

### 2.3 `no_intent`：正例、反例、仲裁例

| 示例 ID | 类型 | 关键输入 | Gold | 判定依据 |
|---|---|---|---|---|
| `guide-no-intent-positive-01` | 正例 | 没有具体想做的事情 | `no_intent` | 明确否定当前行动意图 |
| `guide-no-intent-positive-02` | 正例 | 阳光很好，屋里很安静 | `no_intent` | 环境事实不授权推断行动 |
| `guide-no-intent-positive-03` | 正例 | 也许下个月看展，现在没有安排 | `no_intent` | 只有远期愿望 |
| `guide-no-intent-negative-01` | 反例 | 现在闭眼休息 | `in_scope/rest` | 已主动选择休息 |
| `guide-no-intent-negative-02` | 反例 | 现在听半小时播客 | `oos` | 具体动作只是目录外 |
| `guide-no-intent-negative-03` | 反例 | 散步和看展还没决定 | `ambiguous` | 有行动倾向但候选难分 |
| `guide-no-intent-adjudication-01` | 仲裁 | 想听音乐，但不打算打开也不准备做什么 | `no_intent` | 目录外动作被明确否定，不能标 OOS |
| `guide-no-intent-adjudication-02` | 仲裁 | 周末也许做点什么，现在先不安排 | `no_intent` | 宽泛 later 愿望被推迟，不是当前 ambiguous |
| `guide-no-intent-adjudication-03` | 仲裁 | 刚看完公开帖子，接下来没想好 | `no_intent` | recent activity 不是下一行动 |

### 2.4 `ambiguous`：正例、反例、仲裁例

| 示例 ID | 类型 | 关键输入 | Gold | 判定依据 |
|---|---|---|---|---|
| `guide-ambiguous-positive-01` | 正例 | 可能散步，也可能找咖啡馆 | `ambiguous` | 两个目录内行动同等可接受 |
| `guide-ambiguous-positive-02` | 正例 | 想躺下或弄吃的，还没决定 | `ambiguous` | `rest/eat_at_home` 都有信号但无主次 |
| `guide-ambiguous-positive-03` | 正例 | 看图片或自己画一张，先哪个都行 | `ambiguous` | 浏览与创作不可唯一选择 |
| `guide-ambiguous-negative-01` | 反例 | 先散步，回来再看展览信息 | `in_scope/take_a_walk` | 明确先后消除歧义 |
| `guide-ambiguous-negative-02` | 反例 | 没有特别想做的 | `no_intent` | 没有行动倾向 |
| `guide-ambiguous-negative-03` | 反例 | 整理房间 | `oos` | 动作具体且唯一，只是目录外 |
| `guide-ambiguous-adjudication-01` | 仲裁 | 出去走走或者去书店，哪个都行 | `ambiguous` | “出去”不是可路由的共同上位 intent |
| `guide-ambiguous-adjudication-02` | 仲裁 | 在家吃东西或躺下，没想好先哪个 | `ambiguous` | 状态强弱不能替代主次证据 |
| `guide-ambiguous-adjudication-03` | 仲裁 | 助手给出画画/浏览；用户说都行、未决定 | `ambiguous` | 不按候选顺序或 recent activity 代选 |

以上 36 条的完整 `Context`、`Gold`、竞争标签和 rationale 以机器示例包为准。它们使用 `guide-*` ID，
没有 case/split 字段，不能被数据集 loader 当作正式 Gold。

## 3. 八个 intent 的 inclusion / exclusion 边界

| intent | Inclusion（应命中） | Exclusion（应离开该 intent） | 核心边界 |
|---|---|---|---|
| `create_picture` | 画雨夜街景；把灵感做成新插画 | 浏览摄影作品→`play_xiaohongshu`；裁剪已有照片→`oos` | 必须创作新的视觉作品 |
| `dine_out` | 去餐馆吃晚饭；找咖啡馆喝东西 | 在家点外卖→`eat_at_home`；只下楼走走→`take_a_walk` | 离开住处，以餐饮场所用餐为主要目的 |
| `eat_at_home` | 在厨房煮面；留在家点外卖 | 去小馆吃面→`dine_out`；只有饥饿事实→`no_intent` | 在当前住处准备、订购或享用食物 |
| `play_xiaohongshu` | 浏览/搜索公开帖子；公开互动；创作并发布公开图文 | 自己只画新图→`create_picture`；查看或发送私信→`oos` | 公开内容消费、互动与发布均由运行时 Activity 承接，私信除外 |
| `reach_out_to_user` | 给当前用户发消息；给当前用户写晚安 | 想念但不打扰→`no_intent`；浏览陌生人近况→`play_xiaohongshu` | 联系对象必须是当前用户 |
| `rest` | 闭眼睡一会；靠着放松十分钟 | 只有疲惫事实→`no_intent`；在家吃东西→`eat_at_home` | 必须主动选择暂停、睡眠或安静放松 |
| `take_a_walk` | 沿河步行；下楼散步透气 | 走去餐馆→`dine_out`；步行去博物馆→`visit_cultural_place` | 步行本身是目的，没有更主要的目的地任务 |
| `visit_cultural_place` | 去美术馆看展；去独立书店体验 | 无目的地走走→`take_a_walk`；看线上展览→`oos` | 必须前往线下文化场所 |

当前机器包版本中每个 intent **恰好**固定两条 inclusion 和两条 exclusion；其中 exclusion 可以落到另一个 intent、`oos`、
`no_intent` 或 `ambiguous`，但绝不能仍落回正在排除的 intent。

## 4. near-OOS 与 sibling 配对

near-OOS 是“已经具体，但只差一个决定性语义条件就会命中某个 intent”的 OOS。每条正式 near-OOS 必须：

1. `decision=oos`、`target_intent=null`、tag 含 `near_oos`；
2. `near_oos_sibling_intents` 至少包含一个 taxonomy intent；
3. 与对应 in-scope sibling case 共用 `contrast_group_id`，确保不跨 dev/test；
4. 成对文本只改动必要语义，不靠替换品牌词或增加无关背景制造难度。

| 配对 ID | sibling intent | OOS 一侧 | in-scope 一侧 | 决定性差异 |
|---|---|---|---|---|
| `near-oos-create-picture-edit` | `create_picture` | 编辑已有照片 | 创作新插画 | 编辑既有图像 vs 新视觉创作 |
| `near-oos-dine-out-live-music` | `dine_out` | 去酒吧只听乐队、不吃喝 | 去咖啡馆喝咖啡 | 非餐饮目的 vs 餐饮目的 |
| `near-oos-eat-home-kitchen-task` | `eat_at_home` | 在家整理食谱、不准备吃 | 在家做晚饭 | 厨房相关任务 vs 实际准备/享用食物 |
| `near-oos-xhs-private-message` | `play_xiaohongshu` | 查看小红书私信照片 | 查看博主公开照片 | 私信内容 vs 公开内容 |
| `near-oos-reach-user-friend` | `reach_out_to_user` | 给老朋友发消息 | 给当前用户发消息 | 第三方 vs 当前用户 |
| `near-oos-rest-bath` | `rest` | 洗热水澡放松 | 什么也不做、靠着休息 | 具体目录外活动 vs 暂停行动 |
| `near-oos-walk-cycle` | `take_a_walk` | 沿湖骑车 | 沿湖步行 | 骑行 vs 步行 |
| `near-oos-cultural-online` | `visit_cultural_place` | 在家看线上展览 | 去美术馆看线下展览 | 线上观看 vs 前往线下场所 |

far-OOS 则与现有 intent 没有需要特别防止的近邻混淆，例如整理房间或听播客；它不填写 sibling intent。

## 5. 六个必测边界

### 5.1 Multi-turn（`multi_turn`）

- 把 `state_summary` 和完整对话视为同一输入，但 `evidence_quote` 必须完整落在其中一个 carrier，不能拼接。
- 后续明确选择可以覆盖较早的候选或背景，例如下雨后说“那就在家煮面”应标 `eat_at_home`。
- 助手提出候选不等于用户或 Kindred 已选择；“两个都行，还没决定”必须 abstain 为 `ambiguous`。
- recent activity 或前一轮已完成动作不能自动延续；“刚画完，现在给你发消息”标联系用户。
- 多轮样本的用户承接话术不得由 Gold decision 选择；重复出现的用户话术必须跨 decision 复用，避免标签泄漏。
- 正式样本以 Kindred 最后一轮承载可审计 evidence；覆盖、纠正、撤回和指代等对话骨架应有明确记录，
  相同骨架通过 `scenario_family_id` 归组，不能在 bootstrap 中伪装成独立样本。
- Pilot 将40条样本均分为 override、recency、resolution、reconsideration 四种 pattern，每种10条；每条
  multi-turn 必须且只能带一个对应子标签。

### 5.2 Context distractor（`context_distractor`）

- 天气、时间、位置、疲惫、饥饿和 recent activity 都不能覆盖当前明确行动。
- 只有背景事实且没有行动表达时标 `no_intent`，不能按常识推测散步、吃饭或休息。
- 编写 hard negative 时只放一个有意义的 distractor，避免用大量噪声人为增加阅读难度。
- 每条正式 `context_distractor` 必须有一条 `context_control`，两者共享 scenario/contrast ID、Gold、evidence、
  slots、turn shape 和对话，只允许在背景事实或 recent activity 上出现一个受控差异。
- pair 两侧必须共享同一个 `context_perturbation`：登记稳定 ID、添加到 `state_summary` 的背景前缀和新增的
  `recent_activities`。distractor 必须精确等于把该 delta 应用到 control，不能额外改写前景行动。
- `context_distractor` 与 `context_control` 各 24 条，single/multi 各占一半；报告配对 prediction flip、
  adverse flip 和 error delta，不能只凭困难样本错误率宣称因果一致性。

### 5.3 Brandless XHS（`brandless_xhs`）

- 不要求出现“小红书”品牌名；浏览/搜索公开生活内容、公开互动、创作并发布公开生活图文可标
  `play_xiaohongshu`。
- 不能只凭“看看”“图片”“分享”命中：只创作新图属于 `create_picture`，私信内容为 `oos`，
  主动联系特定对象不属于浏览。
- 正式 brandless case 的意图证据中不得出现“小红书”，否则不能用于验证语义泛化。

### 5.4 Rest / Eat（`rest_eat_confusion`）

- “累”“饿”只是状态；没有行动倾向时是 `no_intent`。
- “闭眼休息”“在家煮面”是明确行动，分别命中 `rest` / `eat_at_home`。
- 两者都被明确提出但没有顺序时为 `ambiguous`；出现“先”时只标下一行动。
- 不能以状态更强、时间更合理或标注者认为更健康为由替 Kindred 排序。
- Pilot 恰好包含 `rest=4 / eat_at_home=4 / no_intent=4 / ambiguous=4`，并平衡为 8 条 single-turn 与
  8 条 multi-turn，避免把 turn shape 或 context distractor 当作 rest/eat 能力。

### 5.5 Hard negative（`hard_negative`）

- hard negative 必须“表面接近一个或多个已知 intent，但 Gold 不是 `in_scope`”；宽泛但没有明确竞争 intent 的
  ambiguous 话语不属于 hard negative。
- 每条必须填写 `hard_negative_against`，列出模型可能被诱导命中的 taxonomy intent；near-OOS 时该字段必须
  与 `near_oos_sibling_intents` 一致，ambiguous 时至少包含两个竞争 intent。
- Pilot 构成固定为 `near-OOS 16 + no_intent 16 + ambiguous 8`，不能仅为凑数给普通负例重贴标签。

### 5.6 Hypothesized weak-model probe（`hypothesized_weak_model_probe`）

- 这是看 test prediction 前冻结的**假设性探针**，不是已经证明的弱模型错题。
- Pilot 恰好是 `near_oos ∪ brandless_xhs ∪ rest_eat_confusion`，共 40 条；它只汇总三个已定义风险面，
  不作为独立于这些切片的额外效果证据。
- 运行后稳定失败的样本只能进入单独的 `empirical_weak_model_failure` badcase 产物；不得回写 frozen Gold tags，
  也不得据此调换 test case。

## 6. Evidence、slots 与 Gold 真值表

- `gold.evidence_quote` 必须是 NFKC 规范化后 `state_summary` 或某条 conversation content 的连续子串；
  它只用于 Gold 审计，绝不传给 baseline。
- quote 选能支撑 decision 的最短完整片段。`no_intent` 应引用否定/无计划表达；纯状态场景可引用该状态事实，
  但 annotation note 要说明为什么不能推出行动。
- `in_scope` 必须有唯一 `target_intent`；另外三类的 target 必须为 `null`。
- 只有 tagged near-OOS 可以填写 `near_oos_sibling_intents`；普通 OOS 和其他 decision 必须为空。
- 只有 tagged hard-negative 可以填写 `hard_negative_against`；其中 intent 必须来自冻结 taxonomy。
- 只有 context-control/distractor 可以填写 `context_perturbation`，且 pair 两侧必须完全相同；普通 case 填 `null`。
- `desired_experience` 与 `object` 只抄录或短语化输入可支持的内容，不补写隐含偏好；未知填 `null`。
- `horizon=now` 表示现在/下一步，`later` 表示明确远期，未提供时间则为 `unspecified`。horizon 不独立决定
  decision：远期具体计划也需依本规范判断是否属于当前 routing 范围。
- `annotation_note` 简述决定性边界，不写模型应模仿的长推理。

## 7. 关系 ID、来源与泄漏控制

每条 case 必须填写以下字段：

| 字段 | 规则 |
|---|---|
| `scenario_family_id` | 相同人物状态、地点、目标或叙事骨架的变体共用；只换行动也仍可能是同一 family |
| `contrast_group_id` | 最小语义对照、near-OOS/sibling、decision 边界对照共用；无真实对照时使用该 case 自己的稳定组 ID |
| `paraphrase_cluster_id` | 保持标签与语义不变的改写共用；语义条件改变后不能继续当 paraphrase |
| `source` | `human_authored` 为人工起草；`llm_assisted_human_reviewed` 必须记录辅助来源并经人工独立定标；`synthetic_fixture` 只用于测试代码，不进入正式 Gold |

共享前三种任一关系 ID 的 case 在图上连边；IE1.3 对完整连通分量生成相同 `split_group_id`，并令
`bootstrap_cluster_id=split_group_id`。DraftCase 编写阶段不包含 `split` 或这两个生成字段；IE1.3 materialization
时才同时加入，禁止提前手填或为了满足 split 比例拆组。
`source` 只作 provenance/分层，不作为 group key。dev/test 只能按完整 group 分配。

## 8. 标注、自检、复核与仲裁

状态只允许按下列方向推进：

```text
draft（起草并自检） -> reviewed（独立复核一致）
                   \-> adjudicated（发生分歧且已裁决）
                   \-> 排除（无法形成唯一可审计 Gold）
```

标准流程：

1. 标注者独立填写 decision、target、evidence 和简短理由，不看 prediction。
2. 复核者先独立判断，再比较标签；分歧先归因为行动性、具体性、目录覆盖、对象边界或主次顺序。
3. 只依据冻结 taxonomy、本规范和输入 evidence 裁决，不依据期望模型效果。
4. 无法得到唯一、可审计 Gold 时修改或排除样本；不引入 `acceptable_intents` 集合掩盖歧义。
5. 记录原提案、理由、裁决人与 resolution，完成后才标 `adjudicated`。

分歧记录至少包含这些机器字段：`case_id`、`annotator_id`、`proposed_label`、`evidence_quote`、
`rationale`、`resolution`、`adjudicator_id`。只有一名标注者时，至少间隔 3 天盲重标；这种复核不能冒充
inter-annotator agreement，必须在 dataset card 披露。未解决分歧一律 `exclude_from_freeze`。

IE1.2 的 initial pass 必须使用 `dataset reviews` 生成的 label-blind workspace：只展示随机 review item ID 与
context，先锁定 decision、target、evidence 和 slots，再揭示候选 Gold 做 tags、关系 ID 与分歧仲裁。不得在
initial response 写入前查看一致性报告或模型 prediction。每个 workspace 绑定 candidate/taxonomy hash；hash
变化后不得沿用旧响应。

机器包使用完整的 `competing_labels = {decision, target_intent}` 记录候选标签，而不是只记录一级 decision：
这使仲裁既能表达 `oos` 与 `ambiguous` 的拒识分歧，也能表达 `dine_out` 与 `take_a_walk` 这种同为
`in_scope` 的 intent 分歧。12 条仲裁 fixture 必须覆盖六种不同 decision 两两组合，并至少包含一组
in-scope intent 间的竞争；最终 Gold 必须是 competing labels 之一。

## 9. Dataset card 的强制限制声明

IE1.3 冻结时，dataset card 必须明确写出以下七项；括号中的机器标识不可省略：

1. 数据是为覆盖标签而构造的合成、平衡样本，不代表真实流量先验（`synthetic_balanced_dataset`）。
2. test 是 non-blind frozen test，仓库维护者可读，不等同第三方盲测（`non_blind_frozen_test`）。
3. 没有使用 Kindred 生产 State、Thought 或用户消息（`no_production_data`）。
4. 单标注者/弱复核限制及具体重检方式（`single_annotator_limit`）。
5. LLM 是否参与场景起草，以及哪些环节由人工独立审核（`llm_assistance_provenance`）。
6. 结论只适用于当前 8-intent taxonomy、合成语体和已测模型，不能外推生产效果（`external_validity_limit`）。
7. 谁可访问 test，以及 test 不得用于 prompt、few-shot、prototype、阈值或规则调优
   （`test_access_and_tuning_boundary`）。

## 10. 编写模板与离线校验

复制 [kir-pilot-v1-case.template.json](../templates/kir-pilot-v1-case.template.json) 后再起草。模板本身带
`non_dataset_case_authoring_template` wrapper，并使用不含 `split/split_group_id/bootstrap_cluster_id` 的
`DraftCase`；即使抽出内部 `case`，也不能通过正式 `Case` schema。必须替换 `draft-template-*` ID 和全部
`replace-*` 字段，且不得直接编辑模板原件来累积数据。

在任何批量 Gold 编写前运行：

```bash
uv run intentbench annotations validate
```

该命令完全离线，并同时校验：四类 decision 的正/反/仲裁覆盖、8-intent inclusion/exclusion、8 组
near-OOS、四个必测 tag、brandless 与 multi-turn 约束、仲裁合同、dataset-card 披露项、taxonomy 对齐和
非数据集模板。校验成功只说明 IE1.1 标注合同自洽，不代表正式 Gold 已经产生或冻结。

## 11. IE1.1 Exit Gate

- [x] 四类 decision 各 3 个正例、3 个反例、3 个仲裁例，共 36 条 guideline fixtures；
- [x] 8 个 intent 在当前 pack 各有恰好 2 个 inclusion / 2 个 exclusion 对照；
- [x] 每个 intent 有一组 near-OOS/sibling 配对；
- [x] multi-turn、context-distractor、brandless XHS、rest/eat confusion 均有规则与机器示例；
- [x] 分歧字段、仲裁流程、单标注者重检限制和 dataset-card 七项披露已固定；
- [x] 非数据集编写模板、离线 validator 与自动测试可用；
- [ ] 160 条正式 Gold、split 与 freeze 属于 IE1.2/IE1.3，本阶段没有编写；
- [ ] test prediction 和 LLM/embedding Provider 调用不属于本阶段，禁止提前执行。
