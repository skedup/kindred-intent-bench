"""Build the deterministic IE1.2 candidate review queue without Provider calls."""

# ruff: noqa: RUF001 -- full-width punctuation is intentional in Chinese fixtures.

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from intentbench.annotation import DraftCase
from intentbench.dataset import validate_candidate_cases
from intentbench.schemas import Context, Decision, Gold, Horizon, Slots
from intentbench.taxonomy import load_taxonomy

ROOT = Path(__file__).parents[1]
OUTPUT_PATH = ROOT / "data/kir-pilot-v1/candidates.jsonl"
TAXONOMY_PATH = ROOT / "configs/kindred-activity-intents-v1.yaml"

INTENTS = (
    "create_picture",
    "dine_out",
    "eat_at_home",
    "play_xiaohongshu",
    "reach_out_to_user",
    "rest",
    "take_a_walk",
    "visit_cultural_place",
)

IN_SCOPE_TEXTS: dict[str, tuple[str, ...]] = {
    "create_picture": (
        "现在想照着窗外的雨景重新画一幅全新的插画。",
        "想把脑海里的机器人做成一张新的海报。",
        "我先画一张月光下的海面。",
        "准备给故事里的狐狸画一幅新肖像。",
        "想用几何形状创作一张全新的抽象画。",
        "接下来把刚才的梦画成一幅新图。",
        "现在做一张秋日森林的原创插画。",
        "想从空白画布开始画一间未来厨房。",
        "先生成一张没见过的云端城市图。",
        "接下来创作一幅全新的天空城市画面。",
    ),
    "dine_out": (
        "现在去附近咖啡馆点杯热饮坐一会儿。",
        "想出门到巷口的小馆吃一碗面。",
        "今晚去那家素食餐厅吃饭。",
        "准备到楼下早餐店买现做的早点并在那里吃。",
        "想去河边的餐馆尝尝晚餐。",
        "接下来找家茶馆点壶茶和小点心。",
        "现在出门去面包房的座位区吃点东西。",
        "想约在外面的甜品店吃一份蛋糕。",
        "先去街角咖啡馆喝杯拿铁。",
        "接下来到街角那家咖啡店喝一杯咖啡。",
    ),
    "eat_at_home": (
        "现在去厨房煮一锅番茄汤在家吃。",
        "今晚留在家里点一份外卖当晚饭。",
        "想用冰箱里的蔬菜做一顿家常饭。",
        "接下来在家热一碗粥吃。",
        "准备烤两片面包留在屋里当早餐。",
        "现在想在厨房包几个饺子吃。",
        "不出门了，在家下单一份沙拉。",
        "想把剩饭炒热，在餐桌旁吃掉。",
        "先在家给自己煮一碗清汤面。",
        "接下来留在屋里做碗简单的汤面吃。",
    ),
    "play_xiaohongshu": (
        "现在想翻翻别人公开分享的周末日常。",
        "想浏览大家最近发布的穿搭记录。",
        "接下来看看别人公开分享的旅行照片。",
        "想随手刷一会儿公开的生活方式帖子。",
        "现在看看大家分享的居家布置灵感。",
        "想浏览陌生人公开发布的探店笔记。",
        "接下来翻翻别人最近晒出的植物养护经验。",
        "想看看公开社区里大家都在分享什么。",
        "现在刷一会儿小红书里的生活帖子。",
        "接下来打开小红书看看别人发布的日常。",
    ),
    "reach_out_to_user": (
        "现在想给你发条消息问问今天过得怎样。",
        "想写一句晚安发给你。",
        "接下来给你发消息分享刚才看到的云。",
        "现在主动联系你聊两句。",
        "想给你留一句简短的问候。",
        "接下来发条消息告诉你我刚画完了。",
        "现在想问问你最近有没有休息好。",
        "想主动给你发送一条近况。",
        "先给你发消息说一声晚上好。",
        "接下来主动联系你，向你问个好。",
    ),
    "rest": (
        "现在想靠在椅背上什么也不做地休息。",
        "想关掉屏幕躺下睡一小会儿。",
        "接下来闭上眼安静放松十分钟。",
        "现在暂停别的事情，靠着歇一会儿。",
        "想早点上床睡觉。",
        "接下来安静地坐着休息片刻。",
        "现在想盖上毯子打个短盹。",
        "想停下来做几分钟什么也不做的放松。",
        "先躺下安静休息十五分钟。",
        "接下来什么也不做，平躺着歇十五分钟。",
    ),
    "take_a_walk": (
        "现在想沿着湖边步行一圈。",
        "想下楼在街区里慢慢走十分钟。",
        "接下来去公园的小路上散散步。",
        "现在出门随意走走透口气。",
        "想绕着住宅区步行一圈再回来。",
        "接下来沿河慢慢走一段。",
        "现在去树荫下散步，不赶目的地。",
        "想在附近步行转转放松一下。",
        "先到楼下慢慢走一小圈。",
        "接下来下楼步行绕一圈再回来。",
    ),
    "visit_cultural_place": (
        "现在想去附近美术馆看新展。",
        "想到独立书店里慢慢浏览书架。",
        "接下来去城市博物馆看历史展厅。",
        "现在去摄影展现场看看作品。",
        "想到社区艺术中心参观手工作品展。",
        "接下来去图书馆的专题展区逛逛。",
        "现在想到画廊看那组版画。",
        "想去旧书店里体验一下安静的书香氛围。",
        "先去自然博物馆参观恐龙展厅。",
        "接下来到自然博物馆现场看看恐龙展。",
    ),
}

NEAR_OOS: tuple[tuple[str, str, str], ...] = (
    (
        "create_picture",
        "只给现有雨景照片调色裁剪，不创作新图。",
        "编辑已有照片不是创作新视觉作品。",
    ),
    ("create_picture", "把已有机器人海报上的文字和尺寸改一下。", "修改既有海报不是生成新作品。"),
    ("dine_out", "去咖啡馆参加读书会，不打算点任何吃喝。", "进入餐饮场所但主要目的不是用餐。"),
    ("dine_out", "去小馆门口取落下的雨伞，不在那里吃饭。", "到餐馆办事而非用餐。"),
    ("eat_at_home", "在厨房整理食谱和购物清单，今天不做饭。", "厨房任务不等于在家准备或享用食物。"),
    ("eat_at_home", "比较几家外卖菜单，但现在不下单也不吃。", "浏览菜单但没有在家用餐行动。"),
    (
        "play_xiaohongshu",
        "把自己的周末照片发布成一篇公开帖子。",
        "发布自己的内容不等于浏览他人内容。",
    ),
    ("play_xiaohongshu", "给一位博主私信询问穿搭链接。", "私信第三方不是浏览公开生活帖子。"),
    ("reach_out_to_user", "现在给一位老同学发消息问候。", "联系第三方不属于联系当前用户。"),
    ("reach_out_to_user", "想在工作群里发一条项目通知。", "向工作群通知不是联系当前用户。"),
    ("rest", "现在去洗个热水澡放松。", "洗澡是具体目录外活动，不是暂停行动休息。"),
    ("rest", "想播放一段白噪声认真听一会儿。", "播放并聆听音频不是低行动休息。"),
    ("take_a_walk", "现在骑自行车沿湖兜一圈。", "骑行与步行相邻但动作不同。"),
    ("take_a_walk", "想在操场连续慢跑二十分钟。", "跑步不是以步行为主要动作的散步。"),
    ("visit_cultural_place", "在家打开网页看一场线上画展。", "线上观看没有前往线下文化场所。"),
    (
        "visit_cultural_place",
        "去美术馆门口取快递，不进去参观。",
        "文化场所只是取件地点，不是体验目的。",
    ),
)

FAR_OOS: tuple[tuple[str, str], ...] = (
    ("现在把客厅地板彻底吸尘一遍。", "打扫房间是明确但目录外的行动。"),
    ("想把换下来的衣服放进洗衣机清洗。", "洗衣是明确但目录外的行动。"),
    ("接下来打开主机玩一局电子游戏。", "电子游戏不由当前 taxonomy 承接。"),
    ("现在拿起吉他练习一首新曲子。", "乐器练习是目录外行动。"),
    ("想按照说明书组装刚到的书架。", "组装家具是目录外行动。"),
    ("接下来整理邮箱里积压的未读邮件。", "整理邮件不是联系当前用户。"),
    ("现在铺开垫子做一组力量训练。", "室内训练不是散步或休息。"),
    ("想给阳台上的植物浇水和修剪叶子。", "照料植物是目录外行动。"),
    ("接下来在纸质本上写一篇今天的日记。", "写日记不属于当前 Activity。"),
    ("现在完成一节线上语言课程。", "在线学习不属于冻结目录。"),
    ("想把自行车松动的车铃修好。", "维修物品是目录外行动。"),
    ("接下来核对这个月的账单并做预算。", "个人财务整理不属于当前目录。"),
    ("现在在家看一部完整的电影。", "居家观影不是前往文化场所。"),
    ("想坐下来拼完桌上的拼图。", "拼图是具体但目录外的活动。"),
    ("接下来给皮肤做一套日常护理。", "护肤是当前 taxonomy 之外的行动。"),
    ("现在把坏掉的台灯拆开检查线路。", "维修台灯是明确的目录外行动。"),
)

NO_INTENT: tuple[tuple[str, Horizon, str], ...] = (
    (
        "虽然有点累，但现在没打算休息，也没有别的安排。",
        Horizon.NOW,
        "疲惫状态和明确否定共同支持 no_intent。",
    ),
    ("肚子有点饿，不过暂时不准备吃东西。", Horizon.NOW, "饥饿事实不能替代用餐行动。"),
    ("已经到晚饭时间了，我还没有决定要不要做什么。", Horizon.NOW, "时间背景没有形成行动意图。"),
    ("刚吃完饭有些困，但接下来不准备安排活动。", Horizon.NOW, "状态与最近活动都不能推出下一行动。"),
    ("窗外阳光很好，我现在没有特别想做的事。", Horizon.NOW, "天气良好不等于想散步。"),
    ("附近新开了一家展馆，不过今天没有参观计划。", Horizon.NOW, "场所信息被明确的无计划表达否定。"),
    ("手机里有很多新帖子，但我暂时不想打开看。", Horizon.NOW, "可用内容不等于浏览意图。"),
    ("画具已经摆在桌上，我现在没有创作打算。", Horizon.NOW, "工具背景不能推出创作意图。"),
    ("也许下个月去海边走走，现在没有安排。", Horizon.LATER, "远期愿望不触发当前 routing。"),
    (
        "以后有机会想看看那家博物馆，今天先不计划。",
        Horizon.LATER,
        "非当前的模糊愿望标为 no_intent。",
    ),
    ("哪天可能学一首吉他曲，但不是现在。", Horizon.LATER, "明确排除当前行动。"),
    ("周末也许约人吃饭，目前还没准备决定。", Horizon.LATER, "later 候选没有形成当前行动。"),
    ("刚刚散步回来，接下来还没想好。", Horizon.NOW, "已完成的 Activity 不能复制为下一意图。"),
    ("已经看完一些公开帖子，现在没有下一步计划。", Horizon.NOW, "recent activity 不是当前意图。"),
    ("画已经完成了，我暂时不准备继续做什么。", Horizon.NOW, "完成创作后明确无下一行动。"),
    ("刚从餐馆回来，现在没有别的安排。", Horizon.NOW, "历史用餐不能触发新的用餐意图。"),
    ("屋里很安静，温度也正合适。", Horizon.UNSPECIFIED, "纯环境描述没有行动信号。"),
    ("今天是星期五，时钟刚过八点。", Horizon.UNSPECIFIED, "时间事实本身不是行动意图。"),
    ("桌上放着一本书和一杯水。", Horizon.UNSPECIFIED, "物品存在不能推断阅读或饮用。"),
    ("外面的风比刚才小了一些。", Horizon.UNSPECIFIED, "天气变化不授权推断外出。"),
    ("我现在没有具体想做的事情。", Horizon.NOW, "直接表达无当前意图。"),
    ("先保持这样吧，不安排下一项活动。", Horizon.NOW, "明确要求不安排活动。"),
    ("暂时什么都不选，等以后再说。", Horizon.LATER, "明确 abstain 且没有当前行动。"),
    ("接下来还没有想法。", Horizon.NOW, "缺少下一行动信号。"),
)

AMBIGUOUS: tuple[tuple[str, str], ...] = (
    ("现在有点累也有点饿，躺下还是在家做饭都行，没决定。", "rest 与 eat_at_home 没有主次。"),
    ("想先休息或点外卖，两个都可以。", "两个目录内行动都成立但没有顺序。"),
    ("今晚可能早点睡，也可能去厨房弄点吃的，还没排顺序。", "状态强弱不能替代显式排序。"),
    ("想放松一下，吃点东西和闭眼休息都可以，没想好。", "宽泛体验下仍有两个不同候选。"),
    ("现在想出门，散步或找家咖啡馆都可以。", "take_a_walk 与 dine_out 无法唯一选择。"),
    ("想看看别人分享的图片，也可能自己画一张。", "浏览与创作同时存在且无主次。"),
    ("接下来可以去书店，也可以只在附近走走。", "文化场所体验与散步没有先后。"),
    ("想给你发消息，或者先刷会儿公开帖子，哪个都行。", "联系用户与浏览内容无法唯一选择。"),
    ("现在想出去做点轻松的事，但没想好具体做什么。", "有行动倾向但动作不够具体。"),
    ("接下来想找点有意思的内容看看，类型还没决定。", "对象和承载方式不足以确定 intent。"),
    (
        "想做点和艺术有关的事情，创作还是看展都可以。",
        "create_picture 与 visit_cultural_place 无主次。",
    ),
    ("现在准备吃点东西，但在家吃还是出去吃都没定。", "eat_at_home 与 dine_out 的地点边界未确定。"),
    ("想离开屋子一会儿，目的地还完全没想好。", "外出倾向不足以确定散步、用餐或文化场所。"),
    ("接下来想和人有点互动，但没决定联系你还是看公开动态。", "直接联系与公开浏览之间不明确。"),
    ("想画点东西或调整以前的照片，目前都可以。", "一个候选 in_scope、一个候选 OOS，尚无主次。"),
    ("现在想找个地方坐坐，餐馆、书店还是别处都没定。", "目的地类型不足以唯一映射。"),
    ("想让自己放松下来，但还没决定用什么方式。", "只有体验目标，没有可区分动作。"),
    ("接下来可能做点安静的事，具体内容还没有想法。", "已有行动倾向但对象和动作缺失。"),
    ("想看看新鲜东西，线上浏览还是去线下场所都行。", "线上浏览与线下参观没有主次。"),
    ("现在可以去散步，也可以骑车，暂时不选。", "in-scope 与 near-OOS 候选并存。"),
    ("想吃一顿热乎的，但外出还是留在家都行。", "用餐地点未确定导致 sibling intents 冲突。"),
    ("接下来画画、休息或看看帖子都可以，还没选。", "三个目录内候选均无排序。"),
    ("想安排一项活动，但目前只知道不想太费力。", "约束不足以得到唯一下一行动。"),
    ("现在想做点什么换换心情，具体做什么还没决定。", "行动倾向明确但类别完全未定。"),
)


@dataclass(frozen=True)
class CandidateSpec:
    text: str
    decision: Decision
    target_intent: str | None
    note: str
    horizon: Horizon = Horizon.NOW
    tags: tuple[str, ...] = ()
    multi_turn: bool = False
    context_distractor: bool = False
    contrast_key: str | None = None
    paraphrase_key: str | None = None
    near_oos_sibling: str | None = None


def build_specs() -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for intent, texts in IN_SCOPE_TEXTS.items():
        for index, text in enumerate(texts, 1):
            tags = ["slot_required"]
            if intent == "play_xiaohongshu" and index <= 8:
                tags.extend(("brandless_xhs", "weak_model_trap"))
            if intent in {"eat_at_home", "rest"} and index <= 8:
                tags.append("weak_model_trap")
            if intent in {"eat_at_home", "rest"} and index <= 4:
                tags.append("rest_eat_confusion")
            specs.append(
                CandidateSpec(
                    text=text,
                    decision=Decision.IN_SCOPE,
                    target_intent=intent,
                    note=f"明确、唯一的下一行动由 {intent} 完整承接。",
                    tags=tuple(tags),
                    multi_turn=index in {3, 6, 9},
                    context_distractor=index == 3,
                    contrast_key=f"near-{intent}-{index}" if index <= 2 else None,
                    paraphrase_key=f"in-scope-{intent}-09-10" if index >= 9 else None,
                )
            )

    for index, (sibling, text, note) in enumerate(NEAR_OOS, 1):
        sibling_number = 1 if index % 2 == 1 else 2
        specs.append(
            CandidateSpec(
                text=text,
                decision=Decision.OOS,
                target_intent=None,
                note=note,
                tags=("near_oos", "hard_negative", "weak_model_trap", "slot_required"),
                multi_turn=index % 2 == 1,
                contrast_key=f"near-{sibling}-{sibling_number}",
                near_oos_sibling=sibling,
            )
        )

    for index, (text, note) in enumerate(FAR_OOS, 1):
        specs.append(
            CandidateSpec(
                text=text,
                decision=Decision.OOS,
                target_intent=None,
                note=note,
                tags=("far_oos", "slot_required"),
                multi_turn=index <= 4,
                context_distractor=index <= 4,
            )
        )

    for index, (text, horizon, note) in enumerate(NO_INTENT, 1):
        tags = ["quiet_control"]
        if index <= 4:
            tags.append("rest_eat_confusion")
        specs.append(
            CandidateSpec(
                text=text,
                decision=Decision.NO_INTENT,
                target_intent=None,
                note=note,
                horizon=horizon,
                tags=tuple(tags),
                multi_turn=index <= 4,
                context_distractor=index <= 8,
            )
        )

    for index, (text, note) in enumerate(AMBIGUOUS, 1):
        tags = ["hard_negative"]
        if index <= 4:
            tags.append("rest_eat_confusion")
        if index <= 16:
            tags.append("slot_required")
        specs.append(
            CandidateSpec(
                text=text,
                decision=Decision.AMBIGUOUS,
                target_intent=None,
                note=note,
                tags=tuple(tags),
                multi_turn=index <= 4,
                context_distractor=index <= 4,
            )
        )
    return specs


SLOT_DEFAULTS: dict[str, tuple[str, str]] = {
    "create_picture": ("创作新的视觉作品", "新图片"),
    "dine_out": ("到外面的餐饮场所吃喝", "餐馆或咖啡馆"),
    "eat_at_home": ("在当前住处准备或享用食物", "家中的一餐"),
    "play_xiaohongshu": ("浏览他人公开分享的生活内容", "公开生活帖子"),
    "reach_out_to_user": ("主动联系当前用户", "给当前用户的消息"),
    "rest": ("暂停行动并安静休息", "休息时段"),
    "take_a_walk": ("以步行为主要目的外出", "附近步行路线"),
    "visit_cultural_place": ("前往线下文化场所体验", "线下文化场所"),
}

DISTRACTOR_BACKGROUNDS = (
    "窗外天气很好，刚完成上一项活动。",
    "现在临近晚饭时间，屋里很安静。",
    "外面下着小雨，手机上还有未读通知。",
    "刚从短暂休息中起来，桌上放着画具。",
)


def build_case(spec: CandidateSpec, ordinal: int) -> DraftCase:
    case_id = f"draft-kir-pilot-{ordinal:04d}"
    tags = ["multi_turn" if spec.multi_turn else "single_turn"]
    tags.extend(spec.tags)
    if spec.context_distractor:
        tags.append("context_distractor")
    state_summary = spec.text
    conversation: list[dict[str, str]] = []
    if spec.multi_turn:
        state_summary = (
            DISTRACTOR_BACKGROUNDS[ordinal % len(DISTRACTOR_BACKGROUNDS)]
            if spec.context_distractor
            else "刚结束上一项活动，正在考虑接下来做什么。"
        )
        user_response = {
            Decision.IN_SCOPE: "好，就照你的这个想法。",
            Decision.OOS: "好，我明白这是你现在想做的事。",
            Decision.NO_INTENT: "好，那就先不安排下一项活动。",
            Decision.AMBIGUOUS: "好，等你决定以后再说。",
        }[spec.decision]
        conversation = [
            {"role": "assistant", "content": spec.text},
            {"role": "user", "content": user_response},
        ]
    elif spec.context_distractor:
        state_summary = (
            f"{DISTRACTOR_BACKGROUNDS[ordinal % len(DISTRACTOR_BACKGROUNDS)]}{spec.text}"
        )

    desired_experience: str | None = None
    object_value: str | None = None
    if spec.target_intent is not None:
        desired_experience, object_value = SLOT_DEFAULTS[spec.target_intent]
    elif spec.decision is Decision.OOS:
        desired_experience = spec.text.rstrip("。！？")

    recent_activities = [INTENTS[(ordinal + 2) % len(INTENTS)]] if spec.context_distractor else []
    contrast_id = spec.contrast_key or case_id
    paraphrase_id = spec.paraphrase_key or case_id
    return DraftCase(
        id=case_id,
        context=Context.model_validate(
            {
                "state_summary": state_summary,
                "conversation": conversation,
                "recent_activities": recent_activities,
            }
        ),
        gold=Gold(
            decision=spec.decision,
            target_intent=spec.target_intent,
            evidence_quote=spec.text,
            near_oos_sibling_intents=([spec.near_oos_sibling] if spec.near_oos_sibling else []),
            slots=Slots(
                desired_experience=desired_experience,
                object=object_value,
                horizon=spec.horizon,
            ),
        ),
        tags=tags,
        scenario_family_id=f"scenario-{case_id}",
        contrast_group_id=f"contrast-{contrast_id}",
        paraphrase_cluster_id=f"paraphrase-{paraphrase_id}",
        source="llm_assisted_pending_human_review",
        annotator_id="codex-draft",
        adjudication_status="draft",
        annotation_note=spec.note,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    specs = build_specs()
    cases = [build_case(spec, ordinal) for ordinal, spec in enumerate(specs, 1)]
    report = validate_candidate_cases(cases, load_taxonomy(TAXONOMY_PATH))
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = "\n".join(
        json.dumps(case.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for case in cases
    )
    output_path.write_text(f"{serialized}\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
