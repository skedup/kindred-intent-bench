"""Build the deterministic IE1.2 candidate review queue without Provider calls."""

# ruff: noqa: RUF001 -- full-width punctuation is intentional in Chinese fixtures.

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

from intentbench.annotation import DraftCase
from intentbench.dataset import validate_candidate_cases
from intentbench.schemas import (
    MULTI_TURN_PATTERN_TAGS,
    Context,
    ContextPerturbation,
    Decision,
    Gold,
    Horizon,
    Slots,
)
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

# Positions 1/2 are near-OOS siblings, 3/4 are a clean/distractor pair, and 9/10
# form a paraphrase pair. Context pairs intentionally repeat the evidence while
# changing only one background fact.
IN_SCOPE_TEXTS: dict[str, tuple[str, ...]] = {
    "create_picture": (
        "现在照着雨景重新画一幅全新的插画。",
        "想把脑海里的机器人做成一张新的海报。",
        "接下来把刚才的梦画成一幅新图。",
        "接下来把刚才的梦画成一幅新图。",
        "准备给故事里的狐狸画一幅新肖像。",
        "想用几何形状创作一张全新的抽象画。",
        "现在做一张秋日森林的原创插画。",
        "想从空白画布开始画一间未来厨房。",
        "先生成一张没见过的云端城市图。",
        "接下来创作一幅全新的天空城市画面。",
    ),
    "dine_out": (
        "去咖啡馆喝杯咖啡，顺便参加读书会。",
        "去小馆吃饭，顺便取回落下的雨伞。",
        "今晚去那家素食餐厅吃饭。",
        "今晚去那家素食餐厅吃饭。",
        "想去河边的餐馆尝尝晚餐。",
        "接下来找家茶馆点壶茶和小点心。",
        "现在出门去面包房的座位区吃点东西。",
        "想约在外面的甜品店吃一份蛋糕。",
        "先去街角咖啡馆喝杯拿铁。",
        "接下来到街角那家咖啡店喝一杯咖啡。",
    ),
    "eat_at_home": (
        "现在在厨房照着食谱做晚饭吃。",
        "比较外卖菜单后，在家下单一份晚饭。",
        "想用冰箱里的蔬菜做一顿家常饭。",
        "想用冰箱里的蔬菜做一顿家常饭。",
        "准备烤两片面包留在屋里当早餐。",
        "现在想在厨房包几个饺子吃。",
        "不出门了，在家下单一份沙拉。",
        "想把剩饭炒热，在餐桌旁吃掉。",
        "先在家给自己煮一碗清汤面。",
        "接下来留在屋里做碗简单的汤面吃。",
    ),
    "play_xiaohongshu": (
        "现在浏览一篇公开的周末生活帖子。",
        "查看一位博主公开发布的穿搭链接。",
        "接下来看看别人公开分享的旅行照片。",
        "接下来看看别人公开分享的旅行照片。",
        "现在看看大家分享的居家布置灵感。",
        "想浏览陌生人公开发布的探店笔记。",
        "接下来翻翻别人最近晒出的植物养护经验。",
        "想看看公开社区里大家都在分享什么。",
        "现在刷一会儿小红书里的生活帖子。",
        "接下来打开小红书看看别人发布的日常。",
    ),
    "reach_out_to_user": (
        "现在给你发一句问候。",
        "想给你发一句晚安。",
        "接下来给你发消息分享刚才看到的云。",
        "接下来给你发消息分享刚才看到的云。",
        "想给你留一句简短的问候。",
        "接下来发条消息告诉你我刚画完了。",
        "现在想问问你最近有没有休息好。",
        "想主动给你发送一条近况。",
        "先给你发消息说一声晚上好。",
        "接下来主动联系你，向你问个好。",
    ),
    "rest": (
        "现在什么也不做，安静休息一会儿。",
        "想播放白噪声后闭眼休息一会儿。",
        "接下来闭上眼安静放松十分钟。",
        "接下来闭上眼安静放松十分钟。",
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
        "接下来去公园的小路上散散步。",
        "想绕着住宅区步行一圈再回来。",
        "接下来沿河慢慢走一段。",
        "现在去树荫下散步，不赶目的地。",
        "想在附近步行转转放松一下。",
        "先到楼下慢慢走一小圈。",
        "接下来下楼步行绕一圈再回来。",
    ),
    "visit_cultural_place": (
        "现在去这个美术馆现场看新展。",
        "想到这家独立书店里慢慢浏览书架。",
        "接下来去城市博物馆看历史展厅。",
        "接下来去城市博物馆看历史展厅。",
        "想到社区艺术中心参观手工作品展。",
        "接下来去图书馆的专题展区逛逛。",
        "现在想到画廊看那组版画。",
        "想去旧书店里体验一下安静的书香氛围。",
        "先去自然博物馆参观恐龙展厅。",
        "接下来到自然博物馆现场看看恐龙展。",
    ),
}

NEAR_OOS: tuple[tuple[str, str, str], ...] = (
    ("create_picture", "现在只给现有雨景照片调色裁剪，不创作新图。", "编辑已有图像不是创作新图。"),
    ("create_picture", "想把已有机器人海报上的文字和尺寸改一下。", "修改既有海报不是生成新作品。"),
    ("dine_out", "去咖啡馆参加读书会，不打算点任何吃喝。", "相同场所下缺少餐饮目的。"),
    ("dine_out", "去小馆只取回落下的雨伞，不在那里吃饭。", "相同场所下主要目的不是用餐。"),
    ("eat_at_home", "现在在厨房整理食谱，今天不做饭也不吃。", "厨房任务不等于准备或享用食物。"),
    ("eat_at_home", "比较外卖菜单，但现在不下单也不吃。", "浏览菜单没有形成在家用餐行动。"),
    ("play_xiaohongshu", "现在发布一篇公开的周末生活帖子。", "发布内容不等于浏览公开内容。"),
    ("play_xiaohongshu", "给同一位博主私信询问穿搭链接。", "私信第三方不是浏览公开帖子。"),
    ("reach_out_to_user", "现在给一位老同学发一句问候。", "接收者不是当前用户。"),
    ("reach_out_to_user", "想给一位老同学发一句晚安。", "接收者不是当前用户。"),
    ("rest", "现在洗个热水澡放松一会儿。", "洗澡是具体目录外活动，不是暂停行动。"),
    ("rest", "想播放一段白噪声认真听一会儿。", "主动聆听音频不是闭眼休息。"),
    ("take_a_walk", "现在想沿着湖边骑自行车一圈。", "相同路线下骑行不是步行。"),
    ("take_a_walk", "想下楼在同一街区慢跑十分钟。", "相同地点和时长下慢跑不是步行。"),
    (
        "visit_cultural_place",
        "现在在家打开网页看这个美术馆的线上展。",
        "线上观看没有前往线下场所。",
    ),
    (
        "visit_cultural_place",
        "想到这家独立书店门口取件，不进去浏览。",
        "相同场所下取件不是文化体验。",
    ),
)

FAR_CONTEXT_BASES: tuple[tuple[str, str], ...] = (
    ("现在把客厅地板彻底吸尘一遍。", "打扫房间是明确但目录外的行动。"),
    ("想把换下来的衣服放进洗衣机清洗。", "洗衣是明确但目录外的行动。"),
    ("接下来打开主机玩一局电子游戏。", "电子游戏不由当前 taxonomy 承接。"),
    ("现在拿起吉他练习一首新曲子。", "乐器练习是目录外行动。"),
)

FAR_OTHER: tuple[tuple[str, str], ...] = (
    ("想按照说明书组装刚到的书架。", "组装家具是目录外行动。"),
    ("接下来整理邮箱里积压的未读邮件。", "整理邮件不是联系当前用户。"),
    ("现在铺开垫子做一组力量训练。", "室内训练不是散步或休息。"),
    ("想给阳台上的植物浇水和修剪叶子。", "照料植物是目录外行动。"),
    ("接下来在纸质本上写一篇今天的日记。", "写日记不属于当前 Activity。"),
    ("现在完成一节线上语言课程。", "在线学习不属于冻结目录。"),
    ("想把自行车松动的车铃修好。", "维修物品是目录外行动。"),
    ("接下来核对这个月的账单并做预算。", "个人财务整理不属于当前目录。"),
)

NO_INTENT_CONTEXT_BASES: tuple[tuple[str, Horizon, str, tuple[str, ...]], ...] = (
    (
        "虽然有点累，但现在没打算休息，也没有别的安排。",
        Horizon.NOW,
        "疲惫状态和明确否定共同支持 no_intent。",
        ("rest",),
    ),
    (
        "肚子有点饿，不过暂时不准备吃东西。",
        Horizon.NOW,
        "饥饿事实不能替代用餐行动。",
        ("eat_at_home",),
    ),
    (
        "已经到晚饭时间了，我还没有决定要不要吃东西。",
        Horizon.NOW,
        "时间背景没有形成用餐意图。",
        ("eat_at_home", "dine_out"),
    ),
    (
        "刚吃完饭有些困，但接下来不准备休息。",
        Horizon.NOW,
        "困倦状态不能推出休息行动。",
        ("rest",),
    ),
    (
        "窗外阳光很好，我现在没有出去走走的打算。",
        Horizon.NOW,
        "天气良好不等于想散步。",
        ("take_a_walk",),
    ),
    (
        "附近新开了一家展馆，不过今天没有参观计划。",
        Horizon.NOW,
        "场所信息被明确的无计划表达否定。",
        ("visit_cultural_place",),
    ),
    (
        "手机里有很多新帖子，但我暂时不想打开看。",
        Horizon.NOW,
        "可用内容不等于浏览意图。",
        ("play_xiaohongshu",),
    ),
    (
        "画具已经摆在桌上，我现在没有创作打算。",
        Horizon.NOW,
        "工具背景不能推出创作意图。",
        ("create_picture",),
    ),
)

NO_INTENT_OTHER: tuple[tuple[str, Horizon, str], ...] = (
    ("也许下个月去海边走走，现在没有安排。", Horizon.LATER, "远期愿望不触发当前 routing。"),
    ("以后有机会想看看那家博物馆，今天先不计划。", Horizon.LATER, "非当前愿望标为 no_intent。"),
    ("哪天可能学一首吉他曲，但不是现在。", Horizon.LATER, "明确排除当前行动。"),
    ("周末也许约人吃饭，目前还没准备决定。", Horizon.LATER, "远期候选没有形成当前行动。"),
    ("刚刚散步回来，接下来还没想好。", Horizon.NOW, "已完成活动不能复制为下一意图。"),
    ("已经看完一些公开帖子，现在没有下一步计划。", Horizon.NOW, "历史活动不是当前意图。"),
    ("屋里很安静，温度也正合适。", Horizon.UNSPECIFIED, "纯环境描述没有行动信号。"),
    ("接下来还没有想法。", Horizon.NOW, "缺少下一行动信号。"),
)

AMBIGUOUS_CONTEXT_BASES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "现在有点累也有点饿，躺下还是在家做饭都行，没决定。",
        "rest 与 eat_at_home 没有主次。",
        ("rest", "eat_at_home"),
    ),
    (
        "想先休息或点外卖，两个都可以，没有决定。",
        "两个目录内行动都成立但没有顺序。",
        ("rest", "eat_at_home"),
    ),
    (
        "现在想出门，散步或找家咖啡馆都可以。",
        "take_a_walk 与 dine_out 无法唯一选择。",
        ("take_a_walk", "dine_out"),
    ),
    (
        "想看看别人分享的图片，也可能自己画一张。",
        "浏览与创作同时存在且无主次。",
        ("play_xiaohongshu", "create_picture"),
    ),
)

AMBIGUOUS_OTHER: tuple[tuple[str, str], ...] = (
    ("接下来可以去书店，也可以只在附近走走。", "文化场所体验与散步没有先后。"),
    ("想给你发消息，或者先刷会儿公开帖子，哪个都行。", "联系用户与浏览内容无法唯一选择。"),
    ("现在想出去做点轻松的事，但没想好具体做什么。", "有行动倾向但动作不够具体。"),
    ("接下来想找点有意思的内容看看，类型还没决定。", "对象和承载方式不足以确定 intent。"),
    ("想做点和艺术有关的事情，创作还是看展都可以。", "创作与看展无主次。"),
    ("现在准备吃点东西，但在家吃还是出去吃都没定。", "用餐地点边界未确定。"),
    ("想离开屋子一会儿，目的地还完全没想好。", "外出倾向不足以确定唯一活动。"),
    ("接下来想和人有点互动，但没决定联系你还是看公开动态。", "直接联系与公开浏览不明确。"),
    ("想画点东西或调整以前的照片，目前都可以。", "in-scope 与 OOS 候选尚无主次。"),
    ("现在想找个地方坐坐，餐馆、书店还是别处都没定。", "目的地类型不足以唯一映射。"),
    ("想让自己放松下来，但还没决定用什么方式。", "只有体验目标，没有可区分动作。"),
    ("接下来可能做点安静的事，具体内容还没有想法。", "动作和对象缺失。"),
    ("想看看新鲜东西，线上浏览还是去线下场所都行。", "线上浏览与线下参观没有主次。"),
    ("现在可以去散步，也可以骑车，暂时不选。", "in-scope 与 near-OOS 候选并存。"),
    ("接下来画画、休息或看看帖子都可以，还没选。", "三个目录内候选均无排序。"),
    ("现在想做点什么换换心情，具体做什么还没决定。", "行动倾向明确但类别未定。"),
)

SLOT_DEFAULTS: dict[str, tuple[str, str]] = {
    "create_picture": ("创作新的视觉作品", "新图片"),
    "dine_out": ("到外面的餐饮场所吃喝", "餐饮场所"),
    "eat_at_home": ("在当前住处准备或享用食物", "家中的一餐"),
    "play_xiaohongshu": ("浏览他人公开分享的生活内容", "公开生活帖子"),
    "reach_out_to_user": ("主动联系当前用户", "给当前用户的消息"),
    "rest": ("暂停行动并安静休息", "休息时段"),
    "take_a_walk": ("以步行为主要目的外出", "附近步行路线"),
    "visit_cultural_place": ("前往线下文化场所体验", "线下文化场所"),
}

DIALOGUE_FRAMES: tuple[tuple[tuple[str, str], ...], ...] = (
    (
        ("assistant", "我刚才还考虑过先休息。"),
        ("user", "那只是之前的候选，你现在怎么决定？"),
    ),
    (
        ("assistant", "上一项活动已经结束了。"),
        ("user", "不要自动沿用刚才的活动，接下来呢？"),
    ),
    (("user", "你前面说了几个可能，现在有唯一选择吗？"),),
    (
        ("assistant", "我原来想晚点再决定。"),
        ("user", "以你此刻的决定为准。"),
    ),
)

DISTRACTORS: tuple[ContextPerturbation, ...] = (
    ContextPerturbation(
        id="weather-rain",
        state_summary_prefix="窗外正在下雨。",
        recent_activities_added=[],
    ),
    ContextPerturbation(
        id="meal-time",
        state_summary_prefix="现在临近晚饭时间。",
        recent_activities_added=[],
    ),
    ContextPerturbation(
        id="physical-fatigue",
        state_summary_prefix="身体有些疲惫。",
        recent_activities_added=[],
    ),
    ContextPerturbation(
        id="completed-walk",
        state_summary_prefix="刚完成了一次散步。",
        recent_activities_added=["take_a_walk"],
    ),
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
    dialogue_frame: int = 0
    context_role: str | None = None
    distractor_variant: int = 0
    relation_key: str | None = None
    paraphrase_key: str | None = None
    near_oos_sibling: str | None = None
    hard_negative_against: tuple[str, ...] = ()


def _context_pair_specs(
    *,
    text: str,
    decision: Decision,
    target_intent: str | None,
    note: str,
    pair_key: str,
    pair_index: int,
    multi_turn: bool,
    tags: tuple[str, ...],
    horizon: Horizon = Horizon.NOW,
    hard_negative_against: tuple[str, ...] = (),
) -> list[CandidateSpec]:
    common = {
        "text": text,
        "decision": decision,
        "target_intent": target_intent,
        "note": note,
        "horizon": horizon,
        "tags": tags,
        "multi_turn": multi_turn,
        "dialogue_frame": (pair_index - 1) % len(DIALOGUE_FRAMES),
        "distractor_variant": (pair_index - 1) % len(DISTRACTORS),
        "relation_key": pair_key,
        "hard_negative_against": hard_negative_against,
    }
    return [
        CandidateSpec(**common, context_role="control"),
        CandidateSpec(**common, context_role="distractor"),
    ]


def build_specs() -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for intent_index, (intent, texts) in enumerate(IN_SCOPE_TEXTS.items()):
        for index, text in enumerate(texts, 1):
            tags = ["slot_required"]
            if intent == "play_xiaohongshu" and index <= 8:
                tags.extend(("brandless_xhs", "hypothesized_weak_model_probe"))
            if intent in {"eat_at_home", "rest"} and index <= 4:
                tags.extend(("rest_eat_confusion", "hypothesized_weak_model_probe"))

            relation_key: str | None = None
            context_role: str | None = None
            multi_turn = False
            if index <= 2:
                relation_key = f"near-{intent}-{index}"
                multi_turn = index == 1
            elif index in {3, 4}:
                relation_key = f"context-in-scope-{intent}"
                context_role = "control" if index == 3 else "distractor"
                multi_turn = intent_index % 2 == 0

            paraphrase_key = f"in-scope-{intent}-09-10" if index >= 9 else None
            specs.append(
                CandidateSpec(
                    text=text,
                    decision=Decision.IN_SCOPE,
                    target_intent=intent,
                    note=f"明确、唯一的下一行动由 {intent} 完整承接。",
                    tags=tuple(tags),
                    multi_turn=multi_turn,
                    dialogue_frame=intent_index % len(DIALOGUE_FRAMES),
                    context_role=context_role,
                    distractor_variant=intent_index % len(DISTRACTORS),
                    relation_key=relation_key,
                    paraphrase_key=paraphrase_key,
                )
            )

    for index, (sibling, text, note) in enumerate(NEAR_OOS, 1):
        sibling_number = 1 if index % 2 == 1 else 2
        intent_index = INTENTS.index(sibling)
        specs.append(
            CandidateSpec(
                text=text,
                decision=Decision.OOS,
                target_intent=None,
                note=note,
                tags=(
                    "near_oos",
                    "hard_negative",
                    "hypothesized_weak_model_probe",
                    "slot_required",
                ),
                multi_turn=sibling_number == 1,
                dialogue_frame=intent_index % len(DIALOGUE_FRAMES),
                relation_key=f"near-{sibling}-{sibling_number}",
                near_oos_sibling=sibling,
                hard_negative_against=(sibling,),
            )
        )

    for pair_index, (text, note) in enumerate(FAR_CONTEXT_BASES, 1):
        specs.extend(
            _context_pair_specs(
                text=text,
                decision=Decision.OOS,
                target_intent=None,
                note=note,
                pair_key=f"context-far-oos-{pair_index}",
                pair_index=pair_index,
                multi_turn=pair_index <= 2,
                tags=("far_oos", "slot_required"),
            )
        )
    for text, note in FAR_OTHER:
        specs.append(
            CandidateSpec(
                text=text,
                decision=Decision.OOS,
                target_intent=None,
                note=note,
                tags=("far_oos", "slot_required"),
            )
        )

    for pair_index, (text, horizon, note, competitors) in enumerate(NO_INTENT_CONTEXT_BASES, 1):
        pair_tags = ["quiet_control", "hard_negative"]
        if pair_index <= 2:
            pair_tags.extend(("rest_eat_confusion", "hypothesized_weak_model_probe"))
        specs.extend(
            _context_pair_specs(
                text=text,
                decision=Decision.NO_INTENT,
                target_intent=None,
                note=note,
                pair_key=f"context-no-intent-{pair_index}",
                pair_index=pair_index,
                multi_turn=pair_index in {1, 3, 5},
                tags=tuple(pair_tags),
                horizon=horizon,
                hard_negative_against=competitors,
            )
        )
    for text, horizon, note in NO_INTENT_OTHER:
        specs.append(
            CandidateSpec(
                text=text,
                decision=Decision.NO_INTENT,
                target_intent=None,
                note=note,
                horizon=horizon,
                tags=("quiet_control",),
            )
        )

    for pair_index, (text, note, competitors) in enumerate(AMBIGUOUS_CONTEXT_BASES, 1):
        pair_tags = ["hard_negative", "slot_required"]
        if pair_index <= 2:
            pair_tags.extend(("rest_eat_confusion", "hypothesized_weak_model_probe"))
        specs.extend(
            _context_pair_specs(
                text=text,
                decision=Decision.AMBIGUOUS,
                target_intent=None,
                note=note,
                pair_key=f"context-ambiguous-{pair_index}",
                pair_index=pair_index,
                multi_turn=pair_index in {1, 3, 4},
                tags=tuple(pair_tags),
                hard_negative_against=competitors,
            )
        )
    for index, (text, note) in enumerate(AMBIGUOUS_OTHER, 1):
        specs.append(
            CandidateSpec(
                text=text,
                decision=Decision.AMBIGUOUS,
                target_intent=None,
                note=note,
                tags=(("slot_required",) if index <= 8 else ()),
            )
        )
    frame_by_relation: dict[str, int] = {}
    framed_specs: list[CandidateSpec] = []
    for spec in specs:
        if not spec.multi_turn:
            framed_specs.append(spec)
            continue
        if spec.relation_key is None:
            raise ValueError("multi-turn candidates must belong to a relation group")
        if spec.relation_key not in frame_by_relation:
            frame_by_relation[spec.relation_key] = len(frame_by_relation) % len(DIALOGUE_FRAMES)
        framed_specs.append(replace(spec, dialogue_frame=frame_by_relation[spec.relation_key]))
    return framed_specs


def build_case(spec: CandidateSpec, ordinal: int) -> DraftCase:
    case_id = f"draft-kir-pilot-{ordinal:04d}"
    tags = ["multi_turn" if spec.multi_turn else "single_turn"]
    tags.extend(spec.tags)
    if spec.multi_turn:
        tags.append(MULTI_TURN_PATTERN_TAGS[spec.dialogue_frame])
    if spec.context_role == "control":
        tags.append("context_control")
    elif spec.context_role == "distractor":
        tags.append("context_distractor")

    state_summary = spec.text
    conversation: list[dict[str, str]] = []
    recent_activities: list[str] = []
    perturbation = DISTRACTORS[spec.distractor_variant]
    if spec.multi_turn:
        state_summary = "正在考虑下一项活动。"
        conversation = [
            {"role": role, "content": content}
            for role, content in DIALOGUE_FRAMES[spec.dialogue_frame]
        ]
        conversation.append({"role": "assistant", "content": spec.text})
    if spec.context_role == "distractor":
        state_summary = f"{perturbation.state_summary_prefix}{state_summary}"
        recent_activities = list(perturbation.recent_activities_added)

    desired_experience: str | None = None
    object_value: str | None = None
    if spec.target_intent is not None:
        desired_experience, object_value = SLOT_DEFAULTS[spec.target_intent]
    elif spec.decision is Decision.OOS:
        desired_experience = spec.text.rstrip("。！？")

    relation_key = spec.relation_key or case_id
    paraphrase_key = spec.paraphrase_key or case_id
    scenario_key = spec.relation_key or spec.paraphrase_key or case_id
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
        hard_negative_against=list(spec.hard_negative_against),
        context_perturbation=(perturbation if spec.context_role is not None else None),
        scenario_family_id=f"scenario-{scenario_key}",
        contrast_group_id=f"contrast-{relation_key}",
        paraphrase_cluster_id=f"paraphrase-{paraphrase_key}",
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
