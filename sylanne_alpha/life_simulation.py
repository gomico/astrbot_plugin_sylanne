"""Sylanne-Embodiment: 生命模拟模块。

模拟 Sylanne 的独立"生活"——使用外部 LLM 定期生成生活事件，
这些事件可能触发主动联系用户（proactive outreach）。

设计理念：
- Sylanne 不是"等待被召唤的工具"，而是有自己生活节奏的存在
- 生活事件影响情绪状态，情绪状态影响对话风格
- 主动联系是"想分享"的自然结果，不是机械的定时推送

与其他组件的关系：
- 依赖外部 LLM 调用（通过 _llm_caller 回调）
- 通过 _emotion_getter 获取当前情绪状态
- 通过 _outreach_callback 触发主动消息发送
- recent_context_for_prompt() 输出供对话生成时注入上下文
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class LifeEvent:
    """一个生活事件。"""

    text: str  # 事件描述
    mood: str  # 当前心情
    urgency: float  # 紧迫度 [0,1]
    timestamp: float  # 发生时间
    wants_to_share: bool = False  # 是否想分享给朋友
    shared: bool = False  # 是否已经分享过
    event_type: str = ""  # 事件类型（对应 LifeEventType）


# ---------------------------------------------------------------------------
# Item 54: 生命模拟事件类型扩展
# ---------------------------------------------------------------------------


class LifeEventType:
    """生命模拟事件类型枚举。"""

    READING = "reading"
    WALKING = "walking"
    COOKING = "cooking"
    THINKING = "thinking"
    CREATING = "creating"
    RESTING = "resting"
    OBSERVING = "observing"


LIFE_EVENT_WEIGHTS: dict[str, dict[str, float]] = {
    "reading": {"valence": 0.2, "arousal": -0.1, "share_tendency": 0.4},
    "walking": {"valence": 0.3, "arousal": 0.1, "share_tendency": 0.3},
    "cooking": {"valence": 0.2, "arousal": 0.2, "share_tendency": 0.5},
    "thinking": {"valence": 0.0, "arousal": -0.2, "share_tendency": 0.6},
    "creating": {"valence": 0.4, "arousal": 0.3, "share_tendency": 0.7},
    "resting": {"valence": 0.1, "arousal": -0.3, "share_tendency": 0.1},
    "observing": {"valence": 0.1, "arousal": 0.0, "share_tendency": 0.5},
}

# 事件类型关键词映射（用于从 LLM 输出推断事件类型）
_EVENT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "reading": ["读", "书", "阅读", "看书", "翻阅", "read", "book", "novel", "article"],
    "walking": ["走", "散步", "漫步", "路", "walk", "stroll", "hike", "wander"],
    "cooking": ["做饭", "烹饪", "厨房", "煮", "烤", "cook", "kitchen", "bak", "meal"],
    "thinking": ["想", "思考", "沉思", "冥想", "think", "ponder", "reflect", "contempl"],
    "creating": ["创作", "画", "写", "做", "制作", "creat", "draw", "writ", "craft", "paint", "compos"],
    "resting": ["休息", "睡", "躺", "放松", "rest", "sleep", "relax", "nap", "doze"],
    "observing": ["观察", "看", "注视", "望", "observ", "watch", "gaze", "notic"],
}

# Open-Meteo WMO 天气代码 → 中文描述映射
_WMO_CODE_MAP: dict[int, str] = {
    0: "晴", 1: "少云", 2: "多云", 3: "阴",
    45: "雾", 48: "冰雾",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "阵雨", 81: "中阵雨", 82: "大阵雨",
    95: "雷暴", 96: "冰雹雷暴", 99: "大冰雹雷暴",
}


@dataclass
class DailySchedule:
    """每日日程框架：上午/下午/晚上的活动 + 穿搭。"""

    date: str = ""  # YYYY-MM-DD

    # 上午
    morning_activity: str = ""
    morning_outfit: str = ""
    morning_outfit_style: str = ""

    # 下午
    afternoon_activity: str = ""
    afternoon_outfit: str = ""
    afternoon_outfit_style: str = ""

    # 晚上
    evening_activity: str = ""
    evening_outfit: str = ""
    evening_outfit_style: str = ""

    # 元信息
    weather_desc: str = ""
    generated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "morning_activity": self.morning_activity,
            "morning_outfit": self.morning_outfit,
            "morning_outfit_style": self.morning_outfit_style,
            "afternoon_activity": self.afternoon_activity,
            "afternoon_outfit": self.afternoon_outfit,
            "afternoon_outfit_style": self.afternoon_outfit_style,
            "evening_activity": self.evening_activity,
            "evening_outfit": self.evening_outfit,
            "evening_outfit_style": self.evening_outfit_style,
            "weather_desc": self.weather_desc,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DailySchedule":
        return cls(
            date=str(data.get("date", "")),
            morning_activity=str(data.get("morning_activity", "")),
            morning_outfit=str(data.get("morning_outfit", "")),
            morning_outfit_style=str(data.get("morning_outfit_style", "")),
            afternoon_activity=str(data.get("afternoon_activity", "")),
            afternoon_outfit=str(data.get("afternoon_outfit", "")),
            afternoon_outfit_style=str(data.get("afternoon_outfit_style", "")),
            evening_activity=str(data.get("evening_activity", "")),
            evening_outfit=str(data.get("evening_outfit", "")),
            evening_outfit_style=str(data.get("evening_outfit_style", "")),
            weather_desc=str(data.get("weather_desc", "")),
            generated_at=float(data.get("generated_at", 0.0)),
        )


@dataclass
class LifeSimulationState:
    """生命模拟的持久化状态。"""

    events: list[LifeEvent] = field(default_factory=list)  # 历史事件列表
    current_activity: str = ""  # 当前正在做的事
    last_simulation_time: float = 0.0  # 上次模拟时间
    last_outreach_time: float = 0.0  # 上次主动联系时间
    simulation_count: int = 0  # 总模拟次数
    outreach_count: int = 0  # 总主动联系次数
    enabled: bool = False  # 是否启用
    locked_until: float = 0.0  # 锁定到期时间戳（0 = 未锁定）
    locked_activity: str = ""  # 锁定期间的活动描述
    _pending_emotion_delta: dict = field(default_factory=dict)  # 待应用的情绪增量
    daily_schedule: DailySchedule | None = None  # 今日日程框架
    daily_schedule_history: list[DailySchedule] = field(default_factory=list)  # 历史日程

    def to_dict(self) -> dict[str, Any]:
        """序列化状态（只保留最近 20 个事件）。"""
        return {
            "events": [
                {
                    "text": e.text,
                    "mood": e.mood,
                    "urgency": e.urgency,
                    "timestamp": e.timestamp,
                    "wants_to_share": e.wants_to_share,
                    "shared": e.shared,
                }
                for e in self.events[-20:]
            ],
            "current_activity": self.current_activity,
            "last_simulation_time": self.last_simulation_time,
            "last_outreach_time": self.last_outreach_time,
            "simulation_count": self.simulation_count,
            "outreach_count": self.outreach_count,
            "locked_until": self.locked_until,
            "locked_activity": self.locked_activity,
            "daily_schedule": self.daily_schedule.to_dict() if self.daily_schedule else None,
            "daily_schedule_history": [s.to_dict() for s in self.daily_schedule_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LifeSimulationState":
        """从字典恢复状态。"""
        state = cls()
        state.current_activity = data.get("current_activity", "")
        state.last_simulation_time = data.get("last_simulation_time", 0.0)
        state.last_outreach_time = data.get("last_outreach_time", 0.0)
        state.simulation_count = data.get("simulation_count", 0)
        state.outreach_count = data.get("outreach_count", 0)
        state.locked_until = data.get("locked_until", 0.0)
        state.locked_activity = data.get("locked_activity", "")
        # 加载后清除已过期的锁
        import time as _time
        if state.locked_until > 0 and _time.time() >= state.locked_until:
            state.locked_until = 0.0
            state.locked_activity = ""
        for e in data.get("events", []):
            state.events.append(
                LifeEvent(
                    text=e.get("text", ""),
                    mood=e.get("mood", "neutral"),
                    urgency=float(e.get("urgency", 0.0)),
                    timestamp=float(e.get("timestamp", 0.0)),
                    wants_to_share=e.get("wants_to_share", False),
                    shared=e.get("shared", False),
                )
            )
        # 恢复日程框架
        ds_data = data.get("daily_schedule")
        if ds_data and isinstance(ds_data, dict):
            state.daily_schedule = DailySchedule.from_dict(ds_data)
        for ds_item in data.get("daily_schedule_history", []) or []:
            if isinstance(ds_item, dict):
                state.daily_schedule_history.append(DailySchedule.from_dict(ds_item))
        return state


LIFE_SIMULATION_PROMPT = """你是一个创意写作助手。请为以下虚构角色生成一个当前时刻的生活片段。

注意：你不是在扮演这个角色对话，而是在模拟她独处时的生活状态——她此刻在做什么、想什么、心情如何。
输出应该是第三人称视角的简短生活快照。

角色设定：
{persona_desc}

当前环境：
|- 时间：{time_desc}
|- 时段：{time_of_day} — 此时模拟角色可能在做什么通常是合理的
|- 角色情绪倾向：{emotion_desc}
|- 距离上次和朋友聊天：{last_chat_desc}
| - 最近在做：{recent_activity}
| - 时段上下文：{period_context}
| - 此时是否适合找朋友聊天：{sociable}

请根据角色设定，生成这个角色此刻可能在做什么、想什么。内容要符合角色的性格和习惯。
用 JSON 格式输出：
{{"activity": "正在做什么（简短）", "thought": "在想什么（简短）", "mood": "当前心情（一个词）", "wants_to_share": true/false, "share_reason": "如果想分享给朋友，原因（简短）", "urgency": 0.0-1.0}}
{conversation_context}"""

DAILY_SCHEDULE_PROMPT = """你是一个生活规划助手。请为以下角色规划今天的日程框架和穿搭。

## 角色设定
{persona_desc}

## 当前信息
- 日期：{date_str} {weekday}{holiday}
- 天气：{weather_desc}
- 角色情绪倾向：{emotion_desc}
- 昨日安排（避免完全重复）：{yesterday_summary}
- 近日日程参考：{recent_schedules}

## 最近对话上下文
{conversation_context}

## 要求
请规划今天上午、下午、晚上三个时段的大致安排和对应穿搭（深夜/凌晨为休息时段，不需要安排）：

1. 上午：通常包含上课/练习/出门等活动
2. 下午：根据角色设定安排合理活动
3. 晚上：通常是在家/练习/社交等活动

每个时段需要：
- activity: 该时段的大致安排（一句话，如「去表演学校上舞蹈和声乐课」）
- outfit: 该时段的穿搭描述（从里到外、从上到下，80-150字，符合天气和场景）
- outfit_style: 穿搭风格标签（从以下选择或自创）

穿搭风格参考：知性学院风、街头休闲风、温柔淑女风、酷飒中性风、慵懒居家风、运动活力风、日系森女风、法式优雅风、韩系甜美风、复古文艺风、极简都市风

## 输出格式
严格返回 JSON（不要 Markdown 代码块）：
{"morning": {"activity": "...", "outfit": "...", "outfit_style": "..."}, "afternoon": {...}, "evening": {...}}"""


class LifeSimulator:
    """管理 Sylanne 的模拟独立生活。

    通过后台异步循环定期调用 LLM 生成生活片段，
    当生成的事件标记为"想分享"时，触发主动联系。

    生命周期：
    1. configure() 注入外部依赖（LLM、回调等）
    2. start() 启动后台循环
    3. 循环中：_simulate_tick() → _build_prompt() → LLM → _parse_response()
    4. 如果事件 wants_to_share 且冷却期已过 → _do_outreach()
    5. stop() 停止循环
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {}
        self.state = LifeSimulationState()
        self._running = False
        self._task: asyncio.Task | None = None
        self._llm_caller: Callable[..., Awaitable[str]] | None = None  # LLM 调用回调
        self._outreach_callback: Callable[[str, str], Awaitable[None]] | None = (
            None  # 主动联系回调
        )
        self._emotion_getter: Callable[[], dict[str, float]] | None = (
            None  # 情绪状态获取
        )
        self._persona_getter: Callable[[], str] | None = None  # 角色描述获取
        self._memory_summary_getter: Callable[[], str] | None = None  # 记忆摘要获取
        self._weather_cache: str = ""  # 天气查询缓存（供 prompt 构建使用）

    @property
    def enabled(self) -> bool:
        return bool(self._config.get("sylanne_alpha_life_simulation_enabled", False))

    @property
    def interval_seconds(self) -> float:
        return max(
            60.0,
            float(
                self._config.get(
                    "sylanne_alpha_life_simulation_interval_seconds", 1800.0
                )
            ),
        )

    @property
    def outreach_cooldown_seconds(self) -> float:
        return max(
            300.0,
            float(
                self._config.get(
                    "sylanne_alpha_life_simulation_outreach_cooldown_seconds", 3600.0
                )
            ),
        )

    @property
    def _current_hour_allowed(self) -> bool:
        """根据当前时间和配置判断是否在允许时段内。"""
        import datetime
        now_hour = datetime.datetime.now().hour
        start = max(0, min(23, int(
            self._config.get("sylanne_alpha_life_simulation_start_hour", 9)
        )))
        end = max(0, min(23, int(
            self._config.get("sylanne_alpha_life_simulation_end_hour", 22)
        )))
        if start <= end:
            return start <= now_hour <= end
        # 跨天配置：start > end 表示允许时段跨越午夜
        return now_hour >= start or now_hour <= end

    @staticmethod
    def _time_of_day(dt) -> str:
        """返回当前时间段的自然语言描述。"""
        h = dt.hour
        if 5 <= h < 8:
            return "清晨"
        if 8 <= h < 12:
            return "上午"
        if 12 <= h < 14:
            return "中午"
        if 14 <= h < 18:
            return "下午"
        if 18 <= h < 22:
            return "晚上"
        return "深夜/凌晨"

    def _archive_today_schedule(self):
        """将当前日程推入历史，保留最近 N 天。"""
        if self.state.daily_schedule is None:
            return
        max_days = int(self._config.get(
            "sylanne_alpha_life_simulation_daily_schedule_history_days", 7
        ))
        if max_days <= 0:
            self.state.daily_schedule = None
            self.state.daily_schedule_history.clear()
            return
        self.state.daily_schedule_history.insert(0, self.state.daily_schedule)
        if len(self.state.daily_schedule_history) > max_days:
            self.state.daily_schedule_history = (
                self.state.daily_schedule_history[:max_days]
            )
        self.state.daily_schedule = None

    def _should_generate_daily_schedule(self, now: float) -> bool:
        """判断当前 tick 是否应生成今日日程框架。"""
        if not self._config.get(
            "sylanne_alpha_life_simulation_daily_schedule_enabled", False
        ):
            return False
        import datetime
        today = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        if self.state.daily_schedule and self.state.daily_schedule.date == today:
            return False
        schedule_hour = int(self._config.get(
            "sylanne_alpha_life_simulation_daily_schedule_hour", 6
        ))
        if datetime.datetime.fromtimestamp(now).hour < schedule_hour:
            return False
        return True

    def configure(
        self,
        llm_caller: Callable[..., Awaitable[str]] | None = None,
        outreach_callback: Callable[[str, str], Awaitable[None]] | None = None,
        emotion_getter: Callable[[], dict[str, float]] | None = None,
        persona_getter: Callable[[], str] | None = None,
        memory_summary_getter: Callable[[], str] | None = None,
        body_delta_callback: Callable[[dict[str, float]], None] | None = None,
        persist_callback: Callable[[], None] | None = None,
        conversation_context_getter: Callable[[], str] | None = None,
    ):
        """注入外部依赖。所有回调都是可选的。"""
        self._llm_caller = llm_caller
        self._outreach_callback = outreach_callback
        self._emotion_getter = emotion_getter
        self._persona_getter = persona_getter
        self._memory_summary_getter = memory_summary_getter
        self._body_delta_callback = body_delta_callback
        self._persist_callback = persist_callback
        self._conversation_context_getter = conversation_context_getter

    def _get_current_period_context(self, now: float) -> str:
        """根据当前时间获取日程框架中的时段上下文。锁定期间以锁定活动为准。"""
        import datetime

        # 锁定优先
        if now < self.state.locked_until and self.state.locked_activity:
            return f"锁定活动（正在进行）：{self.state.locked_activity}"

        schedule = self.state.daily_schedule
        if schedule is None:
            return ""

        hour = datetime.datetime.fromtimestamp(now).hour
        deep_night_start = int(self._config.get(
            "sylanne_alpha_life_simulation_daily_schedule_deep_night_hour", 22
        ))
        morning_start = int(self._config.get(
            "sylanne_alpha_life_simulation_daily_schedule_hour", 6
        ))

        # 深夜/清晨 → 无安排，直接返回固定休息上下文
        if hour >= deep_night_start or hour < morning_start:
            return "深夜时段，正在休息或睡觉"

        if hour < 12:
            period = "上午"
            activity = schedule.morning_activity
            outfit = schedule.morning_outfit
        elif hour < 18:
            period = "下午"
            activity = schedule.afternoon_activity
            outfit = schedule.afternoon_outfit
        else:
            period = "晚上"
            activity = schedule.evening_activity
            outfit = schedule.evening_outfit

        lines = [f"当前时段：{period}"]
        if activity:
            lines.append(f"时段大致安排：{activity}")
        if outfit:
            lines.append(f"当前穿着：{outfit}")
        return "\n".join(lines)

    async def _fetch_weather(self) -> str:
        """查询当前天气（Open-Meteo），失败时返回降级描述。"""
        if not self._config.get(
            "sylanne_alpha_life_simulation_daily_schedule_weather_enabled", True
        ):
            return ""
        import datetime
        lat = float(self._config.get(
            "sylanne_alpha_life_simulation_daily_schedule_weather_lat", 30.67
        ))
        lon = float(self._config.get(
            "sylanne_alpha_life_simulation_daily_schedule_weather_lon", 104.06
        ))
        try:
            import aiohttp
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,weather_code"
                f"&timezone=Asia/Shanghai"
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}")
                    data = await resp.json()
            current = data.get("current", {})
            temp = current.get("temperature_2m", None)
            code = current.get("weather_code", 0)
            weather_text = _WMO_CODE_MAP.get(code, "未知")
            if temp is not None:
                return f"{weather_text}，{int(temp)}°C"
            return f"{weather_text}"
        except Exception:
            month = datetime.datetime.now().month
            if 3 <= month <= 5:
                season = "春季"
            elif 6 <= month <= 8:
                season = "夏季"
            elif 9 <= month <= 11:
                season = "秋季"
            else:
                season = "冬季"
            return f"{season}，气温未知"

    def _build_daily_schedule_prompt(self, now: float) -> str:
        """构建日程框架生成的 LLM prompt。"""
        import datetime
        dt = datetime.datetime.fromtimestamp(now)
        date_str = dt.strftime("%Y-%m-%d")
        weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][dt.weekday()]

        # 星期/节假日提示
        holiday = ""
        try:
            import holidays as _holidays
            cn_holidays = _holidays.China()
            if dt.date() in cn_holidays:
                holiday = f"（{cn_holidays.get(dt.date())}假期）"
        except ImportError:
            if dt.weekday() >= 5:
                holiday = "（周末）"

        # 情绪
        emotion_desc = "平静"
        if self._emotion_getter:
            try:
                emo = self._emotion_getter()
                parts = []
                if emo.get("warmth", 0) > 0.3:
                    parts.append("温暖")
                if emo.get("tension", 0) > 0.3:
                    parts.append("紧绷")
                if emo.get("curiosity", 0) > 0.3:
                    parts.append("好奇")
                emotion_desc = "、".join(parts) if parts else "平静"
            except Exception:
                pass

        # 昨日摘要
        yesterday_summary = "无"
        if self.state.daily_schedule_history:
            prev = self.state.daily_schedule_history[0]
            parts = []
            if prev.morning_activity:
                parts.append(f"上午:{prev.morning_activity}")
            if prev.afternoon_activity:
                parts.append(f"下午:{prev.afternoon_activity}")
            if prev.evening_activity:
                parts.append(f"晚上:{prev.evening_activity}")
            if parts:
                yesterday_summary = "；".join(parts)

        # 近日日程参考（最近 3 天，排除昨日）
        recent_schedules = ""
        if len(self.state.daily_schedule_history) > 1:
            lines = []
            for s in self.state.daily_schedule_history[1:4]:
                acts = []
                if s.morning_activity:
                    acts.append(f"早:{s.morning_activity}")
                if s.afternoon_activity:
                    acts.append(f"下:{s.afternoon_activity}")
                if s.evening_activity:
                    acts.append(f"晚:{s.evening_activity}")
                if acts:
                    lines.append(f"{s.date}→{'；'.join(acts)}")
            if lines:
                recent_schedules = "\n".join(lines)

        # 人设
        persona_desc = "一个有自己生活节奏的虚构角色"
        if self._persona_getter:
            try:
                raw = self._persona_getter()
                if raw:
                    persona_desc = raw[:500]
            except Exception:
                pass

        # 对话上下文
        conversation_context = "无"
        if self._conversation_context_getter:
            try:
                ctx = self._conversation_context_getter()
                if ctx:
                    conversation_context = ctx[:500]
            except Exception:
                pass

        return DAILY_SCHEDULE_PROMPT.format(
            persona_desc=persona_desc,
            date_str=date_str,
            weekday=weekday,
            holiday=holiday,
            weather_desc=self._weather_cache,
            emotion_desc=emotion_desc,
            yesterday_summary=yesterday_summary,
            recent_schedules=recent_schedules or "无",
            conversation_context=conversation_context,
        )

    async def _generate_daily_schedule(self):
        """生成今日日程框架：查询天气→构建 prompt→LLM→解析→存储。"""
        import datetime

        # 1. 旧日程入史
        self._archive_today_schedule()

        # 2. 天气查询（缓存到 self._weather_cache 供 prompt 构建使用）
        self._weather_cache = await self._fetch_weather()

        # 3. 构建 prompt
        now = time.time()
        prompt = self._build_daily_schedule_prompt(now)

        # 4. LLM 调用（temperature=0.8 保持与生活事件一致的多样性）
        if not self._llm_caller:
            return
        try:
            response = await self._llm_caller(prompt, temperature=0.8)
        except Exception:
            return

        # 5. 解析 JSON
        try:
            text = str(response).strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start < 0 or end <= start:
                raise ValueError("no JSON found")
            payload = json.loads(text[start:end])
        except Exception:
            return

        # 6. 构建 DailySchedule
        def _get(block, key):
            return str(block.get(key, "")).strip()[:200] if isinstance(block, dict) else ""

        morning = payload.get("morning", {}) or {}
        afternoon = payload.get("afternoon", {}) or {}
        evening = payload.get("evening", {}) or {}

        dt_obj = datetime.datetime.fromtimestamp(now)
        self.state.daily_schedule = DailySchedule(
            date=dt_obj.strftime("%Y-%m-%d"),
            morning_activity=_get(morning, "activity"),
            morning_outfit=_get(morning, "outfit"),
            morning_outfit_style=_get(morning, "outfit_style"),
            afternoon_activity=_get(afternoon, "activity"),
            afternoon_outfit=_get(afternoon, "outfit"),
            afternoon_outfit_style=_get(afternoon, "outfit_style"),
            evening_activity=_get(evening, "activity"),
            evening_outfit=_get(evening, "outfit"),
            evening_outfit_style=_get(evening, "outfit_style"),
            weather_desc=self._weather_cache,
            generated_at=now,
        )

        # 7. 持久化
        if self._persist_callback:
            try:
                self._persist_callback()
            except Exception:
                pass

    def start(self):
        """启动后台模拟循环。"""
        if not self.enabled or self._running:
            return
        self._running = True
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._loop())
        except RuntimeError:
            pass

    def stop(self):
        """停止模拟循环。"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None

    async def _loop(self):
        """后台循环：以随机间隔模拟生活事件。注意静默时段。"""
        import random

        while self._running and self.enabled:
            try:
                # 时段跳过：不在允许时段时等待 30 分钟再检
                if not self._current_hour_allowed:
                    await asyncio.sleep(1800)
                    continue

                # 每日日程框架生成（在时段允许内，仅在满足条件时自动生成）
                if self._should_generate_daily_schedule(time.time()):
                    await self._generate_daily_schedule()

                # 锁定等待：锁定期间不生成新事件
                if time.time() < self.state.locked_until:
                    remaining = self.state.locked_until - time.time()
                    await asyncio.sleep(min(30, remaining))
                    if time.time() >= self.state.locked_until:
                        await self._simulate_tick()
                        # 锁定到期恢复后持久化
                        if self._persist_callback:
                            try:
                                self._persist_callback()
                            except Exception:
                                pass
                    continue

                base = self.interval_seconds
                jitter = random.uniform(0.4, 1.8)
                wait = base * jitter
                await asyncio.sleep(wait)
                if not self._running:
                    break
                await self._simulate_tick()
            except asyncio.CancelledError:
                break
            except Exception as _exc:
                import logging

                logging.getLogger(__name__).debug(
                    "life_simulation tick error: %s", _exc
                )
                await asyncio.sleep(60.0)

    async def _simulate_tick(self):
        """执行一次模拟周期。"""
        if not self._llm_caller:
            return

        now = time.time()
        self.state.last_simulation_time = now
        self.state.simulation_count += 1

        prompt = self._build_prompt(now)
        try:
            response = await self._llm_caller(prompt)
            event = self._parse_response(response, now)
        except Exception:
            return

        if event:
            self.state.events.append(event)
            self.state.current_activity = event.text
            if len(self.state.events) > 50:
                self.state.events = self.state.events[-30:]

            # Item 54: 根据事件类型应用情绪权重到 body_state
            emotion_weights = self._apply_event_emotion_weights(event)
            if emotion_weights.get("valence", 0.0) != 0.0 or emotion_weights.get("arousal", 0.0) != 0.0:
                self._apply_to_body_state(emotion_weights)

            # share_tendency 调制 wants_to_share
            share_tendency = emotion_weights.get("share_tendency", 0.0)
            if share_tendency > 0.5 and not event.wants_to_share:
                # 高分享倾向的事件类型可以覆盖 LLM 的判断
                import random
                if random.random() < share_tendency * 0.5:
                    event.wants_to_share = True

            # 静默时段：事件仍生成并记录，但不触发 outreach
            if not self._current_hour_allowed:
                event.wants_to_share = False

            if event.wants_to_share and self._should_outreach(now):
                await self._do_outreach(event, now)

            # 每次 tick 完成后持久化
            if self._persist_callback:
                try:
                    self._persist_callback()
                except Exception:
                    pass

    def _build_prompt(self, now: float) -> str:
        """构建 LLM 提示词，包含角色设定、时间、情绪、记忆等上下文。"""
        import datetime

        dt = datetime.datetime.fromtimestamp(now)
        time_desc = dt.strftime("%H:%M, %A")
        time_of_day = self._time_of_day(dt)
        sociable = "是" if self._current_hour_allowed else "否（静默时段）"

        emotion_desc = "neutral"
        if self._emotion_getter:
            try:
                emo = self._emotion_getter()
                parts = []
                if emo.get("warmth", 0) > 0.3:
                    parts.append("warm")
                if emo.get("tension", 0) > 0.3:
                    parts.append("tense")
                if emo.get("curiosity", 0) > 0.3:
                    parts.append("curious")
                emotion_desc = ", ".join(parts) if parts else "calm"
            except Exception:
                pass

        gap = (
            now - self.state.last_outreach_time
            if self.state.last_outreach_time > 0
            else 99999
        )
        if gap < 3600:
            last_chat_desc = "just now"
        elif gap < 86400:
            last_chat_desc = f"{int(gap / 3600)} hours ago"
        else:
            last_chat_desc = f"{int(gap / 86400)} days ago"

        recent = self.state.current_activity or "nothing in particular"

        persona_desc = "一个有自己生活节奏的虚构角色"
        if self._persona_getter:
            try:
                raw_persona = self._persona_getter()
                if raw_persona:
                    persona_desc = raw_persona[:500]
            except Exception:
                pass

        memory_summary = ""
        if self._memory_summary_getter:
            try:
                summary = self._memory_summary_getter()
                if summary:
                    memory_summary = f"\n最近聊天摘要：{summary[:300]}"
            except Exception:
                pass

        conversation_context = ""
        if self._conversation_context_getter:
            try:
                ctx = self._conversation_context_getter()
                if ctx:
                    conversation_context = f"\n最近对话上下文：\n{ctx[:500]}"
            except Exception:
                pass

        period_context = self._get_current_period_context(now)

        return (
            LIFE_SIMULATION_PROMPT.format(
                persona_desc=persona_desc,
                time_desc=time_desc,
                time_of_day=time_of_day,
                emotion_desc=emotion_desc,
                last_chat_desc=last_chat_desc,
                recent_activity=recent,
                period_context=period_context,
                sociable=sociable,
                conversation_context=conversation_context,
            )
            + memory_summary
        )

    def _parse_response(self, response: str, now: float) -> LifeEvent | None:
        """解析 LLM 响应为 LifeEvent。容错处理 JSON 格式。"""
        try:
            text = response.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start < 0 or end <= start:
                return None
            data = json.loads(text[start:end])
            activity = str(data.get("activity", ""))
            thought = str(data.get("thought", ""))
            combined = f"{activity}" if not thought else f"{activity}（{thought}）"
            event_type = self._infer_event_type(combined)
            return LifeEvent(
                text=combined[:200],
                mood=str(data.get("mood", "neutral"))[:20],
                urgency=max(0.0, min(1.0, float(data.get("urgency", 0.0)))),
                timestamp=now,
                wants_to_share=bool(data.get("wants_to_share", False)),
                event_type=event_type,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    @staticmethod
    def _infer_event_type(text: str) -> str:
        """从事件文本推断事件类型。

        通过关键词匹配确定最可能的事件类型。
        如果无法匹配，返回空字符串。
        """
        text_lower = text.lower()
        best_type = ""
        best_score = 0
        for event_type, keywords in _EVENT_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_type = event_type
        return best_type

    def _apply_event_emotion_weights(self, event: LifeEvent) -> dict[str, float]:
        """根据事件类型应用情绪权重，返回 body_state 调制值。

        返回的 dict 包含 valence 和 arousal 的增量，
        以及 share_tendency 用于调制 wants_to_share 判断。
        """
        if not event.event_type or event.event_type not in LIFE_EVENT_WEIGHTS:
            return {"valence": 0.0, "arousal": 0.0, "share_tendency": 0.0}
        return dict(LIFE_EVENT_WEIGHTS[event.event_type])

    def _apply_to_body_state(self, weights: dict[str, float]) -> None:
        """将情绪权重增量应用到当前 body_state。

        通过 body_delta_callback 直接注入到 host 的身体状态。
        """
        delta = {
            "valence": weights.get("valence", 0.0),
            "arousal": weights.get("arousal", 0.0),
        }
        cb = getattr(self, "_body_delta_callback", None)
        if cb:
            try:
                cb(delta)
            except Exception:
                pass
        self.state._pending_emotion_delta = delta

    def _should_outreach(self, now: float) -> bool:
        """检查是否允许主动联系（时段、冷却期、回调）。"""
        if not self._current_hour_allowed:
            return False
        if not self._outreach_callback:
            return False
        if self.state.last_outreach_time > 0:
            gap = now - self.state.last_outreach_time
            if gap < self.outreach_cooldown_seconds:
                return False
        return True

    async def _do_outreach(self, event: LifeEvent, now: float):
        """基于生活事件触发主动联系。"""
        if not self._outreach_callback:
            return
        try:
            reason = f"[life_event] {event.text}"
            await self._outreach_callback(reason, event.mood)
            event.shared = True
            self.state.last_outreach_time = now
            self.state.outreach_count += 1
        except Exception:
            pass

    def pending_share_events(self) -> list[LifeEvent]:
        """获取想分享但尚未分享的事件列表。"""
        return [e for e in self.state.events if e.wants_to_share and not e.shared]

    def recent_context_for_prompt(self, limit: int = 3) -> str:
        """获取近期生活事件作为 LLM 提示词注入的上下文。"""
        # 锁定期间优先返回锁定活动
        if self.state.locked_until > 0 and time.time() < self.state.locked_until:
            if self.state.locked_activity:
                return f"（Sylanne 正在：{self.state.locked_activity}）"

        lines = []

        # 日程框架 → 当前时段上下文
        period_ctx = self._get_current_period_context(time.time())
        if period_ctx:
            lines.append(f"（{period_ctx}）")

        recent = [e for e in self.state.events[-10:] if e.text]
        if not recent:
            return "\n".join(lines) if lines else ""
        if self.state.current_activity:
            lines.append(f"（Sylanne 最近的生活：{self.state.current_activity}）")
        for e in recent[-limit:]:
            lines.append(f"（{e.mood}：{e.text}）")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return self.state.to_dict()

    def from_dict(self, data: dict[str, Any]):
        self.state = LifeSimulationState.from_dict(data)


# ---------------------------------------------------------------------------
# Item 31: 梦境生成系统
# ---------------------------------------------------------------------------


class DreamGenerator:
    """梦境生成：离线时基于记忆和伤痕生成碎片化梦境。"""

    def __init__(self):
        self._last_dream: str = ""
        self._dream_time: float = 0

    def should_dream(self, offline_hours: float) -> bool:
        """离线超过 6h 且距上次做梦超过 12h。"""
        return offline_hours > 6 and (time.time() - self._dream_time > 43200)

    def generate_dream(
        self,
        recent_memories: list[str],
        scar_count: int,
        void_pressure: float,
    ) -> str:
        """基于记忆碎片和状态生成梦境叙事。"""
        import random

        # 从记忆中随机抽取 2-3 条作为素材
        fragments = (
            random.sample(recent_memories, min(3, len(recent_memories)))
            if recent_memories
            else ["模糊的影子"]
        )

        # 根据伤痕数量决定梦境基调
        if scar_count > 5:
            tone = "不安的"
        elif void_pressure > 2:
            tone = "压抑的"
        else:
            tone = "平静的"

        # 拼接碎片化梦境
        dream_parts = [f"做了一个{tone}梦"]
        for frag in fragments:
            # 截取记忆片段的关键词
            short = frag[:20] if len(frag) > 20 else frag
            dream_parts.append(f"梦里出现了关于「{short}」的画面")

        if void_pressure > 3:
            dream_parts.append("梦的最后有什么想说却说不出口")

        self._last_dream = "……".join(dream_parts)
        self._dream_time = time.time()
        return self._last_dream

    def has_dream_to_share(self) -> bool:
        return bool(self._last_dream) and time.time() - self._dream_time < 3600

    def consume_dream(self) -> str:
        dream = self._last_dream
        self._last_dream = ""
        return dream
