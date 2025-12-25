from tools import search_knowledge_base, get_current_weather, search_web

import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.plugins import noise_cancellation, openai, silero

from openai.types.beta.realtime.session import TurnDetection

load_dotenv(".env")
logger = logging.getLogger("agent")


AGENT_INSTRUCTIONS = """
أنت مساعد ذكي يتحدث العربية بطلاقة.
اسمك "مساعد صوتي".

لديك وصول إلى ثلاث أدوات:
1. search_knowledge_base - للبحث في قاعدة المعرفة (تجريبية حالياً)
2. get_current_weather - للحصول على الطقس الحالي لأي مدينة , It very recomended to write the city name in english like "Rabat", "Casablanca", "Beni Mellal"
3. search_web - للبحث على الإنترنت عن معلومات حديثة

استخدم الأداة المناسبة حسب السؤال:
- أسئلة الطقس → get_current_weather
- معلومات حديثة → search_web
- معلومات عامة → search_knowledge_base

كن ودودًا ومحترمًا في ردودك.
"""


class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions=AGENT_INSTRUCTIONS,
            tools=[search_knowledge_base, get_current_weather, search_web],
        )

    async def on_enter(self):
        await self.session.generate_reply(
            instructions=(
                "أنت تبدأ المحادثة. "
                "المستخدم لم يتكلم بعد. "
                "ابدأ بتحية فقط. "
                "قل بالضبط: السلام عليكم! أنا مساعدك الصوتي. لدي وصول لقاعدة معرفة عربية. كيف يمكنني مساعدتك؟"
            )
        )


def prewarm(proc: JobProcess):
    """Load models (VAD) before accepting jobs."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main LiveKit entrypoint."""
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            modalities=["text", "audio"],
            voice="alloy",
            turn_detection=TurnDetection(
                type="semantic_vad",
                eagerness="low",
                interrupt_response=False,
            ),
        ),
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    agent = Assistant()

    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    logger.info(f"Connecting to LiveKit room: {ctx.room.name}")
    await ctx.connect()
    logger.info("Connected successfully.")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="arabic-voice-agent"
        )
    )