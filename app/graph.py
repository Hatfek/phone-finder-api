import asyncio

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.nodes import annotate, ask, extract, intake, plan, search
from app.schemas import GraphState


def route_after_annotate(state: GraphState) -> str:
    if state.get("done") or state.get("question") is None:
        return END
    if state.get("question_count", 0) >= get_settings().max_questions:
        return END
    return "ask"


_graph = None
_lock = asyncio.Lock()


def _build():
    builder = StateGraph(GraphState)
    builder.add_node("intake", intake)
    builder.add_node("plan", plan)
    builder.add_node("search", search)
    builder.add_node("extract", extract)
    builder.add_node("annotate", annotate)
    builder.add_node("ask", ask)

    builder.add_edge(START, "intake")
    builder.add_edge("intake", "plan")
    builder.add_edge("plan", "search")
    builder.add_edge("search", "extract")
    builder.add_edge("extract", "annotate")
    builder.add_conditional_edges("annotate", route_after_annotate, {"ask": "ask", END: END})
    builder.add_edge("ask", "plan")

    return builder


async def get_graph():
    global _graph
    if _graph is not None:
        return _graph
    async with _lock:
        if _graph is None:
            conn = await aiosqlite.connect(
                get_settings().checkpoint_path, check_same_thread=False
            )
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")
            await conn.commit()
            saver = AsyncSqliteSaver(conn)
            await saver.setup()
            _graph = _build().compile(checkpointer=saver)
    return _graph
