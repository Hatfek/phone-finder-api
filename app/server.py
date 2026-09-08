import asyncio
import logging
import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.types import Command

from app.api_models import (
    THREAD_ID_PATTERN,
    AnswerRequest,
    ResetRequest,
    ReviseRequest,
    SlotCatalogue,
    SlotOption,
    StartRequest,
    TurnResponse,
)
from app.config import get_settings
from app.fallbacks import SLOT_LABELS, revision_questions
from app.graph import get_graph
from app.limits import RateLimited, RateLimiter, ThreadBusy, TooManyTurns, TurnGuard
from app.nodes import revision_update
from app.schemas import REVISABLE_SLOTS, Filters
from app.serialize import to_response

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(title="phone-finder", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["content-type"],
    max_age=600,
)

ThreadId = Annotated[str, Path(pattern=THREAD_ID_PATTERN)]


def client_ip(request: Request) -> str:
    client = request.client
    return client.host if client else "unknown"


_limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)
_global_limiter = RateLimiter(
    settings.global_rate_limit_requests, settings.global_rate_limit_window_seconds
)
_turns = TurnGuard(settings.max_concurrent_turns)


@app.middleware("http")
async def guard_request(request: Request, call_next):
    """Caps the body at MAX_REQUEST_BYTES and applies the global per-IP ceiling.

    The ceiling is separate from, and looser than, the per-turn limit below: it stops a
    client from hammering the cheap routes without touching the turn budget.
    """
    length = request.headers.get("content-length")
    if length is not None:
        try:
            if int(length) > settings.max_request_bytes:
                return JSONResponse(status_code=413, content={"detail": "request too large"})
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "bad content-length"})

    try:
        _global_limiter.check(client_ip(request))
    except RateLimited as exc:
        return JSONResponse(
            status_code=429,
            content={"detail": "too many requests"},
            headers={"Retry-After": str(max(1, int(exc.retry_after) + 1))},
        )

    return await call_next(request)


def rate_limit(request: Request) -> None:
    try:
        _limiter.check(client_ip(request))
    except RateLimited as exc:
        raise HTTPException(
            status_code=429,
            detail="too many requests",
            headers={"Retry-After": str(max(1, int(exc.retry_after) + 1))},
        ) from exc


TurnLimit = Depends(rate_limit)


def _config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"turn:{thread_id}",
        "tags": [f"thread:{thread_id}"],
        "recursion_limit": 40,
    }


ASK_NODE = "ask"


async def _snapshot(thread_id: str) -> tuple[dict, tuple[str, ...]]:
    graph = await get_graph()
    snapshot = await graph.aget_state(_config(thread_id))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="thread not found")
    return snapshot.values, tuple(snapshot.next or ())


async def _after_failed_turn(thread_id: str) -> tuple[dict, tuple[str, ...], tuple[str, ...]]:
    try:
        state, pending = await _snapshot(thread_id)
    except HTTPException as exc:
        raise HTTPException(
            status_code=503,
            detail="the turn could not be completed",
            headers={"Retry-After": "5"},
        ) from exc
    return state, pending, ("turn",)


async def _run(thread_id: str, payload) -> tuple[dict, tuple[str, ...], tuple[str, ...]]:
    """Runs one turn. Returns the state, the graph's pending nodes and any degradation."""
    graph = await get_graph()
    try:
        async with _turns.hold(thread_id):
            await asyncio.wait_for(
                graph.ainvoke(payload, config=_config(thread_id)),
                timeout=settings.turn_timeout,
            )
    except ThreadBusy as exc:
        raise HTTPException(
            status_code=409, detail="a turn is already running on this thread"
        ) from exc
    except TooManyTurns as exc:
        raise HTTPException(
            status_code=503, detail="server busy", headers={"Retry-After": "30"}
        ) from exc
    except asyncio.TimeoutError:
        log.warning("turn timeout on %s after %ss", thread_id, settings.turn_timeout)
        return await _after_failed_turn(thread_id)
    except Exception:
        log.exception("turn failed on %s", thread_id)
        return await _after_failed_turn(thread_id)

    state, pending = await _snapshot(thread_id)
    return state, pending, ()


@app.get("/health")
async def health() -> dict:
    checkpointer = True
    try:
        await get_graph()
    except Exception:
        log.exception("checkpointer unavailable")
        checkpointer = False
    return {
        "status": "ok" if checkpointer else "degraded",
        "model": settings.ollama_model,
        "price_reference": settings.price_reference,
        "checkpointer": checkpointer,
        "debug_ui": settings.debug_ui,
    }


@app.post("/threads", response_model=TurnResponse, dependencies=[TurnLimit])
async def start_thread(body: StartRequest) -> TurnResponse:
    thread_id = uuid.uuid4().hex[:12]
    state, pending, degraded = await _run(thread_id, {"profile": body.profile})
    return to_response(thread_id, state, bool(pending), degraded)


@app.post(
    "/threads/{thread_id}/answer",
    response_model=TurnResponse,
    dependencies=[TurnLimit],
)
async def answer(thread_id: ThreadId, body: AnswerRequest) -> TurnResponse:
    state, pending = await _snapshot(thread_id)
    if not pending:
        return to_response(thread_id, state, False)

    if ASK_NODE not in pending:
        log.warning("thread %s is mid-run at %s; finishing it before answering", thread_id, pending)
        state, pending, degraded = await _run(thread_id, None)
        return to_response(thread_id, state, bool(pending), (*degraded, "resumed"))

    state, pending, degraded = await _run(thread_id, Command(resume=body.answer))
    return to_response(thread_id, state, bool(pending), degraded)


@app.post(
    "/threads/{thread_id}/revise",
    response_model=TurnResponse,
    dependencies=[TurnLimit],
)
async def revise_thread(thread_id: ThreadId, body: ReviseRequest) -> TurnResponse:
    """Change one earlier answer without losing the ones that came after it.

    Unlike `/reset`, which drops every filter, this clears exactly the named slot. The
    patch is written into the checkpoint as if `ask` had just run, so the graph picks up
    at `plan` and re-searches on the corrected filters — whether the thread was waiting
    at a question or had already finished.
    """
    state, _ = await _snapshot(thread_id)
    filters = Filters.model_validate(state.get("filters", {}))
    if not filters.is_revisable(body.slot):
        raise HTTPException(status_code=409, detail=f"{body.slot} is not set on this thread")

    graph = await get_graph()
    await graph.aupdate_state(
        _config(thread_id), revision_update(state, body.slot, body.answer), as_node=ASK_NODE
    )
    state, pending, degraded = await _run(thread_id, None)
    return to_response(thread_id, state, bool(pending), degraded)


@app.get("/slots", response_model=SlotCatalogue)
async def slots() -> SlotCatalogue:
    """The canned question behind every revisable filter, so the client invents none."""
    questions = revision_questions()
    return SlotCatalogue(
        slots=[
            SlotOption(
                slot=slot,
                label=SLOT_LABELS[slot],
                question=questions[slot].question,
                options=questions[slot].options,
                hint=questions[slot].hint,
            )
            for slot in REVISABLE_SLOTS
        ]
    )


@app.post(
    "/threads/{thread_id}/reset",
    response_model=TurnResponse,
    dependencies=[TurnLimit],
)
async def reset_thread(thread_id: ThreadId, body: ResetRequest | None = None) -> TurnResponse:
    state, _ = await _snapshot(thread_id)
    profile = ((body.profile if body else None) or state.get("profile", "")).strip()
    if not profile:
        raise HTTPException(status_code=400, detail="nothing to restart from")

    graph = await get_graph()
    if graph.checkpointer is not None:
        await graph.checkpointer.adelete_thread(thread_id)

    state, pending, degraded = await _run(thread_id, {"profile": profile})
    return to_response(thread_id, state, bool(pending), degraded)


@app.get("/threads/{thread_id}", response_model=TurnResponse)
async def get_thread(thread_id: ThreadId) -> TurnResponse:
    state, pending = await _snapshot(thread_id)
    return to_response(thread_id, state, bool(pending))


@app.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(thread_id: ThreadId) -> None:
    graph = await get_graph()
    if graph.checkpointer is not None:
        await graph.checkpointer.adelete_thread(thread_id)

