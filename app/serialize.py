from app.api_models import TurnResponse
from app.config import get_settings
from app.schemas import Filters, PhoneResult, Question

DEGRADED_WARNINGS = {
    "llm": "The local model is unreachable, so questions are the standard ones "
           "and results carry no explanations.",
    "search": "Live search is unavailable right now, so this turn found no new listings.",
    "fetch": "The retailer pages could not be read this turn, so any phones shown "
             "are from an earlier turn.",
    "turn": "That turn did not finish. Your saved answers are intact — try again.",
    "resumed": "The previous turn had stalled, so we finished it first. Please answer once more.",
}


def to_response(
    thread_id: str,
    state: dict,
    awaiting: bool,
    extra_degraded: tuple[str, ...] = (),
) -> TurnResponse:
    settings = get_settings()
    total = settings.max_questions
    question = state.get("question")
    done = not awaiting or question is None
    asked = state.get("question_count", 0)

    degraded = [tag for tag in state.get("degraded", []) if tag in DEGRADED_WARNINGS]
    for tag in extra_degraded:
        if tag not in degraded:
            degraded.append(tag)

    return TurnResponse(
        thread_id=thread_id,
        step="done" if done else f"{min(asked + 1, total)} of {total}",
        done=done,
        price_reference=settings.price_reference,
        notice=state.get("notice"),
        ask_question=None if done else Question.model_validate(question),
        show_phones=[PhoneResult.model_validate(r) for r in state.get("results", [])],
        filters=Filters.model_validate(state.get("filters", {})),
        degraded=degraded,
        warnings=[DEGRADED_WARNINGS[tag] for tag in degraded],
    )
