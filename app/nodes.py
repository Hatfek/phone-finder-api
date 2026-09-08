import asyncio
import json
import logging
import re

from langgraph.types import interrupt

from app.brands import (
    brand_of,
    detect_brands,
    implied_os,
    matches_brands,
    query_words,
)
from app.config import get_settings
from app.fallbacks import FALLBACK_QUESTIONS, price_options
from app.prompts import ANNOTATE_SYSTEM, INTAKE_SYSTEM, PLAN_SYSTEM, QUERIES_SYSTEM
from app.schemas import (
    BRAND_SLOTS,
    DEFAULTS,
    SLOTS,
    Filters,
    FitFactor,
    GraphState,
    PhoneResult,
    PlannerOutput,
    Question,
    SearchQueries,
    SlotFilters,
    WhyBatch,
)
from app.structured import structured_call
from app.tools import (
    extract_prices,
    fetch_page,
    is_listing_page,
    price_band,
    web_search,
)

log = logging.getLogger(__name__)

DEGRADE_LLM = "llm"
DEGRADE_SEARCH = "search"
DEGRADE_FETCH = "fetch"

OUTAGE_NOTICE = (
    "We could not reach the retailer pages for this turn, so no new prices were read. "
    "Your answers are safe — try again in a moment."
)

MIN_PHONE_PRICE_USD = 90
JUNK_RE = re.compile(
    r"\b(case|cover|charger|cable|protector|glasses|watch|tablet|holder|mount|"
    r"adapter|earbuds|headphone|sim|refurbished lot|bundle|gimbal|stabilizer|"
    r"tripod|slider|lens|selfie stick|ring light|power bank|powerbank|memory card|"
    r"sd card|microphone|gamepad|controller|speaker|dock|cradle|lanyard|strap|"
    r"collar|tracker|pet |doorbell|dashcam|dash cam|security camera|projector|"
    r"smart home|drone|printer|router|keyboard|mouse)\b",
    re.IGNORECASE,
)

PHONE_RE = re.compile(
    r"\b(phone|smartphone|iphone|galaxy|pixel|moto|redmi|poco|xperia|nord|"
    r"oneplus|realme)\b",
    re.IGNORECASE,
)

APPLE_RE = re.compile(r"\b(apple|iphone)\b", re.IGNORECASE)

KNOWN_BRANDS = (
    "samsung", "galaxy", "google", "pixel", "apple", "iphone", "motorola", "moto",
    "oneplus", "nothing", "xiaomi", "redmi", "poco", "sony", "xperia", "asus", "honor",
    "huawei", "oppo", "vivo", "realme", "nokia", "tcl",
)
BRAND_RE = re.compile(r"\b(" + "|".join(KNOWN_BRANDS) + r")\b", re.IGNORECASE)

PRIORITY_HINTS = {
    "camera": ("camera", "megapixel", "photo", "zoom", "ois", "ultra"),
    "battery": ("battery", "mah", "power", "endurance", "day battery", "long lasting"),
    "gaming": ("gaming", "snapdragon", "120hz", "144hz", "performance", "ram"),
    "allround": (),
}

PRIORITY_WORDS = {
    "camera": "camera",
    "photo": "camera",
    "battery": "battery",
    "charge": "battery",
    "gaming": "gaming",
    "game": "gaming",
    "everything": "allround",
    "allround": "allround",
    "all-round": "allround",
}

SIZE_HINTS = {
    "compact": ("mini", "compact", " se", "flip"),
    "large": ("ultra", "max", "plus", "note", "fold", "6.7", "6.8"),
}

NO_CAP_WORDS = ("no limit", "no budget", "any budget", "unlimited", "whatever it takes")
PRICE_CONTEXT = ("$", "usd", "dollar", "budget", "spend", "under", "below", "max", "limit", "price")


def _degraded(state: GraphState, *tags: str) -> list[str]:
    """The turn's degradation tags so far, plus `tags`. `plan` starts each turn empty."""
    out = [tag for tag in state.get("degraded", []) if isinstance(tag, str)]
    for tag in tags:
        if tag not in out:
            out.append(tag)
    return out


def _union(base: list[str], extra: list[str]) -> list[str]:
    out = list(base)
    for item in extra:
        if item not in out:
            out.append(item)
    return out


def _apply_brands(filters: Filters, text: str, *, direct: bool) -> Filters:
    prefer, exclude, mentioned = detect_brands(text, direct=direct)
    if not (prefer or exclude or mentioned):
        return filters

    data = filters.model_dump()
    data["exclude_brands"] = _union(filters.exclude_brands, exclude)
    data["brands"] = [
        b for b in _union(filters.brands, prefer) if b not in data["exclude_brands"]
    ]
    data["soft_brands"] = [
        b
        for b in _union(filters.soft_brands, mentioned)
        if b not in data["brands"] and b not in data["exclude_brands"]
    ]
    merged = Filters.model_validate(data)

    hinted = implied_os(merged.brands)
    if hinted and not merged.is_set("os"):
        merged = merged.with_slot("os", hinted, resolve=False)
    return merged


def _os_signal(low: str) -> str | None:
    if any(w in low for w in ("ios", "iphone", "apple")):
        return "iOS"
    if "android" in low:
        return "Android"
    return None


def _price_signal(low: str, asked: bool) -> tuple[bool, int | None]:
    if any(w in low for w in NO_CAP_WORDS):
        return True, None
    if not asked and not any(w in low for w in PRICE_CONTEXT):
        return False, None
    digits = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", low)]
    digits = [d for d in digits if d >= MIN_PHONE_PRICE_USD]
    if not digits:
        return False, None
    return True, max(digits)


def _priority_signal(low: str) -> str | None:
    return next((v for k, v in PRIORITY_WORDS.items() if k in low), None)


def _size_signal(low: str) -> str | None:
    if any(w in low for w in ("compact", "small", "one hand", "one-hand", "pocket")):
        return "compact"
    if any(w in low for w in ("large", "big screen", "big display", "bigger")):
        return "large"
    return None


def _merge_answer(filters: Filters, answer: str, slot: str | None) -> Filters:
    low = answer.lower()
    signals: dict[str, object] = {}

    os_value = _os_signal(low)
    if os_value:
        signals["os"] = os_value

    found_price, price = _price_signal(low, slot == "max_price_usd")
    if found_price:
        signals["max_price_usd"] = price

    priority = _priority_signal(low)
    if priority:
        signals["priority"] = priority

    size = _size_signal(low)
    if size:
        signals["size"] = size

    merged = filters
    for name, value in signals.items():
        merged = merged.with_slot(name, value)

    if slot in SLOTS and not signals:
        merged = merged.with_slot(slot, DEFAULTS[slot])

    return _apply_brands(merged, answer, direct=True)


def _next_slot(filters: Filters) -> str | None:
    unfilled = filters.unfilled_slots()
    return unfilled[0] if unfilled else None


def revision_update(state: GraphState, slot: str, answer: str | None) -> GraphState:
    """The state patch that changes one earlier answer and keeps every later one.

    Written into the checkpoint as if `ask` had just run, so the next pass starts at
    `plan` and re-searches on the corrected filters. Three moves, and each is load-bearing:

    - `clear_slot` drops that one constraint and unsettles it. Every other slot,
      including ones answered after it, is copied through untouched.
    - `merged_answers` jumps to the end of `answers`, so `plan` cannot re-fold the reply
      that set the slot we just cleared. Without this the revision is undone on the very
      next pass.
    - a new answer is appended for `plan` to read, or, with no answer, one question is
      handed back so the loop can ask that slot again even on a thread that had finished.
    """
    filters = Filters.model_validate(state.get("filters", {}))
    answers = list(state.get("answers", []))
    text = (answer or "").strip()

    update: GraphState = {
        "filters": filters.clear_slot(slot).model_dump(),
        "merged_answers": len(answers),
        "relaxed": False,
        "widened": False,
        "notice": None,
        "done": False,
    }
    if text:
        update["answers"] = [*answers, text]
        update["question"] = {**FALLBACK_QUESTIONS[_ask_slot(slot)].model_dump(), "slot": slot}
    else:
        update["question"] = None
        if slot in SLOTS:
            update["question_count"] = max(0, state.get("question_count", 0) - 1)
    return update


def _ask_slot(slot: str) -> str:
    """The slot whose canned question shapes a revision reply. Brands ride on `os`."""
    return "os" if slot in BRAND_SLOTS else slot


async def intake(state: GraphState) -> GraphState:
    """Entry node: read the free-text profile into a starting set of `Filters`.

    Runs once per thread, ahead of the loop. One structured call maps the profile onto
    the four slots, and `_apply_brands` reads brand preferences off the same text by
    cue, so brand filters survive even when the model does not answer.

    Reads `profile`. Writes `filters` plus the per-thread counters the rest of the loop
    relies on: `answers`, `question_count`, `results`, `done`, `widened`, `relaxed`.
    Degrades with `llm` to an empty `Filters` when the model is unreachable.
    """
    profile = state.get("profile", "")
    parsed = await structured_call(
        SlotFilters, INTAKE_SYSTEM, profile, run_name="intake"
    )
    filters = Filters().fill_from(parsed.value or SlotFilters())
    filters = _apply_brands(filters, profile, direct=False)
    return {
        "filters": filters.model_dump(),
        "answers": [],
        "merged_answers": 0,
        "question_count": 0,
        "results": [],
        "done": False,
        "widened": False,
        "relaxed": False,
        "degraded": [DEGRADE_LLM] if parsed.unreachable else [],
    }


async def plan(state: GraphState) -> GraphState:
    """Loop head: fold the last answer into `filters`, then choose the next slot to ask.

    Runs on every pass — first from `intake`, then from `ask` on each return trip — which
    is why this is the node that clears `degraded` for the turn; `search`, `extract` and
    `annotate` append to it afterwards.

    Reads `filters`, `answers`, `merged_answers`, `question`, `question_count`,
    `relaxed`, `widened`. Writes the merged `filters` and either a `question` to keep
    looping or `done: True` once every slot is settled or `max_questions` is reached.
    The planner may only fill slots the shopper has not settled, so a small model cannot
    blank an earlier answer.

    Only answers past `merged_answers` are folded in, and the counter moves to the end
    of the list. Re-reading `answers[-1]` on every pass would undo a revision (§4.27):
    `/revise` clears one slot, and a stale re-merge of the answer that set it would put
    it straight back.
    Degrades with `llm` to `FALLBACK_QUESTIONS[slot]` when the model is unreachable.
    """
    filters = Filters.model_validate(state.get("filters", {}))
    answers = state.get("answers", [])
    merged_answers = min(state.get("merged_answers", 0), len(answers))
    question = state.get("question")
    relaxed = state.get("relaxed", False)
    widened = state.get("widened", False)

    fresh = answers[merged_answers:]
    for index, answer in enumerate(fresh):
        slot_asked = (question or {}).get("slot") if index == len(fresh) - 1 else None
        updated = _merge_answer(filters, answer, slot_asked)
        if updated != filters:
            relaxed = False
            widened = False
        filters = updated

    slot = _next_slot(filters)
    count = state.get("question_count", 0)
    if slot is None or count >= get_settings().max_questions:
        return {
            "filters": filters.model_dump(),
            "merged_answers": len(answers),
            "question": None,
            "done": True,
            "relaxed": relaxed,
            "widened": widened,
            "degraded": [],
        }

    results = state.get("results", [])
    payload = {
        "profile": state.get("profile", "")[:800],
        "answers": answers,
        "filters": filters.model_dump(),
        "slot_to_ask": slot,
        "phones_found": [{"name": r["name"], "price_usd": r["price_usd"]} for r in results][:10],
    }
    planned = await structured_call(
        PlannerOutput,
        PLAN_SYSTEM,
        json.dumps(payload, ensure_ascii=False),
        run_name="plan",
    )

    proposal = planned.value
    if proposal and proposal.next_question and proposal.next_question.slot == slot:
        question_obj = proposal.next_question
        filters = filters.fill_from(proposal.filters, skip=(slot,))
    else:
        question_obj = FALLBACK_QUESTIONS[slot]
        if planned.unreachable:
            log.warning("plan fell back to the canned question for %s", slot)

    question_obj = _validate_options(question_obj, results)
    return {
        "filters": filters.model_dump(),
        "merged_answers": len(answers),
        "question": question_obj.model_dump(),
        "done": False,
        "relaxed": relaxed,
        "widened": widened,
        "degraded": [DEGRADE_LLM] if planned.unreachable else [],
    }


def _validate_options(question: Question, results: list[dict]) -> Question:
    if question.slot != "max_price_usd":
        return question
    prices = [r["price_usd"] for r in results]
    return question.model_copy(update={"options": price_options(prices)})


OS_TERMS = {"iOS": "Apple iPhone", "Android": "Android"}


def _query_seed(filters: Filters) -> str:
    bits = []
    brand = query_words(filters.brands)
    if brand:
        bits.append(brand)
    if filters.os in OS_TERMS and not (brand and filters.os == "iOS"):
        bits.append(OS_TERMS[filters.os])
    if filters.priority == "camera":
        bits.append("camera")
    elif filters.priority == "battery":
        bits.append("big battery")
    elif filters.priority == "gaming":
        bits.append("gaming")
    if filters.os != "iOS":
        bits.append("smartphone")
    if filters.size == "compact":
        bits.append("compact")
    elif filters.size == "large":
        bits.append("large screen")
    return " ".join(bits)


def _os_matches(name: str, os: str) -> bool:
    if os == "iOS":
        return bool(APPLE_RE.search(name))
    if os == "Android":
        return not APPLE_RE.search(name)
    return True


PRIORITY_LABELS = {
    "camera": "camera",
    "battery": "battery life",
    "gaming": "gaming",
    "allround": "all-round",
}
SIZE_LABELS = {"compact": "compact", "large": "large screen"}
OS_LABELS = {"iOS": "an iPhone", "Android": "an Android phone"}


def _usd(value: float) -> str:
    return f"${value:,.0f}" if float(value).is_integer() else f"${value:,.2f}"


def _fit_factors(phone: PhoneResult, filters: Filters) -> list[FitFactor]:
    """Break a phone\'s rank into the filters that earned it, best-scoring first.

    Every branch here is one term of `_fit_score`, which is the sum of the points below.
    The labels name the filter the shopper actually set, so a `why` line can say
    "your $500 budget" rather than a sentence that would fit any phone on the page.
    """
    name = phone.name.lower()
    factors: list[FitFactor] = []

    brand = brand_of(name)
    if brand and brand in filters.brands:
        factors.append(FitFactor(
            filter="brand", label=f"{brand.title()} — the brand you asked for", points=2.5
        ))
    elif brand and brand in filters.soft_brands:
        factors.append(FitFactor(
            filter="brand", label=f"{brand.title()} — the brand you mentioned", points=1.0
        ))

    if BRAND_RE.search(name):
        factors.append(FitFactor(filter="market", label="a brand we recognise", points=3.0))

    priority = filters.priority or "allround"
    for hint in PRIORITY_HINTS.get(priority, ()):
        if hint in name:
            factors.append(FitFactor(
                filter="priority",
                label=f'matches your {PRIORITY_LABELS[priority]} priority ("{hint.strip()}")',
                points=1.5,
            ))
            break

    for hint in SIZE_HINTS.get(filters.size, ()):
        if hint in name:
            factors.append(FitFactor(
                filter="size",
                label=f'{SIZE_LABELS[filters.size]} — "{hint.strip()}" in the name',
                points=0.5,
            ))
            break

    cap = filters.max_price_usd
    if cap and cap < 9999:
        ratio = phone.price_usd / cap
        if ratio <= 1:
            factors.append(FitFactor(
                filter="budget",
                label=f"{_usd(phone.price_usd)} — {round(ratio * 100)}% of your {_usd(cap)} budget",
                points=2.0 * ratio,
            ))
        else:
            factors.append(FitFactor(
                filter="budget",
                label=f"{_usd(phone.price_usd)} — over your {_usd(cap)} budget",
                points=-2.0,
            ))
    else:
        factors.append(FitFactor(
            filter="budget",
            label=f"{_usd(phone.price_usd)} — no budget set, so price alone ranks it",
            points=min(phone.price_usd, 900) / 450,
        ))

    if filters.os in OS_LABELS:
        factors.append(FitFactor(
            filter="os", label=f"{OS_LABELS[filters.os]}, as you asked", points=0.0
        ))

    return sorted(factors, key=lambda f: -f.points)


def _fit_summary(factors: list[FitFactor]) -> str:
    """The ranking reason in the shopper\'s terms — the filters that earned the points."""
    named = [f for f in factors if f.points > 0 and f.filter != "market"]
    if not named:
        named = [f for f in factors if f.filter != "market"][:1] or factors[:1]
    return " · ".join(f.label for f in named[:3])


def _fit_score(phone: PhoneResult, filters: Filters) -> float:
    return sum(f.points for f in _fit_factors(phone, filters))


def _effective_cap(filters: Filters, relaxed: bool) -> int | None:
    if relaxed:
        return None
    cap = filters.max_price_usd
    return None if cap and cap >= 9999 else cap


def _brand_queries(queries: list[str], filters: Filters) -> list[str]:
    brand = query_words(filters.brands)
    if not brand:
        return queries
    out = []
    for query in queries:
        low = query.lower()
        out.append(query if any(b in low for b in filters.brands) else f"{brand} {query}")
    return out


async def search(state: GraphState) -> GraphState:
    """Build 2-4 retailer queries from the profile and the filters, then run them.

    The budget does not travel as words: `price_band` sends it to the retailer as a
    real price filter, which is why widening relaxes `_effective_cap` rather than
    `max_price_usd` itself.

    Reads `profile`, `filters`, `relaxed`. Writes `queries` and up to 8 deduped `urls`.
    Degrades with `llm` to the deterministic `_query_seed` as the only query, and with
    `search` when every provider failed and no URL survived.
    """
    filters = Filters.model_validate(state.get("filters", {}))
    seed = _query_seed(filters)
    generated = await structured_call(
        SearchQueries,
        QUERIES_SYSTEM,
        json.dumps(
            {"profile": state.get("profile", "")[:600], "filters": filters.model_dump()},
            ensure_ascii=False,
        ),
        run_name="search_queries",
    )
    queries = generated.value.queries if generated.value else [seed]
    queries = _brand_queries(queries, filters)
    if seed not in queries:
        queries.append(seed)
    queries = queries[:4]

    band = price_band(_effective_cap(filters, state.get("relaxed", False)))
    outcomes = await asyncio.gather(
        *(web_search(query, band=band) for query in queries), return_exceptions=True
    )

    urls: list[str] = []
    failed = 0
    for query, outcome in zip(queries, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            log.warning("search failed for %r: %s", query, outcome)
            failed += 1
            continue
        if outcome.degraded:
            failed += 1
        for hit in outcome.hits:
            if hit["url"] not in urls:
                urls.append(hit["url"])

    tags = [DEGRADE_LLM] if generated.unreachable else []
    if failed and not urls:
        tags.append(DEGRADE_SEARCH)
    return {
        "queries": queries,
        "urls": urls[:8],
        "degraded": _degraded(state, *tags),
    }


async def extract(state: GraphState) -> GraphState:
    """Fetch every URL, read `PhoneResult`s off each page, and filter down to candidates.

    The filters stack rather than replace, in order: price cap, junk names, a positive
    phone test, listing-page URLs, OS, then brand. Ranking is `_fit_score`, not price.

    Reads `urls`, `filters`, `relaxed`, `widened`, `degraded`. Writes up to 12 `results`
    — each carrying the `fit_factors` that ranked it and the `fit_summary` line built
    from them — and a `notice` when there are none.

    Two empty-result paths, and the distinction matters to the shopper: an empty set
    widens the budget once, but only when there is a cap to widen and the brand filter
    is not what emptied the list. A total source outage instead keeps the previous
    turn's shortlist and degrades with `fetch`, so a network fault is never reported as
    a budget that was too tight.
    """
    urls = state.get("urls", [])
    pages = await asyncio.gather(*(fetch_page(u) for u in urls), return_exceptions=True)
    for url, page in zip(urls, pages, strict=True):
        if isinstance(page, BaseException):
            log.warning("fetch raised for %s: %s", url, page)
    valid = [p for p in pages if isinstance(p, dict)]
    if urls and len(valid) < len(urls):
        log.warning("skipped %d of %d sources this turn", len(urls) - len(valid), len(urls))

    groups = await asyncio.gather(*(extract_prices(p) for p in valid), return_exceptions=True)
    usable = 0
    for page, group in zip(valid, groups, strict=True):
        if isinstance(group, BaseException):
            log.warning("extract raised for %s: %s", page.get("url"), group)
        else:
            usable += 1

    sources_down = bool(urls) and usable == 0
    search_down = DEGRADE_SEARCH in state.get("degraded", [])

    filters = Filters.model_validate(state.get("filters", {}))
    relaxed = state.get("relaxed", False)
    cap = _effective_cap(filters, relaxed)
    seen: dict[str, PhoneResult] = {}
    for group in groups:
        if isinstance(group, Exception):
            continue
        for phone in group:
            if cap and phone.price_usd > cap:
                continue
            key = phone.name.lower()
            if key not in seen or phone.price_usd < seen[key].price_usd:
                seen[key] = phone

    candidates = [p for p in seen.values() if p.price_usd >= MIN_PHONE_PRICE_USD]
    candidates = [p for p in candidates if not JUNK_RE.search(p.name)]
    candidates = [p for p in candidates if PHONE_RE.search(p.name)]
    candidates = [p for p in candidates if not is_listing_page(p.url)]
    candidates = [p for p in candidates if _os_matches(p.name, filters.os)]
    before_brand = len(candidates)
    candidates = [
        p for p in candidates
        if matches_brands(p.name, filters.brands, filters.exclude_brands)
    ]
    brand_filtered = bool(before_brand) and not candidates
    scored = [(p, _fit_factors(p, filters)) for p in candidates]
    scored.sort(key=lambda pair: (-sum(f.points for f in pair[1]), -pair[0].price_usd))
    results = scored[:12]

    if not results and (sources_down or search_down):
        tag = DEGRADE_FETCH if sources_down else DEGRADE_SEARCH
        return {
            "results": state.get("results", []),
            "notice": OUTAGE_NOTICE,
            "degraded": _degraded(state, tag),
        }

    degraded = _degraded(state)
    if not results and cap and not state.get("widened") and not brand_filtered:
        return {
            "results": [],
            "relaxed": True,
            "widened": True,
            "notice": "No phones matched that budget, so we widened the search.",
            "degraded": degraded,
        }
    if not results:
        return {
            "results": [],
            "notice": _empty_notice(filters, brand_filtered),
            "degraded": degraded,
        }
    return {
        "results": [_with_fit(phone, factors) for phone, factors in results],
        "notice": None,
        "degraded": degraded,
    }


def _with_fit(phone: PhoneResult, factors: list[FitFactor]) -> dict:
    """A result dict carrying the filters that ranked it, not just its rank."""
    return phone.model_copy(
        update={"fit_summary": _fit_summary(factors), "fit_factors": factors}
    ).model_dump()


def _empty_notice(filters: Filters, brand_filtered: bool) -> str:
    if filters.brands:
        names = " or ".join(b.title() for b in filters.brands)
        if brand_filtered:
            return f"Nothing from {names} matched those filters, so nothing is shown."
        return f"We could not find live {names} listings for that combination right now."
    return "We could not find live prices for that combination right now."


async def annotate(state: GraphState) -> GraphState:
    """Shortlist, rank, and write the one-line `why` for each surviving result.

    Last node before `route_after_annotate` decides between `ask` and `END`. Works on
    copies of the result dicts so the checkpointed state is not mutated in place.

    Reads `results`, `profile`, `filters`. Writes `results` trimmed to 9, reordered by
    the model and carrying a `why` line each.
    Degrades with `llm` to the same 9 results in `extract`'s order and no `why` lines.
    """
    results = state.get("results", [])
    if not results:
        return {}

    payload = {
        "profile": state.get("profile", "")[:600],
        "priority": state.get("filters", {}).get("priority"),
        "phones": [r["name"] for r in results],
    }
    batch = await structured_call(
        WhyBatch, ANNOTATE_SYSTEM, json.dumps(payload, ensure_ascii=False), run_name="annotate"
    )
    if batch.value is None:
        if batch.unreachable:
            log.warning("annotate fell back to the unranked shortlist")
        return {
            "results": results[:9],
            "degraded": _degraded(state, *([DEGRADE_LLM] if batch.unreachable else [])),
        }

    order = {line.name.lower(): i for i, line in enumerate(batch.value.lines)}
    why = {line.name.lower(): line.why for line in batch.value.lines}

    kept = [dict(r) for r in results if r["name"].lower() in order]
    if not kept:
        return {"results": results[:9], "degraded": _degraded(state)}

    kept.sort(key=lambda r: order[r["name"].lower()])
    for result in kept:
        result["why"] = why.get(result["name"].lower(), "")
    return {"results": kept[:9], "degraded": _degraded(state)}


def ask(state: GraphState) -> GraphState:
    """Loop tail: pause the graph on `interrupt()` and wait for the client's answer.

    The only sync node, and the only one that suspends the run. `interrupt()` emits the
    pending question, the API returns it as the turn payload, and the shopper's next
    POST resumes exactly here with their answer as the return value. The edge back to
    `plan` closes the loop.

    Reads `question`, `answers`, `question_count`. Appends the answer to `answers` and
    increments `question_count`, which is the counter `max_questions` caps.
    """
    answer = interrupt({"question": state.get("question")})
    return {
        "answers": [*state.get("answers", []), str(answer)],
        "question_count": state.get("question_count", 0) + 1,
    }
