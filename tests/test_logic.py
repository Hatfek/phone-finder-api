import sys

sys.path.insert(0, ".")

import asyncio
import ipaddress
import re

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from selectolax.parser import HTMLParser

from app import cache, server, tools
from app.api_models import (
    THREAD_ID_PATTERN,
    AnswerRequest,
    ResetRequest,
    ReviseRequest,
    StartRequest,
    sanitize_text,
)
from app.brands import detect_brands, matches_brands
from app.config import Settings
from app.fallbacks import SLOT_LABELS, price_options, revision_questions
from app.limits import RateLimited, RateLimiter, ThreadBusy, TooManyTurns, TurnGuard
from app.net import UnsafeURL, assert_safe_url, is_blocked_ip
from app.nodes import (
    DEGRADE_FETCH,
    DEGRADE_LLM,
    DEGRADE_SEARCH,
    JUNK_RE,
    OUTAGE_NOTICE,
    PHONE_RE,
    _apply_brands,
    _brand_queries,
    _degraded,
    _effective_cap,
    _fit_factors,
    _fit_score,
    _fit_summary,
    _merge_answer,
    _next_slot,
    _os_matches,
    _query_seed,
    _validate_options,
    annotate,
    extract,
    plan,
    revision_update,
)
from app.schemas import REVISABLE_SLOTS, Filters, PhoneResult, Question, SlotFilters
from app.serialize import DEGRADED_WARNINGS, to_response
from app.structured import StructuredResult, structured_call
from app.tools import (
    _amazon_url,
    _price_in_text,
    _short_name,
    fetch_page,
    is_listing_page,
    parse_cards,
    price_band,
    retailer_for,
    web_search,
)


def test_retailer_allowlist():
    assert retailer_for("https://www.bestbuy.com/site/x") == "Best Buy"
    assert retailer_for("https://smtp.amazon.com.evil.net/x") is None
    assert retailer_for("https://randomblog.net/best-phones") is None


def test_price_must_appear_on_page():
    text = "Pixel 9a now $499.00 with free shipping"
    assert _price_in_text(499.0, text)
    assert not _price_in_text(498.0, text)


def test_price_options_never_offer_unreachable_bracket():
    assert price_options([450.0, 620.0]) == ["$300-600", "$600-900", "No limit"]
    assert "Under $300" not in price_options([450.0, 620.0])


def test_price_options_fall_back_when_no_results():
    assert price_options([]) == ["Under $300", "$300-600", "$600-900", "No limit"]


def test_validate_options_rewrites_price_question():
    question = Question(
        question="Budget?", options=["Under $300", "No limit"], hint="", slot="max_price_usd"
    )
    tightened = _validate_options(question, [{"price_usd": 700.0}])
    assert "Under $300" not in tightened.options


def test_validate_options_leaves_other_slots_alone():
    question = Question(question="OS?", options=["Android", "iOS"], hint="", slot="os")
    assert _validate_options(question, [{"price_usd": 700.0}]).options == ["Android", "iOS"]


@pytest.mark.parametrize(
    "answer,slot,field,expected",
    [
        ("iOS", "os", "os", "iOS"),
        ("Android", "os", "os", "Android"),
        ("No preference", "os", "os", "any"),
        ("$300-600", "max_price_usd", "max_price_usd", 600),
        ("No limit", "max_price_usd", "max_price_usd", None),
        ("Battery", "priority", "priority", "battery"),
        ("A bit of everything", "priority", "priority", "allround"),
        ("Compact", "size", "size", "compact"),
        ("Doesn't matter", "size", "size", "any"),
    ],
)
def test_merge_answer(answer, slot, field, expected):
    assert getattr(_merge_answer(Filters(), answer, slot), field) == expected


def test_slot_order_skips_filled():
    assert _next_slot(Filters()) == "os"
    assert _next_slot(Filters(os="iOS")) == "max_price_usd"
    full = Filters(os="iOS", max_price_usd=600, priority="camera", size="compact")
    assert _next_slot(full) is None


def test_answered_no_preference_is_a_decision_not_a_gap():
    filters = _merge_answer(Filters(), "No preference", "os")
    assert filters.os == "any"
    assert _next_slot(filters) == "max_price_usd"

    filters = _merge_answer(filters, "No limit", "max_price_usd")
    assert filters.max_price_usd is None
    assert _next_slot(filters) == "priority"

    filters = _merge_answer(filters, "Doesn't matter", "size")
    assert filters.size == "any"


def test_answer_stacks_every_signal_it_carries():
    filters = _merge_answer(Filters(), "Large screen android under $600", "size")
    assert filters.size == "large"
    assert filters.os == "Android"
    assert filters.max_price_usd == 600


def test_answer_never_clears_an_earlier_choice():
    filters = _merge_answer(Filters(), "Android", "os")
    filters = _merge_answer(filters, "Under $300", "max_price_usd")
    filters = _merge_answer(filters, "Camera", "priority")
    filters = _merge_answer(filters, "Compact", "size")
    assert (filters.os, filters.max_price_usd, filters.priority, filters.size) == (
        "Android", 300, "camera", "compact",
    )


def test_budget_answer_does_not_settle_the_slot_that_was_asked():
    filters = _merge_answer(Filters(), "Under $300", "size")
    assert filters.max_price_usd == 300
    assert not filters.is_set("size")


def test_brand_preference_survives_later_answers():
    filters = _apply_brands(Filters(), "I prefer Samsung phones", direct=False)
    assert filters.brands == ["samsung"]
    assert filters.os == "Android"

    filters = _merge_answer(filters, "Under $300", "max_price_usd")
    filters = _merge_answer(filters, "Camera", "priority")
    assert filters.brands == ["samsung"]
    assert filters.max_price_usd == 300


def test_brand_filter_and_budget_stack():
    filters = _apply_brands(Filters(max_price_usd=300), "Samsung only", direct=False)
    assert matches_brands("Samsung Galaxy A16 5G", filters.brands, filters.exclude_brands)
    assert not matches_brands("Motorola Moto G Power", filters.brands, filters.exclude_brands)


def test_brand_dislike_becomes_an_exclusion():
    prefer, exclude, _ = detect_brands("I do not want an iPhone", direct=False)
    assert exclude == ["apple"]
    assert prefer == []
    assert not matches_brands("Apple iPhone 15", prefer, exclude)


def test_passing_brand_mention_does_not_hard_filter():
    prefer, exclude, mentioned = detect_brands("my old pixel finally died", direct=False)
    assert prefer == [] and exclude == []
    assert mentioned == ["google"]
    assert matches_brands("Samsung Galaxy A16", prefer, exclude)


def test_answering_with_a_brand_is_a_preference():
    prefer, _, _ = detect_brands("Samsung", direct=True)
    assert prefer == ["samsung"]


def test_planner_output_cannot_clear_settled_filters():
    settled = Filters(os="Android", max_price_usd=300, priority="camera", size="compact")
    filled = settled.fill_from(SlotFilters())
    assert filled == settled


def test_planner_output_fills_only_empty_slots():
    settled = Filters(os="Android")
    filled = settled.fill_from(SlotFilters(os="iOS", priority="battery"))
    assert filled.os == "Android"
    assert filled.priority == "battery"


def test_widening_never_erases_the_stated_budget():
    filters = Filters(max_price_usd=300)
    assert _effective_cap(filters, relaxed=True) is None
    assert _effective_cap(filters, relaxed=False) == 300
    assert filters.max_price_usd == 300


def test_brand_queries_carry_the_brand():
    filters = Filters(brands=["samsung"])
    assert _brand_queries(["android camera phone"], filters) == ["samsung android camera phone"]
    assert _brand_queries(["samsung galaxy 5g"], filters) == ["samsung galaxy 5g"]


def test_query_seed_describes_the_product_not_the_price():
    seed = _query_seed(Filters(os="Android", max_price_usd=600, priority="camera"))
    assert "camera" in seed and "Android" in seed and "smartphone" in seed
    assert "600" not in seed and "$" not in seed


def test_price_band_brackets_the_budget():
    assert price_band(500) == (250, 500)
    assert price_band(120) == (90, 120)
    assert price_band(None) == (90, None)
    assert price_band(9999) == (90, None)


def test_amazon_url_carries_the_price_filter():
    url = _amazon_url("android+camera+phone", (250, 500))
    assert "p_36%3A25000-50000" in url
    assert _amazon_url("phone", (90, None)).endswith("p_36%3A9000-")


def _phone(name, price):
    return PhoneResult(
        name=name, price_usd=price, retailer="Amazon",
        url="https://www.amazon.com/dp/X", fetched_at="2026-09-05",
    )


def test_fit_score_prefers_known_brand_near_budget():
    filters = Filters(priority="camera", max_price_usd=500)
    known = _fit_score(_phone("Samsung Galaxy A57 5G", 449), filters)
    noname = _fit_score(_phone("MIRO Ultra-Light Smartphone", 449), filters)
    cheap = _fit_score(_phone("Samsung Galaxy A16", 149), filters)
    assert known > noname
    assert known > cheap


def test_fit_score_penalises_over_budget():
    filters = Filters(priority="camera", max_price_usd=500)
    assert _fit_score(_phone("Samsung Galaxy S26 Ultra", 1199), filters) < _fit_score(
        _phone("Samsung Galaxy A57 5G", 480), filters
    )


@pytest.mark.parametrize(
    "title,expected",
    [
        (
            "Samsung Galaxy A37 5G (2026), Unlocked Android Smartphone, 128GB",
            "Samsung Galaxy A37 5G (2026)",
        ),
        (
            "Samsung Galaxy A16 4G LTE (128GB + 4GB) International Model",
            "Samsung Galaxy A16 4G LTE",
        ),
        ("Motorola Moto G Play LTE | Unlocked | 50MP Camera", "Motorola Moto G Play LTE"),
        ("Google Pixel 10a - Unlocked Android Phone", "Google Pixel 10a"),
        ("Nothing Phone (3)", "Nothing Phone (3)"),
    ],
)
def test_short_name_trims_marketing_copy(title, expected):
    assert _short_name(title) == expected


AMAZON_CARD = """
<div data-component-type="s-search-result" data-asin="B0GMKXXTV1">
  <img class="s-image" src="https://m.media-amazon.com/images/I/51j.jpg" \
alt="Samsung Galaxy A37 5G, Unlocked">
  <h2><span>Samsung Galaxy A37 5G (2026), Unlocked Android Smartphone</span></h2>
  <div class="a-price"><span class="a-offscreen">$374.99</span></div>
</div>
<div data-component-type="s-search-result" data-asin="B0FJRQT1ZM">
  <img class="s-image" src="https://m.media-amazon.com/images/I/61I.jpg" \
alt="Sponsored Ad - MIRO Smartphone">
  <h2><span>MIRO Smartphone Ultra-Light</span></h2>
  <div class="a-price"><span class="a-offscreen">$89.99</span></div>
</div>
<div data-component-type="s-search-result" data-asin="">
  <h2><span>No asin, no price</span></h2>
</div>
"""


def test_amazon_cards_yield_product_url_and_image():
    cards = parse_cards(HTMLParser(AMAZON_CARD), "https://www.amazon.com/s?k=phone")
    assert len(cards) == 1
    card = cards[0]
    assert card["name"] == "Samsung Galaxy A37 5G (2026)"
    assert card["price_usd"] == 374.99
    assert card["url"] == "https://www.amazon.com/dp/B0GMKXXTV1"
    assert card["image"].startswith("https://m.media-amazon.com/")


def test_parse_cards_ignores_unknown_retailers():
    assert parse_cards(HTMLParser(AMAZON_CARD), "https://www.walmart.com/search?q=phone") == []


def test_fit_score_does_not_match_brands_inside_words():
    filters = Filters(priority="camera", max_price_usd=500)
    gimbal = _fit_score(_phone("IFOOTAGE Pico Pro Motorized Slider Phone Gimbal", 299), filters)
    phone = _fit_score(_phone("Motorola Moto G Power", 299), filters)
    assert phone > gimbal


@pytest.mark.parametrize(
    "name",
    ["IFOOTAGE Motorized Slider Phone Gimbal", "Anker Power Bank 20000mAh", "Camera Lens Kit"],
)
def test_junk_filter_drops_accessories(name):
    assert JUNK_RE.search(name)


def test_junk_filter_keeps_real_phones():
    for name in ("Motorola Moto G Power", "Motorola Moto G Stylus 5G", "Nothing Phone (3a) Pro"):
        assert not JUNK_RE.search(name)


@pytest.mark.parametrize(
    "name,os,expected",
    [
        ("Apple iPhone 16", "iOS", True),
        ("Nothing Phone (4a) Pro Cell Phone 2026 New", "iOS", False),
        ("Motorola Moto G Play LTE", "iOS", False),
        ("Samsung Galaxy A57 5G", "Android", True),
        ("Apple iPhone SE 3rd Gen", "Android", False),
        ("Nothing Phone (4a) Pro", "any", True),
        ("Apple iPhone 16", "any", True),
    ],
)
def test_os_filter_is_enforced_not_hinted(name, os, expected):
    assert _os_matches(name, os) is expected


def test_query_seed_uses_retailer_words_for_ios():
    seed = _query_seed(Filters(os="iOS"))
    assert "Apple iPhone" in seed
    assert "iOS" not in seed


@pytest.mark.parametrize(
    "name",
    [
        "Cat Camera Collar with Phone App & Tracker Tag Pet",
        "4Pack Security Cameras Wireless Outdoor 2.4G&5G WIFI",
    ],
)
def test_non_phone_products_are_rejected(name):
    assert JUNK_RE.search(name) or not PHONE_RE.search(name)


@pytest.mark.parametrize(
    "name",
    ["Apple iPhone 13 Mini", "Samsung Galaxy A37 5G", "Google Pixel 10a", "Nothing Phone (3a) Pro"],
)
def test_real_phones_survive_the_phone_filter(name):
    assert PHONE_RE.search(name) and not JUNK_RE.search(name)


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.amazon.com/s?k=phone&rh=p_36%3A9000-", True),
        ("https://www.walmart.com/search?q=android+phone", True),
        ("https://www.amazon.com/dp/B0DHJCYNS7", False),
        ("https://www.apple.com/shop/buy-iphone/iphone-16", False),
    ],
)
def test_listing_pages_are_not_product_pages(url, expected):
    assert is_listing_page(url) is expected


def test_rate_limiter_allows_up_to_the_limit_then_blocks():
    limiter = RateLimiter(limit=3, window=60.0)
    for _ in range(3):
        limiter.check("1.2.3.4", now=100.0)
    with pytest.raises(RateLimited) as exc:
        limiter.check("1.2.3.4", now=100.0)
    assert exc.value.retry_after == pytest.approx(60.0)


def test_rate_limiter_is_per_key():
    limiter = RateLimiter(limit=1, window=60.0)
    limiter.check("a", now=0.0)
    limiter.check("b", now=0.0)
    with pytest.raises(RateLimited):
        limiter.check("a", now=0.0)


def test_rate_limiter_window_rolls_off():
    limiter = RateLimiter(limit=1, window=60.0)
    limiter.check("a", now=0.0)
    with pytest.raises(RateLimited):
        limiter.check("a", now=30.0)
    limiter.check("a", now=61.0)


def test_rate_limiter_forgets_idle_keys():
    limiter = RateLimiter(limit=1, window=60.0)
    limiter.check("a", now=0.0)
    limiter.check("b", now=200.0)
    assert "a" not in limiter._hits


def test_turn_guard_rejects_a_second_turn_on_the_same_thread():
    guard = TurnGuard(max_concurrent=4)
    guard.acquire("abc123abc123")
    with pytest.raises(ThreadBusy):
        guard.acquire("abc123abc123")
    guard.release("abc123abc123")
    guard.acquire("abc123abc123")


def test_turn_guard_caps_total_concurrency():
    guard = TurnGuard(max_concurrent=2)
    guard.acquire("a")
    guard.acquire("b")
    with pytest.raises(TooManyTurns):
        guard.acquire("c")
    guard.release("a")
    guard.acquire("c")
    assert guard.active == 2


@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_blank_bodies_are_rejected(bad):
    with pytest.raises(ValidationError):
        StartRequest(profile=bad)
    with pytest.raises(ValidationError):
        AnswerRequest(answer=bad)


def test_bodies_are_stripped_and_length_capped():
    assert StartRequest(profile="  I want a camera phone  ").profile == "I want a camera phone"
    assert AnswerRequest(answer=" Under $300 ").answer == "Under $300"
    with pytest.raises(ValidationError):
        StartRequest(profile="x" * 4001)
    with pytest.raises(ValidationError):
        AnswerRequest(answer="x" * 501)


def test_reset_profile_is_optional_but_never_blank():
    assert ResetRequest().profile is None
    assert ResetRequest(profile=" again ").profile == "again"
    with pytest.raises(ValidationError):
        ResetRequest(profile="   ")


@pytest.mark.parametrize(
    "thread_id,ok",
    [("0123456789ab", True), ("0123456789AB", False), ("0123456789", False), ("../etc", False)],
)
def test_thread_id_pattern(thread_id, ok):
    assert bool(re.match(THREAD_ID_PATTERN, thread_id)) is ok


def test_cors_origins_parse_as_a_list():
    assert Settings(cors_origins="http://a.test, http://b.test ,").cors_origin_list == [
        "http://a.test",
        "http://b.test",
    ]


def test_debug_ui_is_off_unless_the_environment_turns_it_on(monkeypatch):
    monkeypatch.delenv("DEBUG_UI", raising=False)
    assert Settings(_env_file=None).debug_ui is False
    monkeypatch.setenv("DEBUG_UI", "true")
    assert Settings(_env_file=None).debug_ui is True


def test_secrets_default_empty_and_read_from_env(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.internal:11434")
    s = Settings()
    assert s.tavily_api_key == "tvly-test"
    assert s.langsmith_api_key == "ls-test"
    assert s.ollama_base_url == "http://ollama.internal:11434"


class _FakeLLM:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def with_structured_output(self, *args, **kwargs):
        return self

    def with_config(self, *args, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        raise self.error


@pytest.mark.asyncio
async def test_unreachable_model_is_reported_without_burning_retries(monkeypatch):
    llm = _FakeLLM(httpx.ConnectError("all connection attempts failed"))
    monkeypatch.setattr("app.structured.get_llm", lambda *a, **k: llm)

    result = await structured_call(SlotFilters, "sys", "user", run_name="t")

    assert result.value is None
    assert result.unreachable
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_a_bad_answer_still_retries_and_is_not_called_unreachable(monkeypatch):
    llm = _FakeLLM(ValueError("Invalid json output:"))
    monkeypatch.setattr("app.structured.get_llm", lambda *a, **k: llm)

    result = await structured_call(SlotFilters, "sys", "user", run_name="t")

    assert result.value is None
    assert not result.unreachable
    assert llm.calls == 3


@pytest.mark.asyncio
async def test_a_slow_model_times_out_rather_than_holding_the_turn(monkeypatch):
    class _SlowLLM(_FakeLLM):
        async def ainvoke(self, messages):
            self.calls += 1
            await asyncio.sleep(5)

    llm = _SlowLLM(None)
    monkeypatch.setattr("app.structured.get_llm", lambda *a, **k: llm)
    monkeypatch.setattr("app.structured.get_settings", lambda: Settings(llm_timeout=0.05))

    result = await structured_call(SlotFilters, "sys", "user", run_name="t")

    assert result.unreachable


@pytest.mark.asyncio
async def test_a_dead_search_provider_is_skipped_and_never_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")

    async def dead(*args, **kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(tools, "_tavily", dead)
    monkeypatch.setattr(tools, "_retailer_search", dead)

    outcome = await web_search("android camera phone")

    assert outcome.hits == []
    assert outcome.failed == ("tavily", "retailer")
    assert cache.get("search2", "android camera phone|90-") is None


@pytest.mark.asyncio
async def test_tavily_failure_falls_through_to_the_retailer_search(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")

    async def dead(*args, **kwargs):
        raise httpx.ReadTimeout("too slow")

    async def retailer(*args, **kwargs):
        return [{"title": "t", "url": "https://www.amazon.com/dp/B0TEST00001", "snippet": ""}]

    monkeypatch.setattr(tools, "_tavily", dead)
    monkeypatch.setattr(tools, "_retailer_search", retailer)

    outcome = await web_search("android camera phone")

    assert len(outcome.hits) == 1
    assert outcome.failed == ("tavily",)


@pytest.mark.asyncio
async def test_a_page_that_will_not_answer_is_skipped_not_raised(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")

    class _DeadClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            raise httpx.ReadTimeout("too slow")

    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda **kwargs: _DeadClient())

    assert await fetch_page("https://www.amazon.com/s?k=phone") is None


PREVIOUS_RESULTS = [
    {
        "name": "Google Pixel 10a",
        "price_usd": 499.0,
        "retailer": "Amazon",
        "url": "https://www.amazon.com/dp/B0TEST00001",
        "image": None,
        "fetched_at": "2026-09-07",
        "why": "Strong camera for the money",
    }
]


@pytest.mark.asyncio
async def test_a_total_fetch_outage_keeps_the_shortlist_and_blames_no_filter(monkeypatch):
    async def dead_fetch(url):
        return None

    monkeypatch.setattr("app.nodes.fetch_page", dead_fetch)

    out = await extract(
        {
            "urls": ["https://www.amazon.com/s?k=phone"],
            "filters": {"max_price_usd": 500},
            "results": PREVIOUS_RESULTS,
            "degraded": [],
        }
    )

    assert out["results"] == PREVIOUS_RESULTS
    assert out["notice"] == OUTAGE_NOTICE
    assert out["degraded"] == [DEGRADE_FETCH]
    assert "widened" not in out


@pytest.mark.asyncio
async def test_a_search_outage_is_not_reported_as_an_empty_budget(monkeypatch):
    out = await extract(
        {
            "urls": [],
            "filters": {"max_price_usd": 500},
            "results": [],
            "degraded": [DEGRADE_SEARCH],
        }
    )

    assert out["notice"] == OUTAGE_NOTICE
    assert out["degraded"] == [DEGRADE_SEARCH]
    assert "widened" not in out


@pytest.mark.asyncio
async def test_a_genuinely_empty_result_still_widens_the_budget(monkeypatch):
    async def ok_fetch(url):
        return {"url": url, "text": "", "image": None, "cards": []}

    async def nothing(page):
        return []

    monkeypatch.setattr("app.nodes.fetch_page", ok_fetch)
    monkeypatch.setattr("app.nodes.extract_prices", nothing)

    out = await extract(
        {
            "urls": ["https://www.amazon.com/s?k=phone"],
            "filters": {"max_price_usd": 500},
            "results": [],
            "degraded": [],
        }
    )

    assert out["widened"] is True
    assert out["relaxed"] is True
    assert out["notice"] == "No phones matched that budget, so we widened the search."


@pytest.mark.asyncio
async def test_annotate_keeps_the_shortlist_when_the_model_is_unreachable(monkeypatch):
    async def unreachable(*args, **kwargs):
        return StructuredResult(unreachable=True)

    monkeypatch.setattr("app.nodes.structured_call", unreachable)
    results = [{"name": f"Phone {i}", "price_usd": 100.0 + i} for i in range(12)]

    out = await annotate({"results": results, "degraded": [], "filters": {}})

    assert len(out["results"]) == 9
    assert out["degraded"] == [DEGRADE_LLM]


def test_degradation_tags_accumulate_without_repeating():
    assert _degraded({}, DEGRADE_LLM) == ["llm"]
    assert _degraded({"degraded": ["llm"]}, DEGRADE_LLM) == ["llm"]
    assert _degraded({"degraded": ["llm"]}, DEGRADE_FETCH) == ["llm", "fetch"]


def test_every_degradation_reaches_the_client_as_a_sentence():
    response = to_response(
        "abc123abc123",
        {"question": None, "results": [], "filters": {}, "degraded": ["llm", "not-a-tag"]},
        False,
        ("turn",),
    )

    assert response.degraded == ["llm", "turn"]
    assert response.warnings == [DEGRADED_WARNINGS["llm"], DEGRADED_WARNINGS["turn"]]


QUESTION = {
    "question": "Which system do you get on with?",
    "options": ["Android", "iOS"],
    "hint": "",
    "slot": "os",
}


class _FakeSnapshot:
    def __init__(self, values, pending):
        self.values = values
        self.next = pending


class _FakeGraph:
    checkpointer = None

    def __init__(self, pending, *, error=None, pending_after=None, filters=None):
        self.pending = pending
        self.error = error
        self.pending_after = pending_after
        self.filters = filters if filters is not None else {}
        self.payloads = []
        self.updates = []

    async def ainvoke(self, payload, config):
        self.payloads.append(payload)
        if self.error:
            raise self.error
        if self.pending_after is not None:
            self.pending = self.pending_after

    async def aupdate_state(self, config, values, as_node=None):
        self.updates.append((values, as_node))

    async def aget_state(self, config):
        values = {
            "question": QUESTION,
            "question_count": 0,
            "results": [],
            "filters": self.filters,
            "answers": [],
            "profile": "I want a camera phone",
            "degraded": [],
        }
        return _FakeSnapshot(values, self.pending)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "_limiter", RateLimiter(1000, 60.0))
    monkeypatch.setattr(server, "_global_limiter", RateLimiter(1000, 60.0))
    return TestClient(server.app)


@pytest.mark.parametrize("debug_ui", [False, True])
def test_health_publishes_the_debug_ui_flag_to_the_client(client, monkeypatch, debug_ui):
    async def get_graph():
        return _FakeGraph(("ask",))

    monkeypatch.setattr(server, "get_graph", get_graph)
    monkeypatch.setattr(server, "settings", Settings(_env_file=None, debug_ui=debug_ui))
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["debug_ui"] is debug_ui


def test_a_node_exception_is_a_retryable_turn_failure_not_a_500(client, monkeypatch):
    graph = _FakeGraph(("ask",), error=RuntimeError("extract blew up"))

    async def get_graph():
        return graph

    monkeypatch.setattr(server, "get_graph", get_graph)
    response = client.post("/threads", json={"profile": "I want a camera phone"})

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] == ["turn"]
    assert body["warnings"] == [DEGRADED_WARNINGS["turn"]]
    assert body["ask_question"]["slot"] == "os"


def test_answering_a_stalled_thread_finishes_the_run_instead_of_eating_the_answer(
    client, monkeypatch
):
    graph = _FakeGraph(("extract",), pending_after=("ask",))

    async def get_graph():
        return graph

    monkeypatch.setattr(server, "get_graph", get_graph)
    response = client.post("/threads/0123456789ab/answer", json={"answer": "Android"})

    assert response.status_code == 200
    assert graph.payloads == [None]
    assert response.json()["degraded"] == ["resumed"]


def test_a_normal_answer_still_resumes_the_interrupt(client, monkeypatch):
    graph = _FakeGraph(("ask",))

    async def get_graph():
        return graph

    monkeypatch.setattr(server, "get_graph", get_graph)
    response = client.post("/threads/0123456789ab/answer", json={"answer": "Android"})

    assert response.status_code == 200
    assert graph.payloads[0].resume == "Android"
    assert response.json()["degraded"] == []


# --- the `why` line names the filter that drove the rank ---------------------

FIT_FILTERS = Filters(
    priority="camera", max_price_usd=500, brands=["samsung"], size="large", os="Android"
)


def test_fit_score_is_exactly_the_sum_of_its_named_factors():
    phone = _phone("Samsung Galaxy S24 Ultra 5G", 449.0)
    factors = _fit_factors(phone, FIT_FILTERS)

    assert _fit_score(phone, FIT_FILTERS) == pytest.approx(sum(f.points for f in factors))


def test_every_factor_names_the_filter_it_came_from():
    factors = _fit_factors(_phone("Samsung Galaxy S24 Ultra 5G", 449.0), FIT_FILTERS)
    found = {f.filter for f in factors}

    assert {"brand", "budget", "priority", "size", "os"} <= found


def test_the_why_line_names_budget_and_brand_rather_than_a_generic_sentence():
    factors = _fit_factors(_phone("Samsung Galaxy S24 Ultra 5G", 449.0), FIT_FILTERS)
    summary = _fit_summary(factors)

    assert "Samsung" in summary
    assert "$500 budget" in summary
    assert "$449" in summary
    assert "camera" in summary


def test_two_phones_on_the_same_filters_get_different_why_lines():
    cheap = _fit_summary(_fit_factors(_phone("Samsung Galaxy A16 5G", 149.0), FIT_FILTERS))
    dear = _fit_summary(_fit_factors(_phone("Samsung Galaxy S24 Ultra 5G", 449.0), FIT_FILTERS))

    assert cheap != dear


def test_an_over_budget_phone_says_so_and_scores_against_itself():
    factors = _fit_factors(_phone("Samsung Galaxy S26 Ultra", 1199.0), FIT_FILTERS)
    budget = next(f for f in factors if f.filter == "budget")

    assert budget.points < 0
    assert "over your $500 budget" in budget.label


def test_the_summary_skips_the_bonus_no_filter_earned():
    factors = _fit_factors(_phone("Samsung Galaxy S24 Ultra 5G", 449.0), FIT_FILTERS)

    assert any(f.filter == "market" for f in factors)
    assert "a brand we recognise" not in _fit_summary(factors)


def test_a_phone_that_matches_nothing_still_gets_a_specific_line():
    plain = Filters()
    summary = _fit_summary(_fit_factors(_phone("Nokia G42 5G", 199.0), plain))

    assert "$199" in summary


@pytest.mark.asyncio
async def test_extract_hands_the_client_the_filters_that_ranked_each_phone(monkeypatch):
    async def ok_fetch(url):
        return {"url": url, "text": "", "image": None, "cards": []}

    async def one_phone(page):
        return [_phone("Samsung Galaxy A57 5G", 449.0)]

    monkeypatch.setattr("app.nodes.fetch_page", ok_fetch)
    monkeypatch.setattr("app.nodes.extract_prices", one_phone)

    out = await extract(
        {
            "urls": ["https://www.amazon.com/dp/B0TEST00001"],
            "filters": FIT_FILTERS.model_dump(),
            "results": [],
            "degraded": [],
        }
    )

    result = out["results"][0]
    assert "$500 budget" in result["fit_summary"]
    assert {f["filter"] for f in result["fit_factors"]} >= {"brand", "budget"}


# --- changing one earlier answer keeps every later one -----------------------

SETTLED = Filters(
    os="Android",
    max_price_usd=300,
    priority="camera",
    size="large",
    brands=["samsung"],
    resolved=["os", "max_price_usd", "priority", "size"],
)
SETTLED_STATE = {
    "filters": SETTLED.model_dump(),
    "answers": ["Android", "Under $300", "Camera", "Large screen"],
    "question_count": 4,
}


def test_clearing_one_slot_leaves_every_other_answer_standing():
    cleared = SETTLED.clear_slot("max_price_usd")

    assert cleared.max_price_usd is None
    assert "max_price_usd" not in cleared.resolved
    assert (cleared.os, cleared.priority, cleared.size) == ("Android", "camera", "large")
    assert cleared.brands == ["samsung"]


def test_clearing_a_brand_drops_the_brand_filter_and_nothing_else():
    cleared = SETTLED.clear_slot("brands")

    assert cleared.brands == []
    assert cleared.max_price_usd == 300
    assert cleared.os == "Android"


@pytest.mark.parametrize(
    "slot,expected",
    [("os", True), ("max_price_usd", True), ("brands", True), ("exclude_brands", False)],
)
def test_only_a_filter_the_shopper_actually_set_can_be_revised(slot, expected):
    assert SETTLED.is_revisable(slot) is expected


def test_a_slot_answered_no_preference_is_still_revisable():
    shrugged = Filters(resolved=["size"])

    assert shrugged.size == "any"
    assert shrugged.is_revisable("size")


def test_a_revision_moves_the_merge_mark_past_the_answer_it_replaces():
    update = revision_update(SETTLED_STATE, "max_price_usd", "Under $600")

    assert update["merged_answers"] == 4
    assert update["answers"][-1] == "Under $600"
    assert update["question"]["slot"] == "max_price_usd"
    assert update["filters"]["max_price_usd"] is None
    assert update["filters"]["priority"] == "camera"


def test_a_revision_with_no_answer_hands_one_question_back():
    update = revision_update(SETTLED_STATE, "priority", None)

    assert update["question"] is None
    assert update["question_count"] == 3
    assert "answers" not in update
    assert update["merged_answers"] == 4
    assert update["filters"]["priority"] is None
    assert update["filters"]["max_price_usd"] == 300


def test_dropping_a_brand_does_not_buy_back_a_question():
    update = revision_update(SETTLED_STATE, "brands", None)

    assert "question_count" not in update
    assert update["filters"]["brands"] == []


def test_a_revision_clears_a_stale_widen_so_the_new_budget_is_respected():
    stale = {**SETTLED_STATE, "widened": True, "relaxed": True}
    update = revision_update(stale, "max_price_usd", "$600")

    assert update["widened"] is False
    assert update["relaxed"] is False


@pytest.mark.asyncio
async def test_plan_does_not_refold_an_answer_it_has_already_merged(monkeypatch):
    async def unreachable(*args, **kwargs):
        return StructuredResult(unreachable=True)

    monkeypatch.setattr("app.nodes.structured_call", unreachable)
    revised = revision_update(SETTLED_STATE, "max_price_usd", None)

    out = await plan({**SETTLED_STATE, **revised, "profile": "camera phone"})

    assert out["filters"]["max_price_usd"] is None
    assert out["question"]["slot"] == "max_price_usd"


@pytest.mark.asyncio
async def test_plan_folds_the_answer_a_revision_supplied(monkeypatch):
    async def unreachable(*args, **kwargs):
        return StructuredResult(unreachable=True)

    monkeypatch.setattr("app.nodes.structured_call", unreachable)
    revised = revision_update(SETTLED_STATE, "max_price_usd", "Under $600")

    out = await plan({**SETTLED_STATE, **revised, "profile": "camera phone"})

    assert out["filters"]["max_price_usd"] == 600
    assert out["filters"]["priority"] == "camera"
    assert out["filters"]["brands"] == ["samsung"]
    assert out["merged_answers"] == 5


@pytest.mark.parametrize("bad", ["", "profile", "results", "soft_brands"])
def test_revise_refuses_a_slot_that_is_not_a_filter(bad):
    with pytest.raises(ValidationError):
        ReviseRequest(slot=bad)


def test_a_blank_revision_answer_reads_as_clear_this_filter():
    assert ReviseRequest(slot="os", answer="   ").answer is None
    assert ReviseRequest(slot="os").answer is None


def test_every_revisable_slot_has_a_question_and_a_label():
    questions = revision_questions()

    for slot in REVISABLE_SLOTS:
        assert slot in questions
        assert slot in SLOT_LABELS
        assert len(questions[slot].options) >= 2


def test_the_exclude_brand_options_carry_the_negation_cue_that_makes_them_exclusions():
    for option in revision_questions()["exclude_brands"].options:
        prefer, exclude, _ = detect_brands(option, direct=True)
        assert not prefer
        if exclude:
            assert exclude != []


REVISABLE_STATE = {"os": "Android", "max_price_usd": 600, "resolved": ["os", "max_price_usd"]}


def test_revising_a_filter_resumes_the_graph_at_plan_rather_than_starting_over(client, monkeypatch):
    graph = _FakeGraph(("ask",), filters=REVISABLE_STATE)

    async def get_graph():
        return graph

    monkeypatch.setattr(server, "get_graph", get_graph)
    response = client.post(
        "/threads/0123456789ab/revise", json={"slot": "max_price_usd", "answer": "Under $300"}
    )

    assert response.status_code == 200
    values, as_node = graph.updates[0]
    assert as_node == "ask"
    assert values["filters"]["max_price_usd"] is None
    assert values["filters"]["os"] == "Android"
    assert values["answers"][-1] == "Under $300"
    assert graph.payloads == [None]


def test_revising_a_filter_that_is_not_set_is_refused_rather_than_silently_run(
    client, monkeypatch
):
    graph = _FakeGraph(("ask",), filters=REVISABLE_STATE)

    async def get_graph():
        return graph

    monkeypatch.setattr(server, "get_graph", get_graph)
    response = client.post("/threads/0123456789ab/revise", json={"slot": "brands"})

    assert response.status_code == 409
    assert graph.updates == []
    assert graph.payloads == []


def test_an_unknown_revision_slot_never_reaches_the_graph(client, monkeypatch):
    graph = _FakeGraph(("ask",), filters=REVISABLE_STATE)

    async def get_graph():
        return graph

    monkeypatch.setattr(server, "get_graph", get_graph)
    response = client.post("/threads/0123456789ab/revise", json={"slot": "profile"})

    assert response.status_code == 422
    assert graph.updates == []


def test_the_slot_catalogue_covers_every_revisable_filter(client):
    response = client.get("/slots")

    assert response.status_code == 200
    slots = response.json()["slots"]
    assert [s["slot"] for s in slots] == list(REVISABLE_SLOTS)
    assert all(s["label"] and s["question"] and len(s["options"]) >= 2 for s in slots)


def test_every_thread_route_is_reachable_without_credentials(client, monkeypatch):
    """The server carries no authentication: nothing is asked of the caller."""

    async def get_graph():
        return _FakeGraph(("ask",))

    monkeypatch.setattr(server, "get_graph", get_graph)
    response = client.post("/threads", json={"profile": "I want a camera phone"})

    assert response.status_code == 200
    assert response.json()["ask_question"]["slot"] == "os"


def test_health_reports_the_backend_without_a_key(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}


def test_the_global_ceiling_is_separate_from_the_turn_limit(monkeypatch):
    monkeypatch.setattr(server, "_limiter", RateLimiter(1000, 60.0))
    monkeypatch.setattr(server, "_global_limiter", RateLimiter(2, 60.0))
    limited = TestClient(server.app)

    assert limited.get("/health").status_code == 200
    assert limited.get("/health").status_code == 200
    blocked = limited.get("/health")

    assert blocked.status_code == 429
    assert blocked.headers["Retry-After"]


def test_a_body_over_the_size_cap_is_refused_before_it_is_parsed(client):
    response = client.post(
        "/threads",
        content=b"{}",
        headers={"content-type": "application/json", "content-length": str(11 * 1024 * 1024)},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "request too large"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost:8000/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://[::1]/",
    ],
)
def test_a_private_or_loopback_target_is_refused(url):
    with pytest.raises(UnsafeURL):
        assert_safe_url(url, frozenset({"amazon.com", "localhost", "127.0.0.1"}))


def test_a_host_off_the_allow_list_is_refused():
    with pytest.raises(UnsafeURL) as exc:
        assert_safe_url("https://evil.example.com/x", frozenset({"amazon.com"}))

    assert "not in ALLOWED_DOMAINS" in str(exc.value)


@pytest.mark.parametrize("scheme", ["file:///etc/passwd", "gopher://x/", "ftp://amazon.com/"])
def test_a_non_http_scheme_is_refused(scheme):
    with pytest.raises(UnsafeURL) as exc:
        assert_safe_url(scheme, frozenset({"amazon.com"}))

    assert "not http(s)" in str(exc.value)


def test_a_public_allow_listed_host_passes():
    assert_safe_url("https://www.amazon.com/dp/B0CMDRCZBJ", frozenset({"amazon.com"}))


def test_blocked_ip_covers_every_range_the_spec_named():
    for addr in ("127.0.0.1", "0.0.0.0", "::1", "169.254.1.1", "10.1.2.3", "172.16.5.5",
                 "192.168.0.1"):
        assert is_blocked_ip(ipaddress.ip_address(addr)), addr
    assert not is_blocked_ip(ipaddress.ip_address("93.184.216.34"))


def test_sanitize_strips_control_characters_and_folds_whitespace():
    assert sanitize_text("I want\x00 a\u200b  camera\tphone\n") == "I want a camera phone"
    assert sanitize_text("  ‍ ") == ""


def test_a_profile_of_only_control_characters_is_rejected_as_blank():
    with pytest.raises(ValidationError):
        StartRequest(profile="\x00\x07\u200b")
