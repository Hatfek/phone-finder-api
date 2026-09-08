import asyncio
import logging
import re
from collections.abc import Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import quote_plus, urlparse

import httpx
from selectolax.parser import HTMLParser

from app import cache
from app.config import get_settings
from app.net import UnsafeURL, host_allowed, safe_get
from app.schemas import ExtractionBatch, PhoneResult
from app.structured import structured_call

log = logging.getLogger(__name__)

# Display labels only. The live allow-list is ALLOWED_DOMAINS in the environment
# (Settings.allowed_domain_set); env wins when the two disagree.
ALLOWED_HOSTS = {
    "amazon.com": "Amazon",
    "bestbuy.com": "Best Buy",
    "bhphotovideo.com": "B&H",
    "gsmarena.com": "GSMArena",
    "store.google.com": "Google Store",
    "apple.com": "Apple",
    "samsung.com": "Samsung",
    "walmart.com": "Walmart",
}

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
PRICE_RE = re.compile(r"\$\s?([0-9]{2,4}(?:,[0-9]{3})?(?:\.[0-9]{2})?)")
NAME_CUT_RE = re.compile(r"\s*[,|\[]|\s+[-\u2013\u2014]\s+")
PAREN_RE = re.compile(r"\(([^)]*)\)?")
ASIN_RE = re.compile(r"^[A-Z0-9]{10}$")
LISTING_RE = re.compile(r"/(s|search|sch)(/|\?|$)|[?&](k|q)=", re.IGNORECASE)

EXTRACT_SYSTEM = """You read retail page text and list the smartphones on sale with \
their US price in USD.

Rules:
- Only list a phone if its price literally appears in the text.
- Never estimate, convert, or guess a price. Omit the phone instead.
- price_usd is a plain number without currency symbols or commas.
- List smartphones only. Skip accessories, cases, chargers, screen protectors,
  smart glasses, watches, tablets, and monthly instalment or trade-in figures.
- Keep the name short: brand and model only, no marketing copy or spec lists.
Return JSON only."""


def _short_name(title: str) -> str:
    head = NAME_CUT_RE.split(title.strip(), 1)[0]

    paren = PAREN_RE.search(head)
    if paren and len(paren.group(1)) > 4:
        head = head[: paren.start()]

    head = re.sub(r"\s+", " ", head).strip(" -\u2013\u2014|,")
    if len(head) < 6:
        head = re.sub(r"\s+", " ", title.strip())
    return head[:60].strip()


def _card_price(node) -> float | None:
    price_node = node.css_first(".a-price .a-offscreen") or node.css_first(".a-price")
    if price_node is None:
        return None
    match = PRICE_RE.search(price_node.text())
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _amazon_cards(tree) -> list[dict]:
    cards = []
    for node in tree.css('div[data-component-type="s-search-result"]'):
        asin = (node.attributes.get("data-asin") or "").strip()
        if not ASIN_RE.match(asin):
            continue

        image_node = node.css_first("img.s-image")
        alt = (image_node.attributes.get("alt") or "") if image_node else ""
        if alt.lower().startswith("sponsored"):
            continue

        title_node = node.css_first("h2 span")
        title = title_node.text().strip() if title_node else alt.strip()
        price = _card_price(node)
        if not title or price is None:
            continue

        cards.append(
            {
                "name": _short_name(title),
                "price_usd": price,
                "url": f"https://www.amazon.com/dp/{asin}",
                "image": image_node.attributes.get("src") if image_node else None,
            }
        )
    return cards


def is_listing_page(url: str) -> bool:
    """True for a search/results page, whose URL belongs to no single product."""
    return bool(LISTING_RE.search(url))


def parse_cards(tree, url: str) -> list[dict]:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    if host.endswith("amazon.com"):
        return _amazon_cards(tree)
    return []


def allowed_domains() -> frozenset[str]:
    return get_settings().allowed_domain_set


def retailer_for(url: str) -> str | None:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    if not host_allowed(host, allowed_domains()):
        return None
    for allowed, label in ALLOWED_HOSTS.items():
        if host == allowed or host.endswith("." + allowed):
            return label
    return host


@dataclass(frozen=True)
class SearchOutcome:
    """Hits plus the providers that failed producing them."""

    hits: list[dict] = field(default_factory=list)
    failed: tuple[str, ...] = ()

    @property
    def degraded(self) -> bool:
        return bool(self.failed)


async def _tavily(query: str, k: int) -> list[dict]:
    key = get_settings().tavily_api_key
    if not key:
        return []
    async with httpx.AsyncClient(timeout=get_settings().fetch_timeout) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": key,
                "query": query,
                "max_results": k,
                "include_domains": sorted(allowed_domains()),
                "include_images": False,
            },
        )
        resp.raise_for_status()
        body = resp.json()
    results = body.get("results", []) if isinstance(body, dict) else []
    return [
        {"title": r.get("title", ""), "url": r["url"], "snippet": r.get("content", "")}
        for r in results
        if isinstance(r, dict) and r.get("url")
    ]


async def _try_provider(name: str, call: Awaitable[list[dict]], failed: list[str]) -> list[dict]:
    budget = get_settings().fetch_timeout + 5
    try:
        return await asyncio.wait_for(call, timeout=budget)
    except asyncio.TimeoutError:
        log.warning("%s search timed out after %ss", name, budget)
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        log.warning("%s search failed: %s", name, exc)
    except Exception:
        log.exception("%s search raised unexpectedly", name)
    failed.append(name)
    return []


MIN_LISTING_USD = 90


def price_band(max_price_usd: int | None) -> tuple[int, int | None]:
    """Floor and ceiling in dollars for a retailer price filter."""
    if not max_price_usd or max_price_usd >= 9999:
        return MIN_LISTING_USD, None
    return max(MIN_LISTING_USD, int(max_price_usd * 0.5)), max_price_usd


def _amazon_url(terms: str, band: tuple[int, int | None]) -> str:
    low, high = band
    rh = f"p_36:{low * 100}-{high * 100}" if high else f"p_36:{low * 100}-"
    return f"https://www.amazon.com/s?k={terms}&rh={quote_plus(rh)}"


async def _retailer_search(query: str, k: int, band: tuple[int, int | None]) -> list[dict]:
    cleaned = re.sub(r"\b(best|buy|price|review|top|under|below|\$\d+|\d{3,4})\b", "", query)
    terms = quote_plus(re.sub(r"\s+", " ", cleaned).strip())
    urls = [
        _amazon_url(terms, band),
        f"https://www.walmart.com/search?q={terms}",
    ]
    return [{"title": query, "url": u, "snippet": ""} for u in urls[:k]]


async def web_search(
    query: str, k: int = 8, band: tuple[int, int | None] = (MIN_LISTING_USD, None)
) -> SearchOutcome:
    cache_key = f"{query}|{band[0]}-{band[1] or ''}"
    cached = cache.get("search2", cache_key)
    if cached is not None:
        return SearchOutcome(hits=cached)

    failed: list[str] = []
    hits = await _try_provider("tavily", _tavily(query, k), failed)
    if not hits:
        hits = await _try_provider("retailer", _retailer_search(query, k, band), failed)

    hits = [h for h in hits if isinstance(h, dict) and h.get("url") and retailer_for(h["url"])]
    if hits or not failed:
        cache.put("search2", cache_key, hits)
    return SearchOutcome(hits=hits, failed=tuple(failed))


async def fetch_page(url: str) -> dict | None:
    cached = cache.get("page", url)
    if cached is not None and "cards" in cached:
        return cached

    settings = get_settings()
    budget = settings.fetch_timeout
    try:
        resp = await safe_get(
            url,
            allowed=allowed_domains(),
            headers={"User-Agent": UA},
            timeout=budget,
            disable_redirects=settings.disable_redirects,
        )
        resp.raise_for_status()
        html = resp.text
    except UnsafeURL as exc:
        log.warning("fetch refused %s: %s", url, exc)
        return None
    except asyncio.TimeoutError:
        log.warning("fetch timed out %s after %ss", url, budget + 5)
        return None
    except (httpx.HTTPError, UnicodeDecodeError, OSError) as exc:
        log.warning("fetch failed %s: %s", url, exc)
        return None
    except Exception:
        log.exception("fetch raised unexpectedly for %s", url)
        return None

    try:
        tree = HTMLParser(html)
        for tag in tree.css("script, style, noscript, svg"):
            tag.decompose()
        text = re.sub(r"\s+", " ", tree.body.text() if tree.body else "")[:12000]

        image = None
        og = tree.css_first('meta[property="og:image"]')
        if og:
            image = og.attributes.get("content")

        payload = {"text": text, "image": image, "url": url, "cards": parse_cards(tree, url)}
    except Exception:
        log.exception("parse failed for %s", url)
        return None

    cache.put("page", url, payload)
    return payload


def _price_in_text(price: float, text: str) -> bool:
    for raw in PRICE_RE.findall(text):
        if abs(float(raw.replace(",", "")) - price) < 0.01:
            return True
    return False


def _cards_to_results(cards: list[dict], retailer: str, today: str) -> list[PhoneResult]:
    out: list[PhoneResult] = []
    for card in cards:
        try:
            out.append(
                PhoneResult(
                    name=card["name"],
                    price_usd=card["price_usd"],
                    retailer=retailer,
                    url=card["url"],
                    image=card.get("image"),
                    fetched_at=today,
                )
            )
        except ValueError:
            continue
    return out


async def extract_prices(page: dict) -> list[PhoneResult]:
    retailer = retailer_for(page["url"])
    if not retailer:
        return []

    cached = cache.get("extract2", page["url"])
    if cached is not None:
        return [PhoneResult.model_validate(p) for p in cached]

    today = datetime.now(timezone.utc).date().isoformat()

    cards = page.get("cards") or []
    if cards:
        out = _cards_to_results(cards, retailer, today)
        log.info("%s: %d cards parsed from DOM", page["url"], len(out))
        cache.put("extract2", page["url"], [p.model_dump() for p in out])
        return out

    if is_listing_page(page["url"]):
        log.info("%s: listing page with no parseable cards, skipping", page["url"])
        cache.put("extract2", page["url"], [])
        return []

    batch = await structured_call(
        ExtractionBatch,
        EXTRACT_SYSTEM,
        f"Page: {page['url']}\n\nText:\n{page['text'][:8000]}",
        run_name="extract_prices",
    )
    if batch.value is None:
        return []

    out: list[PhoneResult] = []
    for phone in batch.value.phones:
        if not phone.price_usd or not phone.name.strip():
            continue
        if not _price_in_text(phone.price_usd, page["text"]):
            log.info("dropped %s: price %s not on page", phone.name, phone.price_usd)
            continue
        try:
            out.append(
                PhoneResult(
                    name=phone.name.strip(),
                    price_usd=phone.price_usd,
                    retailer=retailer,
                    url=page["url"],
                    image=page.get("image"),
                    fetched_at=today,
                )
            )
        except ValueError:
            continue

    cache.put("extract2", page["url"], [p.model_dump() for p in out])
    return out


async def gather_results(queries: list[str], limit: int = 10) -> list[PhoneResult]:
    search_batches = await asyncio.gather(
        *(web_search(q) for q in queries), return_exceptions=True
    )
    urls: list[str] = []
    for batch in search_batches:
        if isinstance(batch, BaseException):
            continue
        for hit in batch.hits:
            if hit["url"] not in urls:
                urls.append(hit["url"])

    pages = await asyncio.gather(*(fetch_page(u) for u in urls[:8]), return_exceptions=True)
    valid = [p for p in pages if isinstance(p, dict)]

    extracted = await asyncio.gather(*(extract_prices(p) for p in valid), return_exceptions=True)

    seen: dict[str, PhoneResult] = {}
    for group in extracted:
        if isinstance(group, Exception):
            continue
        for phone in group:
            key = phone.name.lower()
            if key not in seen or phone.price_usd < seen[key].price_usd:
                seen[key] = phone
    return sorted(seen.values(), key=lambda p: p.price_usd)[:limit]
