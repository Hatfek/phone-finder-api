# phone-finder-api

Agent backend for a guided phone finder. The API returns structured intents, never prose.
The browser client lives in its own repo,
[phone-finder-ui](https://github.com/Hatfek/phone-finder-ui), and talks to this over plain HTTP.

LangGraph (orchestration) · LangSmith (observability) · Ollama (local LLM) · FastAPI (transport).

## Scope

**This server has no authentication.** It is built for personal, local use — anything that can
reach the port can create, read, answer and delete every thread. Do not expose it to the public
internet. If you run it anywhere but `localhost`, put your own network control in front of it: a
VPN, an SSH tunnel, a firewall rule, or a reverse proxy that authenticates.

## Setup

Requires **Python 3.11+** and a running [Ollama](https://ollama.com/download). No API key is
needed to start: without a Tavily key the app falls back to retailer search pages, and every
other service is optional.

### 1. Ollama and the model

Install Ollama, then pull the dev model:

```bash
ollama pull qwen3.5:2b     # ~2.7 GB, the default
ollama list                # confirm it is there
curl -s localhost:11434/api/tags   # confirm the server is up
```

Ollama serves on `http://localhost:11434` and starts on its own when installed as an app; if
`curl` above fails, run `ollama serve` in another terminal. Point `OLLAMA_BASE_URL` elsewhere to
use a remote host.

`qwen3.5:2b` is the dev default — small, quiet and fast enough for the loop. `qwen3.5:9b` gives
noticeably better `why` lines; pull it only when you want a final pass, and drive it through
`scripts/gate_plan.py --quality` rather than editing `.env`:

```bash
ollama pull qwen3.5:9b     # ~6.6 GB
python scripts/gate_plan.py --quality
```

**`LLM_REASONING=false` is mandatory** for `qwen3.5` and every other thinking model. With
reasoning on, the whole reply lands in the model's `thinking` field, `content` comes back empty,
and each structured call fails after ~68s instead of succeeding in ~2s. `.env.example` already
sets it to `false` — leave it there.

### 2. Install and configure

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` is gitignored and is the only place a key belongs. It carries 28 settings. One of them
has no usable default and must be set before a browser client can talk to the server:
**`CORS_ORIGINS`** (empty allows no origin, and `.env.example` ships a placeholder that matches
nothing). The rest run as-is.

### 3. Tavily (optional, but it is the difference between one source and several)

Without `TAVILY_API_KEY` the search node skips the provider and hits retailer search pages
directly — Amazon responds, most others block, so results come from one source. A free key at
[tavily.com](https://tavily.com) lifts that. Create the key, then put it in `.env`:

```bash
TAVILY_API_KEY=tvly-your-key-here
```

A failing or absent provider degrades to the retailer fallback rather than ending the turn.

### 4. Run

```bash
python run.py            # honours DEBUG for reload
# or
uvicorn app.server:app --reload
```

`GET /health` reports which model the process picked up:

```bash
curl -s localhost:8000/health
```

The browser client is a separate repo — see
[phone-finder-ui](https://github.com/Hatfek/phone-finder-ui) — and set `CORS_ORIGINS` to the
origin it is served from.

## API

| Method | Path | Returns |
|---|---|---|
| `POST` | `/threads` | `{thread_id, step, ask_question, show_phones, done}` |
| `POST` | `/threads/{id}/answer` | same shape |
| `POST` | `/threads/{id}/revise` | same shape — clears one named filter, keeps the rest |
| `POST` | `/threads/{id}/reset` | same shape — clears every filter, keeps the profile |
| `GET` | `/slots` | the canned question behind every revisable filter |
| `GET` | `/threads/{id}` | current state |
| `DELETE` | `/threads/{id}` | 204 |
| `GET` | `/health` | model and price reference |

```bash
curl -s localhost:8000/threads \
  -H 'content-type: application/json' \
  -d '{"profile":"I drive all day, rarely near a charger, I photograph equipment."}'

curl -s localhost:8000/threads/$ID/answer \
  -H 'content-type: application/json' \
  -d '{"answer":"Android"}'
```

### Response shape

Every turn-running route — `/threads`, `/answer`, `/revise`, `/reset` — returns the **same**
object, and `GET /threads/{id}` returns it too. The client renders this and decides nothing of
its own.

```json
{
  "thread_id": "a1b2c3d4e5f6",
  "step": "2 of 4",
  "done": false,
  "ask_question": {
    "question": "What is your budget?",
    "options": ["Under $400", "$400-700", "$700-1000", "No preference"],
    "hint": "US retail",
    "slot": "max_price_usd"
  },
  "show_phones": [
    {
      "name": "Samsung Galaxy S24",
      "price_usd": 449,
      "retailer": "Amazon",
      "url": "https://www.amazon.com/dp/B0CMDRCZBJ",
      "image": "https://...",
      "fetched_at": "2026-09-05",
      "why": "Long battery life and a strong main camera.",
      "fit_summary": "Samsung — the brand you asked for · $449 — 90% of your $500 budget",
      "fit_factors": [{"filter": "brand", "label": "Samsung, as asked", "points": 2.5}]
    }
  ],
  "filters": {
    "os": "Android",
    "max_price_usd": 600,
    "priority": "camera",
    "size": "large",
    "brands": ["samsung"],
    "exclude_brands": [],
    "soft_brands": [],
    "resolved": ["os", "size"]
  },
  "price_reference": "US retail, USD",
  "notice": null,
  "degraded": [],
  "warnings": []
}
```

| Field | Meaning |
|---|---|
| `thread_id` | 12 hex chars. Every later call carries it; the graph resumes from `state.db` |
| `step` | `"2 of 4"` — progress against `MAX_QUESTIONS` |
| `done` | derived from the graph, not from a stored flag. `true` = the shortlist is final |
| `ask_question` | the next question, or `null` when there is nothing left to ask. `slot` names the filter it fills |
| `show_phones` | the ranked shortlist, `[]` before the first search. `fit_factors` says *which* filter drove the rank |
| `filters` | the live filter state, drawn as the preference bar. `resolved` lists slots answered "no preference" |
| `price_reference` | `"US retail, USD"` — prices are US retail only; the same phone costs something different in every market |
| `notice` | one sentence when the search had to widen or came back thin |
| `degraded` / `warnings` | tags and their sentences when an upstream failed — see *Degrading instead of failing* |

Both `ask_question` and `show_phones` can be present in the same turn: the shopper sees the
current shortlist and the next question together.

Errors: `400` reset with no profile · `404` unknown thread · `409` a turn is already running, or
`/revise` on an unset filter · `422` bad `thread_id`, blank/oversized body, or a `/revise` slot
that is not a filter · `429` rate limit, with `Retry-After` · `503` turn ceiling. A turn that
fails *after* the thread exists is **not** an error code — it returns `200` with the thread's
real state and a `degraded` tag.

## Extraction

Retailer search pages are parsed structurally before the model is involved. `parse_cards()` reads
each Amazon search-result card for its ASIN, title, price and image, and builds a canonical
`https://www.amazon.com/dp/{ASIN}` link; sponsored rows are skipped. When a page yields cards,
`extract_prices()` returns them directly and makes **no** LLM call — this is both the accurate path
(a DOM price belongs to the listing it was read from) and the fast one.

Pages with no parseable cards fall back to the older path: flatten the text, ask the model for
names and prices, and drop any price that does not literally appear on the page. That fallback is
skipped entirely for listing pages (`is_listing_page`), because a search URL names no single
product — without it, results would link to a list instead of a phone. The invariant is that
**every result links to one product**; `extract` drops anything that does not.

Amazon is currently the only live source. Walmart, Best Buy, B&H and GSMArena all block or return
empty shells; set `TAVILY_API_KEY` to widen coverage.

## Preferences

Filters stack. Nothing the shopper settles is ever cleared by a later turn:

- `Filters` carries `os`, `max_price_usd`, `priority`, `size`, plus `brands` /
  `exclude_brands` read from free text, and `resolved`, the slots the shopper has actually
  settled. "No preference" is a decision, so the slot never gets asked twice.
- One answer can fill several slots. "Large screen android under $600" sets size, OS and
  budget in one turn; a reply that answers a different slot than the one asked is kept, and
  the asked slot is asked again.
- The planner may only fill slots that are still empty. Its output can never overwrite or
  blank a value the shopper gave, which is what state a small model would otherwise drop.
- Brand loyalty is a hard filter, applied on top of everything else: "I prefer Samsung" keeps
  only Samsung listings, and the budget narrows that set rather than replacing it. A brand
  named in passing ("my old Pixel died") only nudges ranking; "not an iPhone" excludes.
- A budget that returns nothing widens the *search*, not the filter — `max_price_usd` stays
  exactly as given, so later turns still respect it.
- Stacking is not a one-way street. `POST /threads/{id}/revise` takes one slot — the four
  question slots plus `brands` / `exclude_brands` — clears exactly that, and re-runs the loop
  on what is left. With an answer it applies it straight away; without one it hands back a
  question so the slot is asked again, even on a thread that had already finished. That is also
  the exit from a brand filter that emptied the shortlist: drop the brand, keep the budget.
- Only answers past `merged_answers` are folded into the filters. Re-reading the last answer on
  every pass, as `plan` used to, would put a revised slot straight back.

## Relevance

The profile drives the search, not just the question flow:

- `intake` maps what the shopper *does* to a priority. A job or habit is treated as evidence;
  `allround` is a last resort, not the default.
- Queries are written from the profile **and** the filters, and must describe the product rather
  than the shopper — "android camera phone 5g", never "phone for a surveyor".
- The budget is applied as a real retailer filter, not as words in the query.
  `price_band(cap)` yields `(max(90, cap//2), cap)` and becomes Amazon's `rh=p_36:` parameter,
  which is the difference between 4/15 and 15/16 results landing inside the budget.
- `os` is enforced on results, not merely searched for: an iOS shopper never sees an Android
  phone. Queries use the catalogue's word for it ("Apple iPhone"), since retailers do not sell
  "iOS phones" and listings mentioning "iOS" are usually Android accessories claiming compatibility.
- A result must look like a phone (`PHONE_RE`) and must not look like an accessory (`JUNK_RE`).
  The positive test matters: a blocklist alone let a pet tracker through on "Phone App".
- `_fit_factors` scores each candidate against the filters the shopper actually set — brand,
  budget, priority, size, OS — and `_fit_score` is the sum. Every term carries the filter it
  came from and a label in the shopper's own terms, so a result ships with both a `fit_summary`
  ("Samsung — the brand you asked for · $449 — 90% of your $500 budget") and the full
  `fit_factors` breakdown. Over-budget results are penalised and say so. `annotate` then
  shortlists and writes the prose `why` line on top.
- The two are different things and are shown as such: the deterministic `fit_summary` is the
  tile line, because it is specific to that phone and cannot repeat across a shortlist the way
  the model's sentence does; the model's `why` leads the detail sheet, with the factor
  breakdown under it.

Ranking reads listing *titles*, not spec sheets — it is a good heuristic, not ground truth, and
the compare table says so above the numbers.

## Graph

```
START → intake → plan → search → extract → annotate ─┬→ ask → plan
                                                     └→ END
```

One turn runs `intake → plan → search → extract → annotate`, then either stops at `ask` or ends.

- **`intake`** runs once per thread. It turns the free-text profile into a first set of `Filters`,
  so the first search is already shaped by what the shopper wrote.
- **`plan`** folds the newest answer into the filters and decides the next question — or that
  there is nothing worth asking. It is the only node that decides the loop continues.
- **`search`** builds 2-4 queries from the profile *and* the filters. The budget goes out as a
  retailer price band, not as words.
- **`extract`** fetches the result pages and pulls `PhoneResult` rows out of them.
- **`annotate`** shortlists, ranks, and writes the `why` line and `fit_factors` per result.
- **`ask`** calls `interrupt()`. The graph pauses mid-run, the API returns the payload, and the
  client's next POST resumes on the same `thread_id` — landing back in `plan` with the answer.

That `ask → plan` edge is the loop: each cycle is one question and one refreshed shortlist, so
results improve while the questions are still being answered. It ends when `plan` has nothing
left to ask or the question cap (`MAX_QUESTIONS`, default 4) is hit — enforced both in `plan` and
on the routing edge — at which point the graph runs to `END` and the response carries
`done: true`.

State lives in `state.db` via the SQLite checkpointer, so the backend is stateless from the
client's point of view: the whole session is recoverable from a `thread_id` alone.

| Node | Owner | Does |
|---|---|---|
| `intake` | planner | Free-text profile → initial `Filters` |
| `plan` | planner | State → filters + next question, or done |
| `search` | search-agent | Builds 2-4 queries, runs web search |
| `extract` | search-agent | Fetches pages, extracts `PhoneResult` |
| `annotate` | planner | Shortlists, ranks, writes the `why` line |
| `ask` | — | Interrupt |

## Model

`get_llm()` in `app/llm.py` is the only place a model is constructed. Swapping Ollama
for a hosted model is a one-file change.

`LLM_REASONING=false` is required for thinking models such as `qwen3.5`. With reasoning
on, the whole reply lands in the `thinking` field, `content` comes back empty, and every
structured call fails after ~68s. With it off the same call takes ~2s.

Dev default is `OLLAMA_MODEL=qwen3.5:2b`. `OLLAMA_MODEL_QUALITY=qwen3.5:9b` is the final-pass
model: `python scripts/gate_plan.py --quality` runs the gate on it without editing `.env`.

## Observability

Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`. Tracing hooks the LangChain layer,
so Ollama runs trace exactly like hosted ones.

Every structured call carries a `run_name` (`intake`, `plan`, `search_queries`,
`extract_prices`, `annotate`) and every turn is tagged `thread:{id}`, so a whole
conversation reconstructs from one filter.

Watch four things:

| Signal | Why | Threshold |
|---|---|---|
| `extract_prices` latency | dominates the turn | ~30s of a 35s turn |
| Structured-output failures | local models return bad JSON | over 5% → change model, not prompts |
| Question count per session | planner losing its stopping condition | must never exceed 4 |
| Dropped results | a retailer changed their page | rising = extraction breaking |

Dropped results log as `dropped {name}: price {n} not on page`.

## Caching

Three namespaces under `.cache/`, 24h TTL: `search` (by query), `page` (by URL),
`extract` (by URL). Extraction is cached because the same retailer page is otherwise
re-extracted on every turn, and that LLM pass is most of the turn's latency.

## Hardening

| # | Guard | Where |
|---|---|---|
| 1 | Structured output retried twice, then a hardcoded question | `structured.py`, `fallbacks.py` |
| 2 | Question cap of 4, in code and the routing edge | `nodes.plan`, `graph.route_after_annotate` |
| 3 | Empty results widen the budget once, then say so | `nodes.extract` |
| 4 | Price options validated against actual results | `nodes._validate_options` |
| 5 | Fetch 10s, one LLM call 60s, turn 150s | `.env` |
| 6 | Price must literally appear on the page or the result is dropped | `tools._price_in_text` |
| 7 | Retailer allow-list | `tools.ALLOWED_HOSTS` |
| 8 | CORS limited to `CORS_ORIGINS`, no credentials, `GET`/`POST`/`DELETE` only | `server.py` |
| 9 | Per-IP rate limit on the three POST routes, `429` + `Retry-After` | `limits.RateLimiter` |
| 10 | One turn per thread (`409`) and a global turn ceiling (`503`) | `limits.TurnGuard` |
| 11 | `thread_id` must match `^[0-9a-f]{12}$` on every thread route | `api_models.THREAD_ID_PATTERN` |
| 12 | Bodies stripped, non-blank, length-capped (profile 4000, answer 500) | `api_models` |
| 13 | A dead search provider or unreadable page is skipped, not raised | `tools.web_search`, `tools.fetch_page` |
| 14 | An unreachable model falls back to canned questions and unranked results | `structured.py`, `nodes.plan`, `nodes.annotate` |
| 15 | A node exception or turn timeout returns the thread's state, never a 500 | `server._run` |
| 16 | An answer to a stalled thread finishes the stalled run instead of being eaten | `server.answer` |
| 17 | Global per-IP ceiling (100/min) on top of the 10/min turn limit | `server.guard_request` |
| 18 | Bodies over `MAX_REQUEST_BYTES` (10MB) refused with `413` before parsing | `server.guard_request` |
| 19 | Outbound fetches restricted to `ALLOWED_DOMAINS`, every redirect hop re-checked | `net.assert_safe_url` |
| 20 | Any host resolving to a private, loopback or link-local address is refused | `net.is_blocked_ip` |
| 21 | Redirects off by default; when on, the whole chain shares a 10s budget and 5-hop cap | `net.safe_get` |
| 22 | Free text NFKC-normalised, control and zero-width characters stripped | `api_models.sanitize_text` |

There is no authentication row: the server has none. Every guard above is independent of who is
calling — see [Scope](#scope).

## Degrading instead of failing

Upstream failures are expected, so no single one ends a thread.

| Upstream | What happens | What the client is told |
|---|---|---|
| Tavily errors or times out | falls through to the retailer search; the failed provider is logged | nothing, the fallback covered it |
| Every search provider fails | no URLs; the turn still asks its question | `degraded: ["search"]` |
| A retailer page 403s, times out or will not parse | that source is skipped, the others still extract | nothing, unless every source fails |
| Every page fails | the previous turn's shortlist is kept and the budget is **not** widened | `degraded: ["fetch"]` |
| Ollama is unreachable at `plan` | the canned question for the slot is asked | `degraded: ["llm"]` |
| Ollama is unreachable at `annotate` | the shortlist is kept, trimmed to 9, with no `why` lines | `degraded: ["llm"]` |
| A node raises, or the turn times out | the run's own state comes back with the failure named | `degraded: ["turn"]` |
| An answer arrives on a thread stalled mid-run | the stalled run is finished first; the answer is not silently dropped | `degraded: ["resumed"]` |

A failed search or page fetch is never cached, so a retry re-reaches the source rather than
replaying an outage for 24 hours. `TurnResponse` carries `degraded` (tags, for the inspector)
and `warnings` (a sentence per tag, rendered as banners). An unreachable model is separated
from a badly-formed answer: a bad answer is retried twice with a repair hint, an unreachable
one returns immediately, which is why a turn with Ollama down takes about a second.

The client tells "the backend is down" apart from "this turn failed". A failed turn keeps the
question, the shortlist and the thread on screen and offers **Try again**; an unreachable
backend offers **Reconnect**, which re-probes `/health` before retrying. Only a `404` — the
thread is genuinely gone — clears the session.

The rate limit is a fixed window per client IP, counted across `POST /threads`,
`POST /threads/{id}/answer` and `POST /threads/{id}/reset` — the three routes that run the
graph. `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` set it; `MAX_CONCURRENT_TURNS` caps
how many turns run at once. The IP comes from the socket, not from `X-Forwarded-For`, so behind
a proxy the limit is per-proxy until that header is explicitly trusted.

## Configuration

Every setting is an environment variable read by `Settings` in `app/config.py`; `.env.example`
lists all of them. Nothing — model name, Ollama host, Tavily key, LangSmith key — is hardcoded
outside that file's defaults.

| Group | Keys |
|---|---|
| Model | `OLLAMA_MODEL`, `OLLAMA_MODEL_QUALITY`, `OLLAMA_BASE_URL`, `LLM_REASONING` |
| Search | `TAVILY_API_KEY` |
| SSRF | `ALLOWED_DOMAINS`, `DISABLE_REDIRECTS` |
| Observability | `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT` |
| Storage | `DATABASE_URL`, `CHECKPOINT_DB`, `CACHE_TTL_HOURS` |
| Limits | `FETCH_TIMEOUT`, `TURN_TIMEOUT`, `MAX_QUESTIONS` |
| Transport | `CORS_ORIGINS`, `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`, `GLOBAL_RATE_LIMIT_REQUESTS`, `GLOBAL_RATE_LIMIT_WINDOW_SECONDS`, `MAX_CONCURRENT_TURNS`, `MAX_REQUEST_BYTES` |
| Display | `PRICE_REFERENCE`, `DEBUG`, `DEBUG_UI` |

`.env` is gitignored and is the only place a key belongs. `get_settings()` exports the
LangSmith variables into the process environment itself, assigning rather than defaulting, so
a stale `LANGCHAIN_TRACING_V2` in the shell cannot override `LANGSMITH_TRACING=false`.

### SSRF and outbound fetching

The `extract` node fetches retailer pages, so it is the one place the server makes a request to
a URL that came from outside. Three controls apply, in `app/net.py`:

1. **Allow-list.** `ALLOWED_DOMAINS` is the live list; a host must match one entry exactly or be
   a subdomain of it. The label map in `tools.ALLOWED_HOSTS` only supplies display names
   ("Amazon") — the environment wins when the two disagree.
2. **Private-address block.** Every resolved address is checked, and refused if it is loopback,
   private, link-local, reserved, multicast or unspecified — `127.0.0.0/8`, `0.0.0.0/8`, `::1`,
   `169.254.0.0/16`, `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, plus `fc00::/7` and
   `fe80::/10`. This is what stops an allow-listed domain pointed at your metadata endpoint.
3. **Redirect handling.** `DISABLE_REDIRECTS=true` (the default) refuses any `3xx` outright. Set
   it to `false` and redirects are followed *manually*, re-running both checks above on every
   hop, capped at 5 hops with the whole chain sharing one 10-second budget.

A refused URL is logged and the page skipped — the turn degrades exactly as any other fetch
failure does, and nothing is cached.

### CORS

`CORS_ORIGINS` is a comma-separated allow-list of real origins. There is no wildcard, no
`allow_credentials`, only `GET`/`POST`/`DELETE` are permitted, and `content-type` is the only
allowed request header — the client sends no credentials of any kind.

**`CORS_ORIGINS` has no default and must be set explicitly.** Unset, it is empty, the allow-list
is empty, and the browser blocks every cross-origin call. A missing setting refuses traffic
rather than quietly serving an origin nobody chose. `.env.example` ships the placeholder
`https://your-frontend-domain.example`, which matches nothing, so it fails loudly instead of
appearing to work.

Set it to the actual origin the UI is served from — `http://localhost:5500` for the local
static server, the real HTTPS origin in production.

### Rate limits

Two limits, both per client IP, both applied:

| Limit | Scope | Default |
|---|---|---|
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW_SECONDS` | the four turn-running routes | 10 per 60s |
| `GLOBAL_RATE_LIMIT_REQUESTS` / `GLOBAL_RATE_LIMIT_WINDOW_SECONDS` | every route | 100 per 60s |

The global ceiling stops a client from hammering the cheap routes; the turn limit protects the
expensive ones. Both are per IP and are the only abuse control the server has.

The IP comes from the socket, not `X-Forwarded-For`, so behind a proxy both limits are
per-proxy until that header is explicitly trusted.

## Security considerations

### Threat model

This service is designed for personal, local use — bound to a machine the operator controls,
serving a browser client on that same machine. **It is unauthenticated and must not be exposed
to the public internet.** It assumes the operator controls `ALLOWED_DOMAINS`, that Ollama is
**not** publicly reachable, and that network access to the port is already restricted.

What is protected: outbound requests to internal addresses, oversized bodies, per-IP flooding,
and cross-origin access from origins outside `CORS_ORIGINS`. Who is calling is not.

### Known limitations

Read these before deploying — they are real and currently unmitigated.

| Limitation | Impact | Status |
|---|---|---|
| **DNS rebinding.** `assert_safe_url` resolves the host, checks the addresses, then httpx resolves again when it connects. An attacker controlling DNS with a sub-TTL record could return a public address to the check and a private one to the connection. | Narrow TOCTOU window on outbound fetches | Open. Fix is pinning the connection to the validated IP with a custom transport |
| **No authentication at all.** Any client that can reach this server can create, read, answer and delete every thread. Do not expose it to the public internet. | Full access to every thread for anyone who reaches the port | By design — personal/local use only. Restrict access at the network layer |
| **Prompt injection.** Retailer page text reaches the model. `sanitize_text` covers client input, not fetched pages. | A hostile page could influence extraction | Partly mitigated: prices must appear literally on the page, and results are schema-validated |
| **`GET /health` discloses the model name and price reference.** Deliberate, so a client can probe the server before starting a thread. | Minor information disclosure | By design |

### Reporting a vulnerability

Open a private security advisory on the repository rather than a public issue.

## Development

```bash
ruff check .          # line-length 100, config in ruff.toml
python -m pytest -q   # 149 tests, ~0.6s
```

The suite is pure logic plus a `TestClient` driven against a fake graph, so it needs **no
Ollama and no network** — which is exactly what `.github/workflows/ci.yml` runs on
every push and pull request (Python 3.11, `ruff check` then `pytest`).

### Gates

These do need a live Ollama, so they are run locally rather than in CI:

```bash
python scripts/gate_plan.py            # 10 runs of intake + plan, must be 10/10
python scripts/gate_plan.py --quality  # same gate on OLLAMA_MODEL_QUALITY
python scripts/session.py              # full 4-question session end to end
./scripts/curl_session.sh              # the same, over HTTP
```

## Known limits

- **Search needs a Tavily key.** Without `TAVILY_API_KEY` the fallback hits retailer
  search pages directly. DuckDuckGo HTML scraping returns a 202 anti-bot challenge and
  no longer works. Best Buy times out, B&H returns 403, GSMArena serves an empty shell.
  Amazon and Walmart respond. Set the key for real coverage.
- **Extraction dominates latency** — roughly 30s of a 40s turn.
- **`why` lines repeat** on a 2b model. Larger model fixes it; architecture does not.

## License

MIT — see [LICENSE](LICENSE).
