# Hotel multi-source review (Sài Gòn)

FastAPI backend: evidence from Chudu24 + TripAdvisor (optional Google later) stored in Supabase/pgvector → chat with quotes (no verdict).

Corpus already lives in Supabase. Crawl/refresh scripts stay **local-only** (gitignored) — see [`scripts/README.md`](scripts/README.md).

## Setup

1. Create a Supabase project, enable the `vector` extension, run:

```bash
# In Supabase SQL Editor, paste:
# supabase/migrations/001_hotel_review.sql
```

2. Copy env:

```bash
cp .env.example .env
# fill SUPABASE_URL, SUPABASE_SERVICE_KEY (or SUPABASE_SECRET_KEY), OPENAI_API_KEY
# optional: GEMINI_API_KEY (Google Search grounding), GOOGLE_PLACES_API_KEY
```

3. Install & run:

```bash
uv sync
uv run uvicorn main:app --reload --port 8000
```

For local crawl/refresh (scripts present on this machine only):

```bash
uv run playwright install chromium
uv run python scripts/ingest_tripadvisor.py --slug <slug>
uv run python scripts/ingest_chudu24.py --slug <slug>
uv run python scripts/refresh_stale.py --days 30
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/places` | List hotels |
| GET | `/api/v1/places/{id}/evidence` | Multi-source evidence card |
| POST | `/api/v1/rag/search` | Quote retrieval |
| POST | `/api/v1/chat/` | Full cascade, lightweight response (reply + source metadata) |
| POST | `/api/v1/chat/review` | Full cascade + evidence tables and quotes |
| GET/POST | `/api/v1/chat/conversations` | Persist chat (Supabase Auth `user_id`) |

## Review chat retrieval cascade

`POST /api/v1/chat/review` resolves sources in order:

Grounding runs **only when needed**, based on the question intent:

| Intent | Example | Sources |
|---|---|---|
| Experience / quality | "phòng có ồn không?" | DB only — never grounded |
| Static fact | "có hồ bơi không?" | DB first; grounding restricted to `WEB_SEARCH_ALLOWED_DOMAINS` when retrieval is weak |
| Price / promotion / live status | "giá phòng bao nhiêu?" | Always grounded on the open web, answer flagged `reference_only` |

Source order for a grounded question:

1. **RAG / DB evidence** (Chudu24 + TripAdvisor + Google snapshots + quote vectors) — the only source for ratings/quotes
2. **Gemini Grounding** (Google Search)
3. **OpenAI `web_search`** — same source policy, used when Gemini is down / quota exhausted
4. **Abstain** — beyond the ingested sample (e.g. 200 reviews) or no usable source remains

Response includes `retrieval_source` (`rag` | `rag+gemini` | `rag+web_search` | `abstain`), `web_citations` when live web was used, `search_suggestion_html` (Google Search Suggestions, must be rendered by the client when grounding ran), and `reference_only` for price/live answers.

Trust rules enforced in code:

- Web findings never replace DB `site_overall` / `sample_mean` / quotes
- Price/live answers are informational: rendered in their own section with a disclaimer, and never used to judge quality
- Citations only come from API-reported sources/annotations — URLs written in prose are ignored
- A live-search result counts only if at least one citation is on an allowed domain; otherwise the answer falls back to RAG or abstains
- There is no offline fallback: if `web_search` cannot run, nothing is labelled `rag+web_search`
- After a Gemini quota error the client skips Gemini for `GEMINI_QUOTA_COOLDOWN_SECONDS`

## Conversation memory

The prompt is assembled in this order:

```
conversation_summary  →  last CHAT_HISTORY_LIMIT messages  →  evidence/quotes (+ web_findings)  →  question
```

- Turns older than the verbatim window are folded into `conversations.summary` (max
  `CHAT_SUMMARY_BATCH` messages per refresh); `conversations.summarized_through` is the
  watermark so nothing is summarized twice.
- The summary is memory only — facts must still come from evidence, quotes or web findings.
- Hotels stored on the conversation are reused when the request omits `place_ids`.
- Persistence requires an authenticated caller (see below). Without a token nothing is
  written and the response returns `conversation_id: null`.

## Auth

The owner of a conversation is always taken from the `Authorization: Bearer <token>`
header — never from the request body. `POST /chat/review` and `POST /chat/` accept
requests without a token (one-off answer, nothing stored); everything that touches
stored data requires one:

| Endpoint | Token | Rule |
| --- | --- | --- |
| `POST /chat/review`, `POST /chat/` | optional | no token → answer only, no persistence |
| the same, with `conversation_id` | required | 403 unless the caller owns it |
| `GET/POST /chat/conversations` | required | scoped to the caller |
| `GET /chat/conversations/{id}/messages` | required | 403 unless the caller owns it |
| `POST /chat/conversations/claim` | required | moves an anonymous session's history over |
| `GET /auth/me` | required | echoes the verified session, for the frontend |

Tokens are verified against the project JWKS (RS256/ES256, derived from `SUPABASE_URL`)
or `SUPABASE_JWT_SECRET` for legacy HS256 projects. If neither is available the API
returns 503 rather than trusting the token.

Verification is provider-agnostic: email/password, anonymous and OAuth (Google) users
all present the same Supabase-issued token, so adding a social login is a frontend +
dashboard change with no backend work.

Anonymous users are first-class: the frontend calls `signInAnonymously()` on first
visit, so a visitor chats immediately and still gets history. `upgradeAnonymousUser()`
later attaches an email/password to that same user id, so nothing is lost on sign-up.

### Keeping guest history through a social login

`linkIdentity({ provider: 'google' })` keeps the same user id, so history survives and
nothing else is needed. It fails when manual linking is disabled or the Google account
already belongs to another user; the frontend then falls back to `signInWithOAuth`,
which creates a *different* user id and orphans the guest's conversations. Recover them
by calling `POST /chat/conversations/claim` with the new session's token in the header
and the anonymous session's token in the body — holding that token is the proof of
ownership. It only accepts anonymous guest tokens, and is a no-op when the ids match.

Note that the API uses the Supabase service key and therefore bypasses RLS — the
ownership checks in the router and in `_load_context` are what actually protect the data.

## Rate limiting

Each chat turn can cost an embedding call, a grounding call and a completion, and
anonymous sign-ups are free and unlimited, so `POST /chat/review` and `POST /chat/`
are capped twice: per identity (the user id, or the IP for tokenless callers) and per
IP. The second bucket is the one that matters — without it, minting a new anonymous
user per request would bypass the first. Exceeding either returns 429 with `Retry-After`.

Counters are in-process, so they are per worker and reset on restart. That is fine for
a single instance; back `_limiter` in `core/rate_limit.py` with Redis before scaling out.
Also enable CAPTCHA for anonymous sign-ins in the Supabase dashboard — the API cannot
rate-limit account creation, only its own endpoints.

## Methodology (shown to users)

- **Chudu24:** up to 100 newest reviews by date + `date_min`/`date_max` + `sample_mean`
- **TripAdvisor:** up to 100 newest reviews by date (Playwright crawl) + site overall
- **Google (optional):** Places API rating + returned reviews (often ≤5) — not claimed as 100 newest
