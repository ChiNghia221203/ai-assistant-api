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
# optional: GOOGLE_PLACES_API_KEY
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
| POST | `/api/v1/chat/review` | Evidence + RAG + LLM (no verdict) |
| GET/POST | `/api/v1/chat/conversations` | Persist chat (Supabase Auth `user_id`) |

## Methodology (shown to users)

- **Chudu24:** up to 100 newest reviews by date + `date_min`/`date_max` + `sample_mean`
- **TripAdvisor:** up to 100 newest reviews by date (Playwright crawl) + site overall
- **Google (optional):** Places API rating + returned reviews (often ≤5) — not claimed as 100 newest
