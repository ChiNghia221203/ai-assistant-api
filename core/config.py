from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ai-assistant-api"
    app_env: str = "development"
    debug: bool = True

    mock_llm: bool = True

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = "text-embedding-3-small"
    # Cheap/fast model for structured intent extract (function calling).
    # gpt-4.1-nano ≈ $0.10/M in; gpt-5-nano ≈ $0.05/M in (cheapest).
    # Keep answer model (openai_model) separate — extract only needs classification.
    openai_extract_model: str = "gpt-4.1-nano"

    supabase_url: str = ""
    supabase_service_key: str = ""
    # New Supabase dashboard key names (fallback)
    supabase_secret_key: str = ""
    supabase_publishable_key: str = ""
    supabase_jwt_secret: str = ""
    supabase_jwks_url: str = ""

    google_places_api_key: str = ""

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_grounding_enabled: bool = True
    gemini_timeout_seconds: int = 45
    # Skip Gemini for this long after a quota error (circuit breaker)
    gemini_quota_cooldown_seconds: int = 900

    # Fallback live search (OpenAI Responses `web_search` tool)
    openai_search_model: str = "gpt-5.5"
    # Trusted sources for static-fact / out-of-catalog hotel grounding.
    # Price / promotion / live-status questions ignore this and search the open web
    # (answer is flagged reference_only).
    web_search_allowed_domains: str = (
        "tripadvisor.com,tripadvisor.com.vn,traveloka.com,www.traveloka.com"
    )

    # RAG sufficiency gate (review chat cascade)
    rag_min_quotes: int = 2
    rag_min_similarity: float = 0.35
    # Quotes fetched per source (tripadvisor / chudu24) before merge + top_k cap
    rag_per_source_k: int = 2

    # How many previous turns are replayed to the model verbatim
    chat_history_limit: int = 8
    # Older turns are folded into a rolling summary, this many per refresh
    chat_summary_batch: int = 20

    # Browser origins allowed to call the API (auth uses bearer tokens)
    cors_allow_origins: str = ""

    # Chat costs money per turn, so cap per identity and per network
    rate_limit_enabled: bool = True
    chat_rate_limit_per_minute: int = 12
    chat_rate_limit_per_day: int = 200
    chat_ip_rate_limit_per_minute: int = 40
    chat_ip_rate_limit_per_day: int = 600

    # Fallback local URL (unused when supabase is configured)
    database_url: str = "sqlite+aiosqlite:///./app.db"

    snapshot_stale_days: int = 30

    @property
    def supabase_key(self) -> str:
        """Prefer service/secret key for server-side ingest."""
        return self.supabase_service_key or self.supabase_secret_key

    @property
    def cors_origins(self) -> list[str]:
        origins = [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]
        return origins or ["http://localhost:3000"]

    @property
    def allowed_search_domains(self) -> list[str]:
        return [
            d.strip().lower()
            for d in self.web_search_allowed_domains.split(",")
            if d.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
