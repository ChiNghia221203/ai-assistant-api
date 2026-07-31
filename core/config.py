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

    supabase_url: str = ""
    supabase_service_key: str = ""
    # New Supabase dashboard key names (fallback)
    supabase_secret_key: str = ""
    supabase_publishable_key: str = ""
    supabase_jwt_secret: str = ""
    supabase_jwks_url: str = ""

    google_places_api_key: str = ""

    # Fallback local URL (unused when supabase is configured)
    database_url: str = "sqlite+aiosqlite:///./app.db"

    snapshot_stale_days: int = 30

    @property
    def supabase_key(self) -> str:
        """Prefer service/secret key for server-side ingest."""
        return self.supabase_service_key or self.supabase_secret_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
