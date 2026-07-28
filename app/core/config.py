"""
Cấu hình ứng dụng.

NestJS tương đương:
  ConfigModule.forRoot() + ConfigService.getOrThrow('...')
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Đọc biến môi trường từ .env — giống @nestjs/config."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "my-ai-app"
    app_env: str = "development"
    debug: bool = True

    # Học FastAPI: mặc định mock để chạy được không cần API key
    mock_llm: bool = True

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"

    database_url: str = "sqlite+aiosqlite:///./app.db"


@lru_cache
def get_settings() -> Settings:
    """
    Singleton settings — ≈ provider ConfigService trong NestJS.
    Dùng với Depends(get_settings) trong router.
    """
    return Settings()
