
from __future__ import annotations

from functools import lru_cache

from supabase import Client, create_client

from core.config import Settings, get_settings


class DatabaseNotConfiguredError(RuntimeError):
    pass


@lru_cache
def get_supabase() -> Client:
    settings = get_settings()
    key = settings.supabase_key
    if not settings.supabase_url or not key:
        raise DatabaseNotConfiguredError(
            "Set SUPABASE_URL and SUPABASE_SECRET_KEY (or SUPABASE_SERVICE_KEY) in .env"
        )
    return create_client(settings.supabase_url, key)


def get_database(settings: Settings | None = None) -> Client:
    _ = settings or get_settings()
    return get_supabase()
