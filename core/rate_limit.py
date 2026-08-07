
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Iterable

from fastapi import Depends, HTTPException, Request, status

from core.config import Settings, get_settings
from core.security import AuthUser, optional_user

logger = logging.getLogger(__name__)

_PRUNE_EVERY_SECONDS = 300


@dataclass(frozen=True)
class Quota:
    limit: int
    window_seconds: int


class FixedWindowLimiter:
    """Counts hits per (key, window). Rejected requests are not counted."""

    def __init__(self) -> None:
        self._hits: dict[tuple[str, int, int], int] = {}
        self._lock = threading.Lock()
        self._last_prune = time.monotonic()

    def retry_after(self, key: str, quotas: Iterable[Quota]) -> int | None:
        """Seconds to wait if any quota is used up, or None when allowed."""
        quotas = [q for q in quotas if q.limit > 0]
        if not quotas:
            return None

        now = time.time()
        with self._lock:
            self._prune(now)
            for quota in quotas:
                bucket = int(now // quota.window_seconds)
                used = self._hits.get((key, quota.window_seconds, bucket), 0)
                if used >= quota.limit:
                    resets_at = (bucket + 1) * quota.window_seconds
                    return max(1, int(resets_at - now))

            for quota in quotas:
                bucket = int(now // quota.window_seconds)
                index = (key, quota.window_seconds, bucket)
                self._hits[index] = self._hits.get(index, 0) + 1
        return None

    def _prune(self, now: float) -> None:
        monotonic = time.monotonic()
        if monotonic - self._last_prune < _PRUNE_EVERY_SECONDS:
            return
        self._last_prune = monotonic
        self._hits = {
            (key, window, bucket): hits
            for (key, window, bucket), hits in self._hits.items()
            if bucket >= int(now // window)
        }

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


_limiter = FixedWindowLimiter()


def get_limiter() -> FixedWindowLimiter:
    return _limiter


def client_ip(request: Request) -> str:
    """Caller IP, trusting the first X-Forwarded-For entry when behind a proxy."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else ""


def _enforce(key: str, quotas: Iterable[Quota], *, scope: str) -> None:
    retry_after = _limiter.retry_after(key, quotas)
    if retry_after is None:
        return
    logger.warning("Rate limit hit (%s): %s, retry in %ss", scope, key, retry_after)
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Bạn đang gửi quá nhiều câu hỏi. Vui lòng thử lại sau.",
        headers={"Retry-After": str(retry_after)},
    )


def chat_rate_limit(
    request: Request,
    user: AuthUser | None = Depends(optional_user),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.rate_limit_enabled:
        return

    ip = client_ip(request)
    identity = f"user:{user.id}" if user else (f"ip:{ip}" if ip else "")
    if identity:
        _enforce(
            identity,
            [
                Quota(settings.chat_rate_limit_per_minute, 60),
                Quota(settings.chat_rate_limit_per_day, 86_400),
            ],
            scope="identity",
        )

    # Guards fresh anonymous users being minted to bypass the identity bucket.
    if ip:
        _enforce(
            f"net:{ip}",
            [
                Quota(settings.chat_ip_rate_limit_per_minute, 60),
                Quota(settings.chat_ip_rate_limit_per_day, 86_400),
            ],
            scope="network",
        )
