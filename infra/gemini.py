

from __future__ import annotations

import logging
import time
from typing import Any

from core.config import Settings, get_settings
from infra.citations import WebCitation, WebContext
from infra.urls import host_in_domains

logger = logging.getLogger(__name__)

# Google reports quota problems as 429 / RESOURCE_EXHAUSTED depending on transport.
_QUOTA_MARKERS = (
    "resource_exhausted",
    "resource exhausted",
    "quota",
    "rate limit",
    "too many requests",
)

# Backwards-compatible alias: grounding returns a generic web context.
GroundedResult = WebContext

__all__ = [
    "GeminiGroundingClient",
    "GeminiGroundingError",
    "GeminiQuotaExhausted",
    "GroundedResult",
    "WebCitation",
    "WebContext",
    "get_gemini_client",
]


class GeminiQuotaExhausted(Exception):
    """Raised when Gemini returns 429 / RESOURCE_EXHAUSTED (fallback to web_search)."""


class GeminiGroundingError(Exception):
    """Non-quota Gemini failures."""


def _extract_citations(response: Any) -> list[WebCitation]:
    citations: list[WebCitation] = []
    seen: set[str] = set()
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return citations
        meta = getattr(candidates[0], "grounding_metadata", None)
        if meta is None:
            return citations
        for chunk in getattr(meta, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web is None:
                continue
            url = (getattr(web, "uri", None) or getattr(web, "url", None) or "").strip()
            title = (getattr(web, "title", None) or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            citations.append(WebCitation(title=title or url, url=url, source="gemini"))
    except Exception:  # noqa: BLE001
        logger.exception("Failed to parse Gemini grounding chunks")
    return citations


def _extract_search_suggestions(response: Any) -> str:
    """`searchEntryPoint.renderedContent` — required by Google's usage terms."""
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""
        meta = getattr(candidates[0], "grounding_metadata", None)
        entry = getattr(meta, "search_entry_point", None) if meta else None
        return (getattr(entry, "rendered_content", None) or "").strip()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to parse Gemini search entry point")
        return ""


def _is_quota_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _QUOTA_MARKERS)


class GeminiGroundingClient:
    # Shared across instances: the factory builds a new client per request.
    _quota_blocked_until: float = 0.0

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @classmethod
    def mark_quota_exhausted(cls, cooldown_seconds: int) -> None:
        cls._quota_blocked_until = time.monotonic() + max(cooldown_seconds, 0)

    @classmethod
    def reset_quota_block(cls) -> None:
        cls._quota_blocked_until = 0.0

    @property
    def quota_blocked(self) -> bool:
        return time.monotonic() < type(self)._quota_blocked_until

    @property
    def available(self) -> bool:
        configured = bool(
            self.settings.gemini_grounding_enabled and self.settings.gemini_api_key
        )
        return configured and not self.quota_blocked

    async def grounded_complete(
        self,
        system: str,
        user: str,
        *,
        prefer_url: str | None = None,
        preferred_domains: list[str] | None = None,
    ) -> WebContext:
        if not self.available:
            raise GeminiGroundingError(
                "Gemini grounding disabled, key missing, or in quota cooldown"
            )

        if self.settings.mock_llm:
            return WebContext(
                text=(
                    "[MOCK GEMINI GROUNDING] Không gọi API thật.\n"
                    f"→ Câu hỏi: {user[:200]}"
                ),
                ok=True,
            )

        from google import genai
        from google.genai import types

        # The Gemini Developer API has no hard domain filter, so preference is
        # expressed in the prompt and by ordering citations afterwards.
        domains = list(preferred_domains or [])
        hint = ""
        if domains:
            hint = f"\nƯu tiên các nguồn: {', '.join(domains)}."
        if prefer_url:
            hint += (
                f"\nƯu tiên trang: {prefer_url}. "
                "Chỉ nêu thông tin có thể cite bằng URL."
            )

        client = genai.Client(
            api_key=self.settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=self.settings.gemini_timeout_seconds * 1000
            ),
        )
        config = types.GenerateContentConfig(
            system_instruction=system + hint,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.2,
        )

        try:
            response = await client.aio.models.generate_content(
                model=self.settings.gemini_model,
                contents=user,
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            if _is_quota_error(exc):
                self.mark_quota_exhausted(self.settings.gemini_quota_cooldown_seconds)
                raise GeminiQuotaExhausted(str(exc)) from exc
            raise GeminiGroundingError(str(exc)) from exc

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise GeminiGroundingError("Gemini returned empty grounded text")

        citations = _extract_citations(response)
        if domains:
            preferred = [c for c in citations if host_in_domains(c.url, domains)]
            citations = preferred + [c for c in citations if c not in preferred]

        return WebContext(
            text=text,
            citations=citations,
            search_suggestion_html=_extract_search_suggestions(response),
            ok=True,
        )


def get_gemini_client() -> GeminiGroundingClient:
    return GeminiGroundingClient()
