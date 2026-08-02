"""Domain-restricted live search fallback (OpenAI Responses `web_search` tool).

Trust rules: a result only counts when the API itself reports citations/sources
on an allowed domain. There is no offline fallback — an answer must never be
labelled as "from the web" unless a real search produced it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from core.config import Settings, get_settings
from infra.citations import WebCitation
from infra.urls import host_in_domains

logger = logging.getLogger(__name__)


class WebSearchUnavailable(Exception):
    """Live search could not run (missing key, unsupported model, HTTP error)."""


@dataclass
class WebSearchResult:
    text: str = ""
    citations: list[WebCitation] = field(default_factory=list)
    ok: bool = False


class WebSearchClient:
    """Fallback when Gemini grounding is unavailable / quota exhausted."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def available(self) -> bool:
        return bool(self.settings.openai_api_key) and not self.settings.mock_llm

    async def search(
        self,
        *,
        hotel_name: str,
        question: str,
        source_url: str | None = None,
        allowed_domains: list[str] | None = None,
    ) -> WebSearchResult:
        """Run a live search. Empty `allowed_domains` means the open web."""
        if not self.available:
            logger.debug("web_search skipped: mock_llm or missing OPENAI_API_KEY")
            return WebSearchResult()

        domains = list(allowed_domains or [])
        if domains:
            scope = f"Chỉ dùng kết quả từ các domain: {', '.join(domains)}. "
        else:
            scope = "Ưu tiên trang chính thức của khách sạn và các trang đặt phòng lớn. "

        instructions = (
            "Bạn tra cứu thông tin khách sạn trên web. "
            + scope
            + "Trả lời ngắn bằng tiếng Việt và luôn kèm URL nguồn. "
            "Không bịa số liệu. Nếu không tìm thấy, nói rõ là không tìm thấy."
        )
        user = (
            f"Khách sạn: {hotel_name}\n"
            f"Trang nguồn (nếu có): {source_url or 'N/A'}\n"
            f"Câu hỏi: {question}"
        )

        try:
            data = await self._responses_web_search(
                instructions=instructions, user=user, domains=domains
            )
        except WebSearchUnavailable as exc:
            logger.warning("web_search unavailable: %s", exc)
            return WebSearchResult()
        except Exception:  # noqa: BLE001
            logger.exception("web_search request failed")
            return WebSearchResult()

        text = self._extract_text(data)
        citations = self._extract_citations(data)
        if domains:
            citations = [c for c in citations if host_in_domains(c.url, domains)]

        # No verifiable citation → do not label the answer as web-sourced.
        if not text or not citations:
            logger.info(
                "web_search produced no usable citation (text=%s, citations=%s)",
                bool(text),
                len(citations),
            )
            return WebSearchResult(text=text, citations=citations, ok=False)

        return WebSearchResult(text=text, citations=citations, ok=True)

    async def _responses_web_search(
        self,
        *,
        instructions: str,
        user: str,
        domains: list[str],
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        tool: dict[str, Any] = {"type": "web_search"}
        if domains:
            tool["filters"] = {"allowed_domains": domains}

        payload: dict[str, Any] = {
            "model": self.settings.openai_search_model,
            "instructions": instructions,
            "input": user,
            "tools": [tool],
            # Search must actually run; a model-memory answer is not acceptable here.
            "tool_choice": "required",
            "include": ["web_search_call.action.sources"],
        }
        url = f"{self.settings.openai_base_url.rstrip('/')}/responses"

        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.status_code >= 400:
            raise WebSearchUnavailable(
                f"HTTP {response.status_code} from /responses "
                f"(model={self.settings.openai_search_model}): {response.text[:300]}"
            )
        return response.json()

    def _extract_text(self, data: dict[str, Any]) -> str:
        if isinstance(data.get("output_text"), str) and data["output_text"]:
            return data["output_text"].strip()

        parts: list[str] = []
        for item in data.get("output") or []:
            if item.get("type") != "message":
                continue
            for block in item.get("content") or []:
                if block.get("type") in ("output_text", "text"):
                    parts.append(block.get("text") or "")
        return "\n".join(p for p in parts if p).strip()

    def _extract_citations(self, data: dict[str, Any]) -> list[WebCitation]:
        """Only API-reported citations/sources — never URLs scraped from prose."""
        citations: list[WebCitation] = []
        seen: set[str] = set()

        def add(url: str, title: str = "") -> None:
            url = (url or "").strip()
            if not url or url in seen:
                return
            seen.add(url)
            citations.append(
                WebCitation(title=title or url, url=url, source="web_search")
            )

        for item in data.get("output") or []:
            item_type = item.get("type") or ""

            if item_type == "message":
                for block in item.get("content") or []:
                    for ann in block.get("annotations") or []:
                        if ann.get("type") in ("url_citation", "citation"):
                            add(ann.get("url") or "", ann.get("title") or "")

            if "web_search" in item_type:
                action = item.get("action") or {}
                sources = item.get("sources") or action.get("sources") or []
                for src in sources:
                    if isinstance(src, dict):
                        add(src.get("url") or "", src.get("title") or "")

        return citations


def get_web_search_client() -> WebSearchClient:
    return WebSearchClient()
