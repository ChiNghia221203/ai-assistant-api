from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from infra.urls import host_in_domains

logger = logging.getLogger(__name__)


@dataclass
class WebCitation:
    title: str = ""
    url: str = ""
    source: str = "gemini"


@dataclass
class WebContext:
    """Live web evidence handed to the answering LLM (never DB ratings)."""

    text: str = ""
    citations: list[WebCitation] = field(default_factory=list)
    # Google requires rendering Search Suggestions when using Search grounding.
    search_suggestion_html: str = ""
    ok: bool = True


_OFF_TOPIC_CITATION_RE = re.compile(
    r"("
    r"samsung|iphone|galaxy\s*s\d|pixel\s*\d|xiaomi|oppo|vivo|"
    r"điện\s*thoại|dien\s*thoai|smartphone|laptop|macbook|"
    r"bầu\s*cử|bau\s*cu|election|politics|chính\s*trị|chinh\s*tri|"
    r"crypto|bitcoin|chứng\s*khoán|chung\s*khoan"
    r")",
    re.IGNORECASE,
)


def _fold(text: str) -> str:
    raw = unicodedata.normalize("NFD", text or "")
    raw = "".join(c for c in raw if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", raw.casefold()).strip()


def _hotel_tokens(hotel_name: str) -> list[str]:
    folded = _fold(hotel_name)
    stop = {
        "khach",
        "san",
        "hotel",
        "the",
        "a",
        "an",
        "tp",
        "hcm",
        "saigon",
        "sai",
        "gon",
    }
    tokens = [
        t
        for t in re.split(r"[^\w]+", folded)
        if len(t) >= 3 and t not in stop
    ]
    return tokens


def _citation_mentions_hotel(citation: WebCitation, hotel_name: str) -> bool:
    blob = _fold(f"{citation.title} {citation.url}")
    tokens = _hotel_tokens(hotel_name)
    if not tokens:
        return False
    return any(t in blob for t in tokens)


def _text_mentions_hotel(text: str, hotel_name: str) -> bool:
    blob = _fold(text)
    tokens = _hotel_tokens(hotel_name)
    if not tokens:
        return bool(blob)
    return any(t in blob for t in tokens)


def filter_grounding_context(
    web: WebContext,
    *,
    hotel_name: str,
    allowed_domains: list[str] | tuple[str, ...] = (),
) -> WebContext | None:
    """Drop off-topic / unrelated grounding; keep ToS search_suggestion_html.

    Keep a citation when host is in allowed_domains (when set) or title/url
    mentions a hotel token. Drop phones/politics/etc. If nothing usable
    remains and text does not mention the hotel, return None.
    """
    if not web.ok or not (web.text or "").strip():
        return None

    text_ok = _text_mentions_hotel(web.text, hotel_name)
    if not web.citations:
        # No citations to judge — keep unless the body is clearly off-topic.
        if _OFF_TOPIC_CITATION_RE.search(web.text) and not text_ok:
            logger.info(
                "grounding_filter discarded off-topic body for hotel=%s", hotel_name
            )
            return None
        return WebContext(
            text=web.text,
            citations=[],
            search_suggestion_html=web.search_suggestion_html,
            ok=True,
        )

    domains = [d.strip().lower() for d in allowed_domains if d and d.strip()]
    kept: list[WebCitation] = []
    dropped = 0
    for c in web.citations:
        blob = f"{c.title} {c.url}"
        if _OFF_TOPIC_CITATION_RE.search(blob):
            dropped += 1
            continue
        if domains:
            in_allow = host_in_domains(c.url, domains)
            mentions = _citation_mentions_hotel(c, hotel_name)
            if in_allow or mentions:
                kept.append(c)
            else:
                dropped += 1
        else:
            kept.append(c)

    if dropped:
        logger.info(
            "grounding_filter hotel=%s kept=%d dropped=%d domains=%s",
            hotel_name,
            len(kept),
            dropped,
            domains or None,
        )

    if not kept and not text_ok:
        logger.info(
            "grounding_filter discarded all evidence for hotel=%s", hotel_name
        )
        return None

    return WebContext(
        text=web.text,
        citations=kept,
        search_suggestion_html=web.search_suggestion_html,
        ok=True,
    )
