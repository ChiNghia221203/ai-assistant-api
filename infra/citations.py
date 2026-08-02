

from __future__ import annotations

from dataclasses import dataclass, field


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
