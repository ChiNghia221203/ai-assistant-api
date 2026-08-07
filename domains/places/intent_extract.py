"""LLM structured intent extract (function calling) — replaces brittle regex NLU.

Pattern: cheap model returns {wants_compare, hotel_mentions, ...}; code still
binds catalog, RAG, and grounding. The answer LLM never self-searches.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM = """Bạn là bộ phân loại ý định cho trợ lý review khách sạn TP.HCM.
Chỉ gọi tool extract_review_intent đúng schema. Không viết câu trả lời cho user.
- wants_compare=true khi user muốn so sánh / chọn cái tốt hơn giữa ≥2 khách sạn
  (kể cả: tốt hơn, đáng ở hơn, cái nào ổn, nên chọn cái nào, vs...).
- hotel_mentions: tên/thương hiệu KS user nhắc (giữ gần đúng cách viết, bỏ chữ
  khách sạn/KS thừa). Tối đa 4.
- Nếu user dùng đại từ / tham chiếu ("cái kia", "ks lúc nãy", "còn chỗ đó",
  "so với cái trước") và có conversation context: điền hotel_mentions từ
  tên KS đã nhắc trong summary/history. Không bịa KS ngoài c    ontext.
- area_mentions: quận/khu vực nếu có (vd Tân Bình, Quận 1).
- criteria: tiêu chí trải nghiệm nếu có (ồn, sạch, ăn sáng, gần sân bay...).
- wants_score_overview: true nếu hỏi bảng điểm / số lượng review / so sánh nguồn điểm.
"""

EXTRACT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "extract_review_intent",
        "description": (
            "Extract compare intent, hotel names, areas, and criteria "
            "from a hotel-review assistant user message."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "wants_compare": {
                    "type": "boolean",
                    "description": "True if user wants to compare or pick the better hotel.",
                },
                "hotel_mentions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Hotel names/brands mentioned (max 4).",
                },
                "area_mentions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "District/area mentions if any.",
                },
                "criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Experience criteria (noise, cleanliness, breakfast...).",
                },
                "wants_score_overview": {
                    "type": "boolean",
                    "description": "True if user asks for score table / review counts.",
                },
            },
            "required": [
                "wants_compare",
                "hotel_mentions",
                "area_mentions",
                "criteria",
                "wants_score_overview",
            ],
            "additionalProperties": False,
        },
    },
}

_HOTEL_PAIR_RE = re.compile(
    r"(?:khách\s*sạn|khach\s*san|\bks\b|\bhotel\b)\s+"
    r"(?P<a>[A-Za-zÀ-ỹ0-9][\wÀ-ỹ\.\-]*(?:\s+[A-Za-zÀ-ỹ0-9][\wÀ-ỹ\.\-]*){0,6})"
    r"\s+và\s+"
    r"(?:khách\s*sạn|khach\s*san|\bks\b|\bhotel\b)\s+"
    r"(?P<b>[A-Za-zÀ-ỹ0-9][\wÀ-ỹ\.\-]*(?:\s+[A-Za-zÀ-ỹ0-9][\wÀ-ỹ\.\-]*){0,6})",
    re.IGNORECASE,
)

_COMPARE_SOFT_RE = re.compile(
    r"(so\s*sánh|đối\s*chiếu|vs\.?|versus|hơn\s*kém|cái\s*nào\s*hơn|"
    r"tốt\s*hơn|đáng\s*ở\s*hơn|nào\s*.{0,40}hơn|chọn\s*cái\s*nào|"
    r"cái\s*nào\s*ổn)",
    re.IGNORECASE,
)

_PRONOUN_FOLLOWUP_RE = re.compile(
    r"(cái\s*kia|ks\s*lúc\s*nãy|lúc\s*nãy|còn\s*(chỗ|ks|khách)|"
    r"so\s*với\s*(cái|ks)|chỗ\s*đó|hotel\s*(đó|kia)|the\s*other\s*one)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class IntentExtract:
    wants_compare: bool = False
    hotel_mentions: list[str] = field(default_factory=list)
    area_mentions: list[str] = field(default_factory=list)
    criteria: list[str] = field(default_factory=list)
    wants_score_overview: bool = False
    source: str = "empty"  # rules | llm | empty


@dataclass(frozen=True, slots=True)
class IntentContext:
    """Optional conversation memory for pronoun / follow-up resolution."""

    summary: str = ""
    history: tuple[dict[str, str], ...] = ()
    known_hotel_names: tuple[str, ...] = ()


def _heuristic_extract(message: str) -> IntentExtract:
    """Fast offline / mock path for obvious pairs."""
    text = message or ""
    hotels: list[str] = []
    pair = _HOTEL_PAIR_RE.search(text)
    if pair:
        a = re.sub(r"\s+thì\s+.+$", "", pair.group("a").strip(), flags=re.I).strip()
        b = re.sub(r"\s+thì\s+.+$", "", pair.group("b").strip(), flags=re.I).strip()
        hotels = [a, b]
    wants = bool(_COMPARE_SOFT_RE.search(text)) or len(hotels) >= 2
    if len(hotels) >= 2:
        wants = True
    return IntentExtract(
        wants_compare=wants,
        hotel_mentions=hotels[:4],
        area_mentions=[],
        criteria=[],
        wants_score_overview=bool(
            re.search(r"(điểm|score|rating|số\s*lượng\s*review)", text, re.I)
        ),
        source="rules",
    )


def _normalize_extract(raw: dict[str, Any], *, source: str) -> IntentExtract:
    hotels = [
        str(h).strip()
        for h in (raw.get("hotel_mentions") or [])
        if str(h).strip()
    ]
    areas = [
        str(a).strip()
        for a in (raw.get("area_mentions") or [])
        if str(a).strip()
    ]
    criteria = [
        str(c).strip() for c in (raw.get("criteria") or []) if str(c).strip()
    ]
    return IntentExtract(
        wants_compare=bool(raw.get("wants_compare")),
        hotel_mentions=hotels[:4],
        area_mentions=areas[:4],
        criteria=criteria[:6],
        wants_score_overview=bool(raw.get("wants_score_overview")),
        source=source,
    )


def _format_context_blob(ctx: IntentContext | None) -> str:
    if ctx is None:
        return ""
    parts: list[str] = []
    if ctx.summary.strip():
        parts.append(f"Summary: {ctx.summary.strip()}")
    if ctx.known_hotel_names:
        parts.append(
            "Known hotels in this conversation: "
            + ", ".join(ctx.known_hotel_names[:6])
        )
    if ctx.history:
        lines = []
        for turn in ctx.history[-6:]:
            role = (turn.get("role") or "user").strip()
            content = (turn.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content[:400]}")
        if lines:
            parts.append("Recent turns:\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _needs_llm_for_context(message: str, heuristic: IntentExtract, ctx: IntentContext | None) -> bool:
    """Follow-ups with pronouns + known hotels should not skip LLM even if heuristic empty."""
    if ctx is None:
        return False
    if not (ctx.summary or ctx.history or ctx.known_hotel_names):
        return False
    if heuristic.hotel_mentions:
        return False
    return bool(_PRONOUN_FOLLOWUP_RE.search(message or ""))


class IntentExtractor:
    """One structured LLM call (function calling) per message when needed."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def extract(
        self,
        message: str,
        *,
        context: IntentContext | None = None,
    ) -> IntentExtract:
        heuristic = _heuristic_extract(message)
        # Strong rules hit (pair of "khách sạn X và khách sạn Y") → skip LLM.
        if (
            heuristic.wants_compare
            and len(heuristic.hotel_mentions) >= 2
            and _HOTEL_PAIR_RE.search(message or "")
            and not _needs_llm_for_context(message, heuristic, context)
        ):
            logger.info(
                "intent_extract decision=%s",
                json.dumps(
                    {"source": "rules", **{k: v for k, v in asdict(heuristic).items() if k != "source"}},
                    ensure_ascii=False,
                ),
            )
            return heuristic

        if self.settings.mock_llm or not self.settings.openai_api_key:
            # Mock: resolve pronouns from known_hotel_names when heuristic empty.
            if (
                not heuristic.hotel_mentions
                and context
                and context.known_hotel_names
                and _PRONOUN_FOLLOWUP_RE.search(message or "")
            ):
                names = list(context.known_hotel_names[:2])
                heuristic = IntentExtract(
                    wants_compare=bool(_COMPARE_SOFT_RE.search(message or ""))
                    or len(names) >= 2,
                    hotel_mentions=names,
                    area_mentions=[],
                    criteria=[],
                    wants_score_overview=heuristic.wants_score_overview,
                    source="rules",
                )
            logger.info(
                "intent_extract decision=%s",
                json.dumps(
                    {
                        "source": "mock_heuristic",
                        "model": None,
                        **{k: v for k, v in asdict(heuristic).items() if k != "source"},
                    },
                    ensure_ascii=False,
                ),
            )
            return heuristic

        try:
            raw = await self._openai_tool_extract(message, context=context)
            out = _normalize_extract(raw, source="llm")
            if len(out.hotel_mentions) < 2 and len(heuristic.hotel_mentions) >= 2:
                out.hotel_mentions = heuristic.hotel_mentions
            if not out.wants_compare and len(out.hotel_mentions) >= 2:
                out.wants_compare = True
            logger.info(
                "intent_extract decision=%s",
                json.dumps(
                    {
                        "source": "llm",
                        "model": self.settings.openai_extract_model
                        or self.settings.openai_model,
                        **{k: v for k, v in asdict(out).items() if k != "source"},
                    },
                    ensure_ascii=False,
                ),
            )
            return out
        except Exception:  # noqa: BLE001
            logger.exception("IntentExtract LLM failed; using heuristic")
            heuristic.source = "rules_fallback"
            logger.info(
                "intent_extract decision=%s",
                json.dumps(
                    {
                        "source": "rules_fallback",
                        **{k: v for k, v in asdict(heuristic).items() if k != "source"},
                    },
                    ensure_ascii=False,
                ),
            )
            return heuristic

    async def _openai_tool_extract(
        self,
        message: str,
        *,
        context: IntentContext | None = None,
    ) -> dict[str, Any]:
        model = self.settings.openai_extract_model or self.settings.openai_model
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        messages: list[dict[str, str]] = [
            {"role": "system", "content": EXTRACT_SYSTEM},
        ]
        blob = _format_context_blob(context)
        if blob:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Conversation context for resolving pronouns / follow-ups:\n"
                        + blob
                    ),
                }
            )
        messages.append({"role": "user", "content": message})
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": [EXTRACT_TOOL],
            "tool_choice": {
                "type": "function",
                "function": {"name": "extract_review_intent"},
            },
            "temperature": 0,
        }
        url = f"{self.settings.openai_base_url.rstrip('/')}/chat/completions"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        message_out = data["choices"][0]["message"]
        tool_calls = message_out.get("tool_calls") or []
        if not tool_calls:
            content = (message_out.get("content") or "").strip()
            if content.startswith("{"):
                return json.loads(content)
            raise ValueError("No tool_calls in extract response")

        args = tool_calls[0]["function"]["arguments"]
        if isinstance(args, str):
            return json.loads(args)
        return dict(args)
