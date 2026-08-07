
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from core.config import Settings, get_settings
from core.security import AuthUser
from domains.places.entity_resolve import EntityResolver
from domains.places.intent_extract import IntentContext, IntentExtractor
from domains.places.retrieval_gate import (
    MIXED_OFF_TOPIC_REFUSE,
    REFERENCE_NOTICE,
    GateResult,
    QuestionIntent,
    RetrievalDecision,
    ScopeKind,
    abstain_message,
    ambiguous_entity_message,
    ambiguous_name_message,
    classify_scope,
    decide_retrieval,
)
from domains.places.schemas import (
    PlaceEvidenceResponse,
    QuoteOut,
    RetrievalSource,
    ReviewChatRequest,
    ReviewChatResponse,
    WebCitationOut,
)
from domains.places.service import PlacesService, get_places_service
from infra.citations import WebCitation, WebContext, filter_grounding_context
from infra.database import get_supabase
from infra.gemini import (
    GeminiGroundingClient,
    GeminiGroundingError,
    GeminiQuotaExhausted,
    get_gemini_client,
)
from infra.llm import LlmClient, get_llm_client
from infra.web_search import WebSearchClient, get_web_search_client

logger = logging.getLogger(__name__)

# Out-of-catalog / hybrid-compare: prefer review OTAs when place has no tripadvisor_url.
_REFERENCE_REVIEW_DOMAINS: tuple[str, ...] = (
    "tripadvisor.com",
    "www.tripadvisor.com",
    "tripadvisor.com.vn",
    "traveloka.com",
    "www.traveloka.com",
)


def _tripadvisor_search_hint(hotel_name: str) -> str:
    """Soft prefer_url for TripAdvisor.vn search (preferred over Traveloka-only)."""
    from urllib.parse import quote_plus

    q = quote_plus(f"{(hotel_name or '').strip()} Ho Chi Minh City")
    return f"https://www.tripadvisor.com.vn/Search?q={q}"


def _traveloka_search_hint(hotel_name: str) -> str:
    """Soft prefer_url so Gemini/web_search can land on Traveloka hotel pages."""
    from urllib.parse import quote_plus

    q = quote_plus((hotel_name or "").strip())
    return f"https://www.traveloka.com/vi-vn/hotel/search?q={q}"


_SCORE_QUESTION_RE = re.compile(
    r"(điểm|score|so\s*sánh\s*(điểm|nguồn)|rating|sample|số\s*lượng\s*review|"
    r"bao\s*nhiêu\s*review|methodology|bảng\s*điểm)",
    re.IGNORECASE,
)

_COMPARE_QUESTION_RE = re.compile(
    r"(so\s*sánh|đối\s*chiếu|khác\s*nhau|hơn\s*kém|nên\s*chọn\s*cái\s*nào|"
    r"cái\s*nào\s*hơn|tốt\s*hơn|đáng\s*ở\s*hơn|nào\s*.{0,40}hơn|"
    r"chọn\s*cái\s*nào|cái\s*nào\s*ổn|vs\.?|versus)",
    re.IGNORECASE,
)


def _include_score_overview(message: str, context: ConversationContext | None) -> bool:
    """First turn (or explicit score ask) shows the overview table; follow-ups skip it."""
    if _SCORE_QUESTION_RE.search(message or ""):
        return True
    if context and (context.history or context.summary):
        return False
    return True


def _explicit_compare_ask(message: str) -> bool:
    return bool(_COMPARE_QUESTION_RE.search(message or ""))


def _wants_compare(message: str, *, area_recommend: bool, hotel_count: int) -> bool:
    """True only for an explicit compare ask with ≥2 hotels.

    area_recommend alone is area_list (not compare). hotel_count≥2 alone is NOT
    enough — that used to force 'so sánh' intros on routing mistakes.
    """
    del area_recommend  # precedence handled in _answer_mode
    if hotel_count < 2:
        return False
    return _explicit_compare_ask(message)


def _display_place_name(name: str) -> str:
    return re.sub(r"^\d+[\.\)]\s*", "", (name or "").strip()).strip()


def _fmt_score(value: Any, scale: Any = 5) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    try:
        s = int(scale or 5)
    except (TypeError, ValueError):
        s = 5
    return f"{v:.1f}/{s}"


def _fmt_day(value: Any) -> str:
    if value is None:
        return "—"
    text = str(value).strip()
    return text[:10] if text else "—"


def _source_label(source: str) -> str:
    key = (source or "").lower()
    return {
        "tripadvisor": "TripAdvisor",
        "chudu24": "Chudu24",
        "google": "Google",
    }.get(key, source or "—")


def build_score_overview_markdown(
    evidences: list[PlaceEvidenceResponse],
    *,
    multi_hotel: bool | None = None,
) -> str:
    """Deterministic score table from evidence — LLM must paste as-is.

    Uses sample.date_min/max + sample.size (ingested window), NOT site_n_total.
    """
    if not evidences:
        return ""
    show_hotel = multi_hotel if multi_hotel is not None else len(evidences) > 1
    if show_hotel:
        lines = [
            "| Khách sạn | Nguồn | Score | Từ ngày | Đến ngày | Số lượng review |",
            "|---|---|---|---|---|---|",
        ]
    else:
        lines = [
            "| Nguồn | Score | Từ ngày | Đến ngày | Số lượng review |",
            "|---|---|---|---|---|",
        ]

    for ev in evidences:
        hotel = _display_place_name(ev.place.name)
        for src in ev.sources:
            overall = (src.scores or {}).get("site_overall") or {}
            score = _fmt_score(overall.get("value"), overall.get("scale") or 5)
            dmin = _fmt_day(src.sample.date_min if src.sample else None)
            dmax = _fmt_day(src.sample.date_max if src.sample else None)
            n = src.sample.size if src.sample else 0
            n_cell = str(n) if n else "—"
            label = _source_label(src.source)
            if show_hotel:
                lines.append(
                    f"| {hotel} | {label} | {score} | {dmin} | {dmax} | {n_cell} |"
                )
            else:
                lines.append(
                    f"| {label} | {score} | {dmin} | {dmax} | {n_cell} |"
                )

    lines.append("")
    lines.append(
        "Lưu ý: mỗi nguồn chỉ lấy tối đa 100 review mới nhất đã thu thập "
        "(cột Số lượng review = sample đã ingest, không phải tổng review trên site)."
    )
    return "\n".join(lines)


def _answer_mode(
    message: str,
    *,
    hotel_count: int,
    area_recommend: bool,
    reference_count: int = 0,
) -> str:
    """single | compare | area_list — compare wins over area when both match."""
    total = hotel_count + reference_count
    if total >= 2 and _explicit_compare_ask(message):
        return "compare"
    if area_recommend:
        return "area_list"
    return "single"


def _place_quote_score(place_id: str, quotes: list[QuoteOut]) -> float:
    best = 0.0
    for q in quotes:
        if q.place_id != place_id:
            continue
        sim = q.similarity if q.similarity is not None else 0.0
        if sim > best:
            best = sim
    return best


def _guard_multi_hotel_anomaly(
    message: str,
    evidences: list[PlaceEvidenceResponse],
    quotes: list[QuoteOut],
    *,
    area_recommend: bool,
) -> tuple[list[PlaceEvidenceResponse], list[QuoteOut], bool]:
    """If ≥2 hotels without compare/area intent → routing anomaly.

    Prefer top-1 by quote similarity; if no usable scores → caller should abstain
    (returns empty evidences and abstain=True).
    """
    if len(evidences) < 2:
        return evidences, quotes, False
    if area_recommend or _explicit_compare_ask(message):
        return evidences, quotes, False

    scored: list[tuple[float, PlaceEvidenceResponse]] = []
    for ev in evidences:
        pid = str(ev.place.id)
        scored.append((_place_quote_score(pid, quotes), ev))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_ev = scored[0]
    if best_score <= 0:
        logger.warning(
            "routing_anomaly_multi_hotel: %d hotels, no quote scores → abstain",
            len(evidences),
        )
        return [], [], True

    keep_id = str(best_ev.place.id)
    logger.warning(
        "routing_anomaly_multi_hotel: collapsing %d hotels → top-1 %s (score=%.3f)",
        len(evidences),
        best_ev.place.name,
        best_score,
    )
    kept_quotes = [q for q in quotes if q.place_id == keep_id]
    return [best_ev], kept_quotes, False


def _response_policy(
    message: str,
    evidences: list[PlaceEvidenceResponse],
    context: ConversationContext | None,
    *,
    scope: ScopeKind | None = None,
    area_recommend: bool = False,
    available_places: list[dict[str, str]] | None = None,
    quotes: list[QuoteOut] | None = None,
    reference_hotels: list[str] | None = None,
    corpus_hotels: list[str] | None = None,
) -> dict[str, Any]:
    refs = list(reference_hotels or [])
    hotel_count = len(evidences)
    scope_kind = scope or classify_scope(message)
    mode = _answer_mode(
        message,
        hotel_count=hotel_count,
        area_recommend=area_recommend,
        reference_count=len(refs),
    )
    compare_mode = mode == "compare"
    include_overview = (
        _include_score_overview(message, context) or area_recommend
    ) and hotel_count > 0
    corpus = corpus_hotels or [
        _display_place_name(e.place.name) for e in evidences if e.place.name
    ]
    allowed = list(dict.fromkeys([*corpus, *refs]))
    policy: dict[str, Any] = {
        "include_score_overview": include_overview,
        "hotel_count": hotel_count,
        "compare_mode": compare_mode,
        "answer_mode": mode,
        "allowed_hotels": allowed,
        "corpus_hotels": corpus,
        "reference_hotels": refs,
        "scope": scope_kind.value,
        "refuse_off_topic": scope_kind == ScopeKind.MIXED,
        "hard_refuse": scope_kind == ScopeKind.HARD_OUT,
        "ambiguous_entity": scope_kind == ScopeKind.AMBIGUOUS_ENTITY,
        "quotes_available": bool(quotes),
        "area_recommend": area_recommend,
        # Set true in _llm_from_evidence when corpus evidence exists alongside web.
        "web_is_supplement": False,
    }
    if available_places is not None:
        policy["available_places"] = available_places
    if scope_kind == ScopeKind.MIXED:
        policy["off_topic_refuse_line"] = MIXED_OFF_TOPIC_REFUSE
    return policy


_OFF_TOPIC_SOLICIT_RE = re.compile(
    r"(?is)(?:^|\n+)[^\n]*(?:laptop|điện\s*thoại|dien\s*thoai|iphone|máy\s*tính|"
    r"may\s*tinh|mua\s*sắm|mua\s*sam)[^\n]*"
    r"(?:tư\s*vấn|tu\s*van|cho\s*biết|cho\s*biet|nhu\s*cầu|nhu\s*cau|"
    r"thoải\s*mái|thông tin về laptop)[^\n]*",
)


def enforce_mixed_scope_reply(reply: str) -> str:
    """Drop laptop/shopping solicitations; ensure fixed refuse footer when MIXED."""
    text = (reply or "").strip()
    parts: list[str] = []
    for para in re.split(r"\n{2,}", text):
        chunk = para.strip()
        if not chunk:
            continue
        if _OFF_TOPIC_SOLICIT_RE.search(chunk):
            continue
        if re.search(
            r"(?i)(thông tin về laptop|tư vấn.*(laptop|điện thoại)|"
            r"nhu cầu sử dụng.*(laptop|máy tính)|mua giúp.*(laptop|iphone))",
            chunk,
        ):
            continue
        parts.append(chunk)
    text = "\n\n".join(parts).strip()
    if MIXED_OFF_TOPIC_REFUSE not in text:
        text = f"{text}\n\n{MIXED_OFF_TOPIC_REFUSE}".strip() if text else MIXED_OFF_TOPIC_REFUSE
    return text


class ConversationAccessError(PermissionError):
    """The caller may not read or write this conversation."""


@dataclass
class ResolvedReply:
    reply: str
    retrieval_source: RetrievalSource = "rag"
    citations: list[WebCitation] = field(default_factory=list)
    search_suggestion_html: str = ""
    reference_only: bool = False


@dataclass(frozen=True)
class WebPolicy:
    """Which sources a grounding step may use, and how to label the answer."""

    allowed_domains: tuple[str, ...] = ()
    reference_only: bool = False


@dataclass
class ConversationContext:
    place_ids: list[UUID] = field(default_factory=list)
    summary: str = ""
    history: list[dict[str, str]] = field(default_factory=list)

SYSTEM_PROMPT = """Bạn là trợ lý tổng hợp đánh giá khách sạn đa nguồn.
Giúp USER tự quyết định từ evidence — trả lời tiếng Việt, markdown đúng cấu trúc.

## Ngôn ngữ
- Toàn bộ câu trả lời bằng tiếng Việt. Paraphrase review EN → VI (không dán đoạn EN dài).
- Giữ tên nguồn TripAdvisor / Chudu24 / Traveloka. Tên KS: place.name, bỏ prefix số kiểu "36. ".
- User hỏi ngôn ngữ khác → trả lời cùng ngôn ngữ đó.

## Cấu trúc (đúng thứ tự; tuân response_policy)

### 1) Mở đầu + bảng điểm — chỉ khi include_score_overview=true
- answer_mode=single: "Dựa trên dữ liệu đã thu thập, dưới đây là thông tin về khách sạn {Tên} tại TP. Hồ Chí Minh:"
- answer_mode=compare: "… thông tin so sánh {Tên1} và {Tên2} …"
- answer_mode=area_list: "… thông tin các khách sạn trong khu vực …" (liệt kê tên trong allowed_hotels)
Sau đó ### Đánh giá từ các nguồn
NẾU có score_overview_markdown trong input: dán NGUYÊN văn bảng đó (không sửa số/ngày/cột).
KHÔNG tự bịa bảng từ site_n_total; KHÔNG để "—" nếu score_overview_markdown đã có ngày.
Nếu không có score_overview_markdown (hiếm): mới tự dựng từ evidence.sources[].sample
  (date_min/date_max/size) + scores.site_overall.value — cấm dùng n_total làm số lượng sample.
KS trong reference_hotels (không có evidence): KHÔNG để hàng trống "Không có dữ liệu".
  Ghi nguồn Web/Traveloka/TripAdvisor từ web_findings + rating nếu web nêu được;
  thiếu rating → "Tham khảo web (chưa có trong dữ liệu đã thu thập)".
Nếu có reference_hotels: thêm 1 dòng "(KS ngoài hệ thống lấy từ web — chỉ mang tính tham khảo.)"
include_score_overview=false → bỏ mở đầu + bỏ cả mục bảng điểm.

### 2) ### Trả lời ngắn — 1–3 câu đúng loại hỏi
- single + yes/no: được mở Có./Không. rồi giải thích ngắn.
- compare: nêu khác biệt cụ thể từng KS; cấm "hài lòng với cả hai" chung chung.
- hybrid_compare / reference_hotels: PHẢI dùng web_findings cho KS ngoài corpus;
  cấm kết luận "không có dữ liệu" cho KS đó nếu web_findings đã có nội dung.
- area_list: gợi ý ngắn theo evidence; không bắt buộc chọn một KS.

### 3) ### Điểm nổi bật — 3–5 bullet VI; cân bằng nguồn khi đủ quotes
`- {ý} ({Nguồn}) [Xem chi tiết]({quotes[].review_url})`
Chỉ chữ "Xem chi tiết" là link; không URL trần; không review_url → chỉ (Nguồn).

### 4) ### Điểm cần lưu ý — 0–3 bullet yếu; không có →
"Không thấy phản hồi tiêu cực rõ trong dữ liệu đã thu thập."

### 5) ### Nhận xét tiêu biểu — 2–4 mục paraphrase ≤1–2 câu, không ngày
`- "{paraphrase}" ({Nguồn}) [Xem chi tiết](url)`

### 6) ### Độ tin cậy — một dòng
"Tổng hợp từ X review đã thu thập (Y Chudu24 + Z TripAdvisor)."

### 7) ### Kết luận — 2–3 câu theo dữ liệu; không "nên/không nên chọn".

## Dữ liệu
- Chỉ dùng evidence / quotes / web_findings. Chỉ nhắc KS trong allowed_hotels.
- Tuân answer_mode, hotel_count, include_score_overview, quotes_available.
- web_is_supplement=true: ưu tiên evidence/quotes/score_overview (review đã thu thập);
  web_findings chỉ bổ sung thời sự / tiện ích thiếu trong corpus / KS reference.
  Conflict review vs website quảng cáo → nghiêng corpus; web ghi tham khảo.
- corpus_missing=true / reference_hotels: KS ngoài corpus — chỉ dùng web_findings,
  ghi rõ tham khảo; bảng Score ingest chỉ cho corpus_hotels.
- hybrid compare: corpus_hotels từ evidence/quotes; reference_hotels từ web; so sánh
  tiêu chí user hỏi, không bịa sample Chudu24 cho KS ngoài hệ thống.
- conversation_summary/history: ngữ cảnh; không lặp nguyên văn trả lời trước.
- web_findings + reference_only=true → NOTICE tham khảo (đã gắn sẵn hoặc mục riêng).

## CẤM
Không bịa số/URL; không lộ thuật ngữ nội bộ (RAG, quotes, JSON); không chấm 8.5/10 hòa nguồn; không "đáng ở / nên đặt phòng".
Wifi/tiện ích: chỉ khẳng định khi quotes/evidence có nhắc; thiếu → nói chưa thấy trong dữ liệu đã thu thập.

## refuse_off_topic=true (câu MIXED: KS + mua sắm/laptop/điện thoại…)
- CHỈ trả lời phần khách sạn / review.
- CẤM tư vấn mua laptop/điện thoại; CẤM hỏi lại "cho biết nhu cầu để tư vấn".
- Kết thúc bằng đúng nội dung response_policy.off_topic_refuse_line (không tự viết bản khác).
"""

SUMMARY_SYSTEM = """Bạn nén hội thoại cũ thành bộ nhớ ngắn cho trợ lý review khách sạn.
Gộp previous_summary với older_messages thành MỘT bản tóm tắt tiếng Việt <=150 từ.

Giữ lại: khách sạn / địa điểm đang bàn, tiêu chí user quan tâm (sạch sẽ, ồn, vị trí,
giá, gia đình...), ràng buộc user nêu (ngày đi, ngân sách, số người), kết luận đã chốt.
Bỏ: lời chào, câu dẫn, số liệu chi tiết, quote, URL.
Chỉ xuất bản tóm tắt, không thêm tiêu đề hay lời bình.
"""

GROUNDING_SYSTEM = """Bạn chỉ tìm thông tin về đúng khách sạn được nêu (amenities,
đánh giá, vị trí, chính sách, giá/KM nếu hỏi).
Ưu tiên TripAdvisor và Traveloka — trang chi tiết / điểm rating khách sạn.
Bỏ qua kết quả không liên quan: chính trị, điện thoại, celebrity, crypto, tin ngoài KS.
Trả lời ngắn bằng tiếng Việt. Chỉ nêu fact có thể kiểm chứng; kèm URL nếu có.
Không bịa điểm rating hay số lượng review. Không kết luận "nên ở / không nên ở"
từ nội dung quảng cáo website.
"""


class ReviewChatService:
    def __init__(
        self,
        places: PlacesService | None = None,
        llm: LlmClient | None = None,
        gemini: GeminiGroundingClient | None = None,
        web_search: WebSearchClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.places = places or get_places_service()
        self.llm = llm or get_llm_client()
        self.gemini = gemini or get_gemini_client()
        self.web_search = web_search or get_web_search_client()
        self.settings = settings or get_settings()
        self.entities = EntityResolver()
        self.intent_extractor = IntentExtractor(self.settings)

    async def chat(
        self, payload: ReviewChatRequest, user: AuthUser | None = None
    ) -> ReviewChatResponse:
        seed_ids: list[UUID] = []
        seen: set[str] = set()
        for pid in list(payload.place_ids) + (
            [payload.place_id] if payload.place_id else []
        ):
            key = str(pid)
            if key not in seen:
                seen.add(key)
                seed_ids.append(pid)

        context = ConversationContext()
        if payload.conversation_id:
            if user is None:
                raise ConversationAccessError(
                    "Sign in to continue an existing conversation"
                )
            context = await self._load_context(payload.conversation_id, user.id)

        # Cheap hard-out BEFORE intent LLM / catalog bind (cost + avoid
        # sending sensitive content to extract model).
        scope = classify_scope(payload.message)
        if scope == ScopeKind.HARD_OUT:
            logger.info(
                "chat_decision hard_out=true skipped_intent=true message_len=%d",
                len(payload.message or ""),
            )
            return await self._abstain_response(
                payload,
                user,
                reply=abstain_message(RetrievalDecision.ABSTAIN_OUT_OF_SCOPE),
                place_ids=list(seed_ids) or list(context.place_ids),
            )

        catalog = self.places.list_places()
        available_places = [
            {
                "name": p.name,
                "address": p.address or "",
                "id": str(p.id),
            }
            for p in catalog
        ]
        known_names = self._known_hotel_names(
            catalog, seed_ids=seed_ids, conversation_place_ids=context.place_ids
        )
        intent = await self.intent_extractor.extract(
            payload.message,
            context=IntentContext(
                summary=context.summary or "",  
                history=tuple(context.history or ()),
                known_hotel_names=tuple(known_names),
            ),
        )
        entity = self.entities.resolve(
            payload.message,
            catalog,
            seed_place_ids=seed_ids,
            conversation_place_ids=context.place_ids,
            intent=intent,
        )
        unique_ids = entity.place_ids
        area_recommend = entity.area_recommend
        logger.info(
            "chat_decision entity_source=%s place_ids=%d refs=%s ambiguous=%s "
            "intent_compare=%s intent_hotels=%s",
            entity.source,
            len(entity.place_ids),
            entity.reference_hotels,
            entity.is_ambiguous,
            intent.wants_compare,
            intent.hotel_mentions,
        )

        # Explicit compare with ≥1 hotel outside catalog → RAG hits + ground misses.
        if entity.needs_web_compare:
            return await self._chat_hybrid_compare(
                payload, user, entity, context, available_places, scope
            )

        # Same brand, multiple catalog branches → ask user (do not auto-pick).
        if entity.is_ambiguous and entity.source == "ambiguous_name":
            candidates = [
                {
                    "name": p.name,
                    "address": p.address or "",
                    "id": str(p.id),
                }
                for p in entity.named
            ]
            return await self._abstain_response(
                payload,
                user,
                reply=ambiguous_name_message(
                    candidates, hotel_label=entity.hotel_label
                ),
                place_ids=[],
            )

        # Split original rules:
        # A) Hotel named but not in catalog/RAG → LLM + web grounding (reference).
        # B) Area miss / other ambiguous → short catalog abstain (do NOT invent peers).
        # Never conflate (A) with (B): missing corpus ≠ routing-wrong answer.
        if entity.is_ambiguous and entity.source == "unresolved_hotel":
            hotel_query = (entity.hotel_label or "").strip()
            if not hotel_query:
                return await self._abstain_response(
                    payload,
                    user,
                    reply=ambiguous_entity_message(
                        available_places,
                        area_label=entity.area_label,
                        hotel_label=entity.hotel_label,
                    ),
                    place_ids=[],
                )
            gate = GateResult(
                decision=RetrievalDecision.NEED_GROUNDING_REFERENCE,
                intent=QuestionIntent.EXPERIENCE,
                reason="hotel_not_in_catalog_ground_web",
                scope=ScopeKind.IN_SCOPE,
            )
            resolved = await self._resolve_reply(
                payload.message,
                [],
                [],
                gate,
                context,
                area_recommend=False,
                available_places=None,
                hotel_query=hotel_query,
            )
            conversation_id = payload.conversation_id
            if user is not None:
                conversation_id = self._persist(
                    user_id=user.id,
                    conversation_id=conversation_id,
                    message=payload.message,
                    reply=resolved.reply,
                    place_ids=[],
                    evidences=[],
                    quotes=[],
                )
            else:
                conversation_id = None
                logger.info("Chat without a token: conversation not persisted")
            return ReviewChatResponse(
                reply=resolved.reply,
                evidence=[],
                quotes=[],
                conversation_id=conversation_id,
                mock=self.settings.mock_llm or not self.settings.openai_api_key,
                retrieval_source=resolved.retrieval_source,
                web_citations=[
                    WebCitationOut(title=c.title, url=c.url, source=c.source)
                    for c in resolved.citations
                ],
                search_suggestion_html=resolved.search_suggestion_html,
                reference_only=resolved.reference_only,
            )

        if entity.is_ambiguous:
            return await self._abstain_response(
                payload,
                user,
                reply=ambiguous_entity_message(
                    available_places,
                    area_label=entity.area_label,
                    hotel_label=entity.hotel_label,
                ),
                place_ids=unique_ids,
            )

        evidences: list[PlaceEvidenceResponse] = []
        all_quotes: list[QuoteOut] = []

        if unique_ids:
            quotes = await self.places.search_quotes(
                payload.message,
                place_ids=unique_ids,
                top_k=payload.top_k,
            )
            all_quotes = quotes
            for pid in unique_ids:
                place_quotes = [q for q in quotes if q.place_id == str(pid)]
                evidences.append(
                    self.places.get_evidence(pid, quotes=place_quotes)
                )
        elif entity.allow_global_rag:
            quotes = await self.places.search_quotes(
                payload.message, top_k=payload.top_k
            )
            all_quotes = quotes
            place_from_quotes: list[str] = []
            for q in quotes:
                if q.place_id and q.place_id not in place_from_quotes:
                    place_from_quotes.append(q.place_id)
            for pid in place_from_quotes[:3]:
                pq = [q for q in quotes if q.place_id == pid]
                evidences.append(self.places.get_evidence(pid, quotes=pq))
        else:
            # Constrained ask without bindable places — refuse rather than invent.
            return await self._abstain_response(
                payload,
                user,
                reply=ambiguous_entity_message(
                    available_places,
                    area_label=entity.area_label,
                    hotel_label=entity.hotel_label,
                ),
                place_ids=[],
            )

        evidences, all_quotes, anomaly_abstain = _guard_multi_hotel_anomaly(
            payload.message,
            evidences,
            all_quotes,
            area_recommend=area_recommend,
        )
        if anomaly_abstain:
            return await self._abstain_response(
                payload,
                user,
                reply=ambiguous_entity_message(
                    available_places,
                    area_label=entity.area_label,
                    hotel_label=entity.hotel_label,
                ),
                place_ids=[],
            )
        if evidences:
            unique_ids = [e.place.id for e in evidences]

        gate = decide_retrieval(
            payload.message, all_quotes, evidences, settings=self.settings
        )
        # Preserve MIXED (and any future scopes) from message classify when gate
        # still says in_scope for retrieval purposes.
        if scope == ScopeKind.MIXED:
            gate = GateResult(
                decision=gate.decision,
                intent=gate.intent,
                reason=gate.reason,
                scope=ScopeKind.MIXED,
            )

        resolved = await self._resolve_reply(
            payload.message,
            evidences,
            all_quotes,
            gate,
            context,
            area_recommend=area_recommend,
            available_places=available_places,
        )

        conversation_id = payload.conversation_id
        if user is not None:
            conversation_id = self._persist(
                user_id=user.id,
                conversation_id=conversation_id,
                message=payload.message,
                reply=resolved.reply,
                place_ids=unique_ids,
                evidences=evidences,
                quotes=all_quotes,
            )
        else:
            # Without an owner there is nothing to write, so the caller must not
            # be handed an id it can keep appending to. FE signs in anonymously.
            conversation_id = None
            logger.info("Chat without a token: conversation not persisted")

        return ReviewChatResponse(
            reply=resolved.reply,
            evidence=evidences,
            quotes=all_quotes,
            conversation_id=conversation_id,
            mock=self.settings.mock_llm or not self.settings.openai_api_key,
            retrieval_source=resolved.retrieval_source,
            web_citations=[
                WebCitationOut(title=c.title, url=c.url, source=c.source)
                for c in resolved.citations
            ],
            search_suggestion_html=resolved.search_suggestion_html,
            reference_only=resolved.reference_only,
        )

    async def _chat_hybrid_compare(
        self,
        payload: ReviewChatRequest,
        user: AuthUser | None,
        entity,
        context: ConversationContext,
        available_places: list[dict[str, str]],
        scope: ScopeKind,
    ) -> ReviewChatResponse:
        """Compare with limit=2 when at least one side is outside the corpus.

        - corpus place_ids → RAG evidence/quotes
        - reference_hotels → Gemini/web (TripAdvisor preferred)
        Never fills missing sides from unrelated catalog peers.
        """
        del available_places  # not shown on hybrid path
        evidences: list[PlaceEvidenceResponse] = []
        all_quotes: list[QuoteOut] = []
        unique_ids = list(entity.place_ids)

        if unique_ids:
            quotes = await self.places.search_quotes(
                payload.message,
                place_ids=unique_ids,
                top_k=payload.top_k,
            )
            all_quotes = quotes
            for pid in unique_ids:
                place_quotes = [q for q in quotes if q.place_id == str(pid)]
                evidences.append(
                    self.places.get_evidence(pid, quotes=place_quotes)
                )

        gate = GateResult(
            decision=RetrievalDecision.NEED_GROUNDING_REFERENCE,
            intent=QuestionIntent.EXPERIENCE,
            reason=f"compare_{entity.source}",
            scope=scope if scope != ScopeKind.HARD_OUT else ScopeKind.IN_SCOPE,
        )
        resolved = await self._resolve_reply(
            payload.message,
            evidences,
            all_quotes,
            gate,
            context,
            area_recommend=False,
            available_places=None,
            reference_hotels=list(entity.reference_hotels),
            corpus_hotels=[
                _display_place_name(p.name) for p in entity.named
            ],
        )
        return await self._pack_chat_response(
            payload,
            user,
            resolved,
            evidences=evidences,
            quotes=all_quotes,
            place_ids=unique_ids,
        )

    async def _pack_chat_response(
        self,
        payload: ReviewChatRequest,
        user: AuthUser | None,
        resolved: ResolvedReply,
        *,
        evidences: list[PlaceEvidenceResponse],
        quotes: list[QuoteOut],
        place_ids: list[UUID],
    ) -> ReviewChatResponse:
        conversation_id = payload.conversation_id
        if user is not None:
            conversation_id = self._persist(
                user_id=user.id,
                conversation_id=conversation_id,
                message=payload.message,
                reply=resolved.reply,
                place_ids=place_ids,
                evidences=evidences,
                quotes=quotes,
            )
        else:
            conversation_id = None
            logger.info("Chat without a token: conversation not persisted")
        return ReviewChatResponse(
            reply=resolved.reply,
            evidence=evidences,
            quotes=quotes,
            conversation_id=conversation_id,
            mock=self.settings.mock_llm or not self.settings.openai_api_key,
            retrieval_source=resolved.retrieval_source,
            web_citations=[
                WebCitationOut(title=c.title, url=c.url, source=c.source)
                for c in resolved.citations
            ],
            search_suggestion_html=resolved.search_suggestion_html,
            reference_only=resolved.reference_only,
        )

    async def _abstain_response(
        self,
        payload: ReviewChatRequest,
        user: AuthUser | None,
        *,
        reply: str,
        place_ids: list[UUID],
    ) -> ReviewChatResponse:
        conversation_id = payload.conversation_id
        if user is not None:
            conversation_id = self._persist(
                user_id=user.id,
                conversation_id=conversation_id,
                message=payload.message,
                reply=reply,
                place_ids=place_ids,
                evidences=[],
                quotes=[],
            )
        else:
            conversation_id = None
        return ReviewChatResponse(
            reply=reply,
            evidence=[],
            quotes=[],
            conversation_id=conversation_id,
            mock=self.settings.mock_llm or not self.settings.openai_api_key,
            retrieval_source="abstain",
        )

    async def _resolve_reply(
        self,
        message: str,
        evidences: list[PlaceEvidenceResponse],
        quotes: list[QuoteOut],
        gate: GateResult,
        context: ConversationContext | None = None,
        *,
        area_recommend: bool = False,
        available_places: list[dict[str, str]] | None = None,
        hotel_query: str | None = None,
        reference_hotels: list[str] | None = None,
        corpus_hotels: list[str] | None = None,
    ) -> ResolvedReply:
        if gate.decision in (
            RetrievalDecision.ABSTAIN_BEYOND_SAMPLE,
            RetrievalDecision.ABSTAIN_NO_DATA,
            RetrievalDecision.ABSTAIN_OUT_OF_SCOPE,
            RetrievalDecision.ABSTAIN_AMBIGUOUS_ENTITY,
        ):
            return ResolvedReply(abstain_message(gate.decision), "abstain")

        if gate.decision == RetrievalDecision.RAG_ONLY:
            reply = await self._llm_from_evidence(
                message,
                evidences,
                quotes,
                None,
                context=context,
                scope=gate.scope,
                area_recommend=area_recommend,
                available_places=available_places,
                reference_hotels=reference_hotels,
                corpus_hotels=corpus_hotels,
            )
            return ResolvedReply(
                ReviewChatService._finalize_reply(reply, gate.scope), "rag"
            )

        policy = self._web_policy(gate.decision)
        web = await self._fetch_web_context(
            message,
            evidences,
            policy,
            hotel_query=hotel_query,
            reference_hotels=reference_hotels,
        )
        if web is None:
            if evidences or quotes:
                reply = await self._llm_from_evidence(
                    message,
                    evidences,
                    quotes,
                    None,
                    context=context,
                    scope=gate.scope,
                    area_recommend=area_recommend,
                    available_places=available_places,
                    reference_hotels=reference_hotels,
                    corpus_hotels=corpus_hotels,
                )
                return ResolvedReply(
                    ReviewChatService._finalize_reply(reply, gate.scope), "rag"
                )
            label = hotel_query or (
                reference_hotels[0] if reference_hotels else None
            )
            if label:
                return ResolvedReply(
                    ambiguous_entity_message(None, hotel_label=label),
                    "abstain",
                )
            return ResolvedReply(
                abstain_message(RetrievalDecision.ABSTAIN_NO_DATA), "abstain"
            )

        grounded, source = web
        reply = await self._llm_from_evidence(
            message,
            evidences,
            quotes,
            grounded,
            policy=policy,
            context=context,
            scope=gate.scope,
            area_recommend=area_recommend,
            available_places=available_places,
            hotel_query=hotel_query,
            reference_hotels=reference_hotels,
            corpus_hotels=corpus_hotels,
        )
        if policy.reference_only:
            reply = f"{REFERENCE_NOTICE}\n\n{reply}"
        reply = ReviewChatService._finalize_reply(reply, gate.scope)
        return ResolvedReply(
            reply,
            source,
            citations=grounded.citations,
            search_suggestion_html=grounded.search_suggestion_html,
            reference_only=policy.reference_only,
        )

    @staticmethod
    def _finalize_reply(reply: str, scope: ScopeKind | None) -> str:
        if scope == ScopeKind.MIXED:
            return enforce_mixed_scope_reply(reply)
        return reply

    def _web_policy(self, decision: RetrievalDecision) -> WebPolicy:
        if decision == RetrievalDecision.NEED_GROUNDING_REFERENCE:
            return WebPolicy(allowed_domains=(), reference_only=True)
        return WebPolicy(
            allowed_domains=tuple(self.settings.allowed_search_domains),
            reference_only=False,
        )

    async def _ground_one_hotel(
        self,
        message: str,
        hotel_name: str,
        policy: WebPolicy,
        *,
        ta_url: str | None = None,
        prefer_review_sites: bool = False,
    ) -> tuple[WebContext, RetrievalSource] | None:
        domains = list(policy.allowed_domains)
        if domains:
            search_domains = domains
        elif prefer_review_sites:
            search_domains = list(_REFERENCE_REVIEW_DOMAINS)
        else:
            search_domains = []

        prefer_url = ta_url
        if prefer_review_sites and not prefer_url:
            # Prefer TripAdvisor search landing; Traveloka remains in preferred_domains.
            prefer_url = _tripadvisor_search_hint(hotel_name)

        user_blob = (
            f"Khách sạn: {hotel_name} (TP.HCM / TP. Hồ Chí Minh)\n"
            f"Câu hỏi: {message}\n"
            "Chỉ tìm thông tin về khách sạn này. "
            "Ưu tiên TripAdvisor (tripadvisor.com.vn) hoặc Traveloka "
            "(đánh giá / điểm rating). "
            f"Gợi ý Traveloka: {_traveloka_search_hint(hotel_name)}"
        )
        if self.gemini.available:
            try:
                result = await self.gemini.grounded_complete(
                    GROUNDING_SYSTEM,
                    user_blob,
                    prefer_url=prefer_url if search_domains or prefer_url else None,
                    preferred_domains=search_domains,
                )
                filtered = filter_grounding_context(
                    result,
                    hotel_name=hotel_name,
                    allowed_domains=search_domains,
                )
                if filtered is not None and filtered.ok and filtered.text:
                    return filtered, "rag+gemini"
            except GeminiQuotaExhausted as exc:
                logger.warning(
                    "Gemini quota exhausted, falling back to web_search: %s", exc
                )
            except GeminiGroundingError as exc:
                logger.warning(
                    "Gemini grounding failed, falling back to web_search: %s", exc
                )

        search = await self.web_search.search(
            hotel_name=hotel_name,
            question=message,
            source_url=prefer_url or ta_url,
            allowed_domains=search_domains,
        )
        if search.ok and search.text:
            filtered = filter_grounding_context(
                WebContext(text=search.text, citations=search.citations, ok=True),
                hotel_name=hotel_name,
                allowed_domains=search_domains,
            )
            if filtered is not None:
                return filtered, "rag+web_search"
        return None

    async def _fetch_web_context(
        self,
        message: str,
        evidences: list[PlaceEvidenceResponse],
        policy: WebPolicy,
        *,
        hotel_query: str | None = None,
        reference_hotels: list[str] | None = None,
    ) -> tuple[WebContext, RetrievalSource] | None:
        labels = [x.strip() for x in (reference_hotels or []) if x and x.strip()]
        if not labels and hotel_query and hotel_query.strip():
            labels = [hotel_query.strip()]

        # Compare / out-of-catalog: ground only the reference labels (not corpus).
        if labels:
            chunks: list[str] = []
            citations: list[WebCitation] = []
            suggestion_html = ""
            source: RetrievalSource = "rag+gemini"
            for name in labels:
                part = await self._ground_one_hotel(
                    message,
                    name,
                    policy,
                    prefer_review_sites=True,
                )
                if not part:
                    logger.warning("No web context for reference hotel %s", name)
                    continue
                ctx, src = part
                chunks.append(f"### {name}\n{ctx.text}")
                citations.extend(ctx.citations)
                if ctx.search_suggestion_html and not suggestion_html:
                    suggestion_html = ctx.search_suggestion_html
                source = src
            if not chunks:
                return None
            merged = WebContext(
                text="\n\n".join(chunks),
                citations=citations,
                search_suggestion_html=suggestion_html,
                ok=True,
            )
            return merged, source

        if not evidences:
            logger.info("Skipping web grounding: no place resolved from request")
            return None

        hotel_name = evidences[0].place.name
        ta_url = evidences[0].place.tripadvisor_url
        return await self._ground_one_hotel(
            message,
            hotel_name,
            policy,
            ta_url=ta_url,
            prefer_review_sites=False,
        )

    async def _llm_from_evidence(
        self,
        message: str,
        evidences: list[PlaceEvidenceResponse],
        quotes: list[QuoteOut],
        web: WebContext | None,
        policy: WebPolicy | None = None,
        context: ConversationContext | None = None,
        scope: ScopeKind | None = None,
        area_recommend: bool = False,
        available_places: list[dict[str, str]] | None = None,
        hotel_query: str | None = None,
        reference_hotels: list[str] | None = None,
        corpus_hotels: list[str] | None = None,
    ) -> str:
        user_payload: dict[str, Any] = {}
        if context and context.summary:
            user_payload["conversation_summary"] = context.summary
        if context and context.history:
            user_payload["history"] = context.history
        refs = list(reference_hotels or [])
        if hotel_query and hotel_query not in refs:
            refs = [hotel_query, *refs]
        user_payload["response_policy"] = _response_policy(
            message,
            evidences,
            context,
            scope=scope,
            area_recommend=area_recommend,
            available_places=available_places
            if area_recommend or scope == ScopeKind.AMBIGUOUS_ENTITY
            else None,
            quotes=quotes,
            reference_hotels=refs,
            corpus_hotels=corpus_hotels,
        )
        if refs and not evidences:
            user_payload["response_policy"]["corpus_missing"] = True
        if refs and evidences:
            user_payload["response_policy"]["hybrid_compare"] = True
        if web is not None and evidences:
            user_payload["response_policy"]["web_is_supplement"] = True
        user_payload["evidence"] = [e.model_dump(mode="json") for e in evidences]
        user_payload["quotes"] = [q.model_dump(mode="json") for q in quotes]
        policy_meta = user_payload["response_policy"]
        if policy_meta.get("include_score_overview") and evidences:
            overview = build_score_overview_markdown(
                evidences,
                multi_hotel=len(evidences) > 1 or bool(refs),
            )
            if overview:
                user_payload["score_overview_markdown"] = overview
        if web is not None:
            user_payload["web_findings"] = {
                "text": web.text,
                "reference_only": bool(policy and policy.reference_only),
                "citations": [
                    {"title": c.title, "url": c.url, "source": c.source}
                    for c in web.citations
                ],
            }
        user_payload["question"] = message
        policy_meta = user_payload.get("response_policy") or {}
        logger.info(
            "answer_llm decision=%s",
            json.dumps(
                {
                    "model": self.settings.openai_model,
                    "answer_mode": policy_meta.get("answer_mode"),
                    "hotel_count": policy_meta.get("hotel_count"),
                    "evidence_count": len(evidences),
                    "quote_count": len(quotes),
                    "has_web": web is not None,
                    "reference_hotels": policy_meta.get("reference_hotels"),
                    "corpus_hotels": policy_meta.get("corpus_hotels"),
                    "scope": policy_meta.get("scope"),
                },
                ensure_ascii=False,
            ),
        )
        return await self.llm.complete(
            SYSTEM_PROMPT,
            json.dumps(user_payload, ensure_ascii=False, indent=2),
        )

    def _known_hotel_names(
        self,
        catalog: list,
        *,
        seed_ids: list[UUID],
        conversation_place_ids: list[UUID],
    ) -> list[str]:
        """Resolve conversation/seed place ids to names for intent context."""
        by_id = {str(p.id): p.name for p in catalog if getattr(p, "id", None)}
        names: list[str] = []
        seen: set[str] = set()
        for pid in list(seed_ids) + list(conversation_place_ids):
            name = by_id.get(str(pid))
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    async def _load_context(
        self, conversation_id: UUID, user_id: UUID
    ) -> ConversationContext:
        """Conversation memory: stored summary + the last N turns verbatim.

        Ownership is checked here, before any of it can reach the prompt. Read
        errors are not swallowed: failing closed beats leaking someone's history.
        """
        ctx = ConversationContext()
        sb = get_supabase()
        row = self._fetch_conversation_row(conversation_id)
        if not row:
            raise ConversationAccessError("Conversation not found")
        if str(row.get("user_id")) != str(user_id):
            raise ConversationAccessError("Conversation belongs to another user")

        ctx.place_ids = self._parse_place_ids(row.get("place_ids"))
        ctx.summary = row.get("summary") or ""
        summarized_through = row.get("summarized_through")

        limit = max(self.settings.chat_history_limit, 0)
        if not limit:
            return ctx

        try:
            recent = (
                sb.table("messages")
                .select("role, content, created_at")
                .eq("conversation_id", str(conversation_id))
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
                .data
                or []
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load conversation history")
            return ctx

        ctx.history = [
            {"role": r.get("role") or "user", "content": r.get("content") or ""}
            for r in reversed(recent)
            if r.get("content")
        ]

        # Only once the window is full can there be older turns to fold in.
        if len(recent) >= limit and "summary" in row:
            ctx.summary = await self._refresh_summary(
                conversation_id,
                previous_summary=ctx.summary,
                window_start=recent[-1].get("created_at"),
                summarized_through=summarized_through,
            )
        return ctx

    def _fetch_conversation_row(self, conversation_id: UUID) -> dict[str, Any] | None:
        """Load conversation; tolerate DBs that have not applied summary migration."""
        sb = get_supabase()
        try:
            rows = (
                sb.table("conversations")
                .select("user_id, place_ids, summary, summarized_through")
                .eq("id", str(conversation_id))
                .limit(1)
                .execute()
                .data
                or []
            )
            return rows[0] if rows else None
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "summary" not in msg and "summarized_through" not in msg:
                raise
            logger.warning(
                "conversations.summary missing — run supabase/migrations/"
                "002_conversation_summary.sql (%s)",
                exc,
            )
            rows = (
                sb.table("conversations")
                .select("user_id, place_ids")
                .eq("id", str(conversation_id))
                .limit(1)
                .execute()
                .data
                or []
            )
            return rows[0] if rows else None

    def _parse_place_ids(self, raw_ids: Any) -> list[UUID]:
        out: list[UUID] = []
        for raw in raw_ids or []:
            try:
                out.append(UUID(str(raw)))
            except ValueError:
                logger.warning("Skipping invalid place_id in conversation: %r", raw)
        return out

    async def _refresh_summary(
        self,
        conversation_id: UUID,
        *,
        previous_summary: str,
        window_start: Any,
        summarized_through: Any,
    ) -> str:
        """Fold turns older than the verbatim window into the stored summary."""
        if not window_start:
            return previous_summary

        sb = get_supabase()
        try:
            query = (
                sb.table("messages")
                .select("role, content, created_at")
                .eq("conversation_id", str(conversation_id))
                .lt("created_at", window_start)
            )
            if summarized_through:
                query = query.gt("created_at", summarized_through)
            older = (
                query.order("created_at")
                .limit(max(self.settings.chat_summary_batch, 1))
                .execute()
                .data
                or []
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load messages for summarization")
            return previous_summary

        if not older:
            return previous_summary

        try:
            summary = await self.llm.complete(
                SUMMARY_SYSTEM,
                json.dumps(
                    {
                        "previous_summary": previous_summary,
                        "older_messages": [
                            {
                                "role": m.get("role") or "user",
                                "content": m.get("content") or "",
                            }
                            for m in older
                            if m.get("content")
                        ],
                    },
                    ensure_ascii=False,
                ),
            )
        except Exception:  # noqa: BLE001
            logger.exception("Summarization failed; keeping previous summary")
            return previous_summary

        summary = (summary or "").strip()
        if not summary:
            return previous_summary

        try:
            sb.table("conversations").update(
                {
                    "summary": summary,
                    "summarized_through": older[-1].get("created_at"),
                }
            ).eq("id", str(conversation_id)).execute()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to store conversation summary")
        return summary

    def _persist(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID | None,
        message: str,
        reply: str,
        place_ids: list[UUID],
        evidences: list[PlaceEvidenceResponse],
        quotes: list[QuoteOut],
    ) -> UUID:
        sb = get_supabase()
        if conversation_id is None:
            title = message[:80] or "Hotel review"
            created = (
                sb.table("conversations")
                .insert(
                    {
                        "user_id": str(user_id),
                        "title": title,
                        "place_ids": [str(p) for p in place_ids],
                    }
                )
                .execute()
                .data[0]
            )
            conversation_id = UUID(created["id"])
        else:
            # `updated_at` has no DB trigger, so bump it here: the conversation
            # list is ordered by it and would otherwise never reflect activity.
            # Owner is pinned in the filter as a second line of defence.
            sb.table("conversations").update(
                {
                    "place_ids": [str(p) for p in place_ids],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            ).eq("id", str(conversation_id)).eq("user_id", str(user_id)).execute()

        sources = [q.model_dump(mode="json") for q in quotes]
        evidence_payload: Any = [e.model_dump(mode="json") for e in evidences]
        sb.table("messages").insert(
            [
                {
                    "conversation_id": str(conversation_id),
                    "role": "user",
                    "content": message,
                    "sources": [],
                },
                {
                    "conversation_id": str(conversation_id),
                    "role": "assistant",
                    "content": reply,
                    "sources": sources,
                    "evidence": evidence_payload,
                },
            ]
        ).execute()
        return conversation_id


def get_review_chat_service() -> ReviewChatService:
    return ReviewChatService()
