
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
from domains.places.place_match import (
    is_area_recommend_intent,
    match_places_in_area,
    match_places_in_message,
)
from domains.places.retrieval_gate import (
    MIXED_SCOPE_NOTICE,
    REFERENCE_NOTICE,
    GateResult,
    RetrievalDecision,
    ScopeKind,
    abstain_message,
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
from infra.citations import WebCitation, WebContext
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

_SCORE_QUESTION_RE = re.compile(
    r"(điểm|score|so\s*sánh\s*(điểm|nguồn)|rating|sample|số\s*lượng\s*review|"
    r"bao\s*nhiêu\s*review|methodology|bảng\s*điểm)",
    re.IGNORECASE,
)

_COMPARE_QUESTION_RE = re.compile(
    r"(so\s*sánh|đối\s*chiếu|khác\s*nhau|hơn\s*kém|nên\s*chọn\s*cái\s*nào|"
    r"cái\s*nào\s*hơn|vs\.?|versus)",
    re.IGNORECASE,
)


def _include_score_overview(message: str, context: ConversationContext | None) -> bool:
    """First turn (or explicit score ask) shows the overview table; follow-ups skip it."""
    if _SCORE_QUESTION_RE.search(message or ""):
        return True
    if context and (context.history or context.summary):
        return False
    return True


def _is_compare_question(message: str, hotel_count: int) -> bool:
    if hotel_count >= 2:
        return True
    return bool(_COMPARE_QUESTION_RE.search(message or ""))


def _response_policy(
    message: str,
    evidences: list[PlaceEvidenceResponse],
    context: ConversationContext | None,
    *,
    scope: ScopeKind | None = None,
    area_recommend: bool = False,
    available_places: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    hotel_count = len(evidences)
    scope_kind = scope or classify_scope(message)
    compare = _is_compare_question(message, hotel_count) or (
        area_recommend and hotel_count >= 2
    )
    policy: dict[str, Any] = {
        "include_score_overview": _include_score_overview(message, context)
        or area_recommend,
        "hotel_count": hotel_count,
        "compare_mode": compare,
        "scope": scope_kind.value,
        "refuse_off_topic": scope_kind == ScopeKind.MIXED,
        "area_recommend": area_recommend,
    }
    if available_places is not None:
        policy["available_places"] = available_places
    return policy

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
Nhiệm vụ: giúp USER TỰ QUYẾT ĐỊNH dựa trên evidence — trả lời rõ, ngắn, đúng cấu trúc.

## Ngôn ngữ (BẮT BUỘC)
- Toàn bộ câu trả lời (mọi heading, bullet, paraphrase, kết luận) viết bằng tiếng Việt.
- Quotes TripAdvisor / nguồn khác thường là tiếng Anh: PHẢI diễn đạt lại ý sang tiếng Việt.
  Không dán nguyên đoạn review tiếng Anh vào UI.
- Được giữ tối đa một cụm ngắn trong ngoặc kép (EN hoặc VI) nếu cần nhấn mạnh; phần còn lại vẫn tiếng Việt.
- Tên nguồn giữ nguyên: TripAdvisor, Chudu24.
- Tên khách sạn: dùng place.name, BỎ số thứ tự / prefix kiểu "36. ", "12)" ở đầu tên.
- Nếu user hỏi bằng ngôn ngữ khác tiếng Việt: trả lời cùng ngôn ngữ đó.

## Cấu trúc BẮT BUỘC (Markdown, đúng thứ tự)

### 1) Mở đầu + bảng điểm (khi response_policy.include_score_overview = true)

Câu mở đầu (không dùng heading):
- 1 KS: "Dựa trên dữ liệu đã thu thập, dưới đây là thông tin về khách sạn {Tên} tại TP. Hồ Chí Minh:"
- ≥2 KS: "Dựa trên dữ liệu đã thu thập, dưới đây là thông tin so sánh {Tên1} và {Tên2} tại TP. Hồ Chí Minh:"

Sau đó heading: ### Đánh giá từ các nguồn

Bảng markdown — ĐÚNG các cột sau (một cột Score duy nhất, không lặp):
- 1 KS: | Nguồn | Score | Từ ngày | Đến ngày | Số lượng review |
- ≥2 KS (response_policy.hotel_count ≥ 2 hoặc compare_mode = true):
  | Khách sạn | Nguồn | Score | Từ ngày | Đến ngày | Số lượng review |

Map dữ liệu:
- Khách sạn ← place.name (đã bỏ số đầu tên)
- Nguồn ← sources[].source (Chudu24 / TripAdvisor; chỉ các nguồn có trong evidence)
- Score ← sources[].scores.site_overall.value dạng 4.7/5 (kèm scale nếu có);
  thiếu value → "Không có dữ liệu"
- Từ ngày / Đến ngày ← sample.date_min / date_max; thiếu → "Không có dữ liệu"
- Số lượng review ← sample.size (thiếu thì 0)

Mỗi nguồn một hàng. Không thêm cột Score thứ hai, không sample_mean, không địa chỉ,
không GPS, không URL, không mục "Tóm tắt thông tin khách sạn".
Không dùng emoji sao.

Ngay dưới bảng (giữ nguyên câu):
"Lưu ý: mỗi nguồn chỉ lấy tối đa 100 review mới nhất đã thu thập."

Khi include_score_overview = false: BỎ mở đầu kiểu trên + BỎ cả mục Đánh giá từ các nguồn
(không bảng, không nhắc lại Score/số lượng review).

### 2) Trả lời ngắn
Heading: ### Trả lời ngắn
1–3 câu TRẢ LỜI ĐÚNG LOẠI câu hỏi — dựa evidence/quotes, không khuôn mẫu giả.

- Hỏi có/không về MỘT tiêu chí (1 KS): được mở "Có./Không." rồi giải thích ngắn theo dữ liệu.
- Hỏi mô tả / "thế nào" (1 KS): trả lời trực tiếp tiêu chí, KHÔNG bắt buộc "Có./Không.".
- So sánh ≥2 KS (compare_mode = true hoặc user bảo so sánh):
  PHẢI nêu khác biệt hoặc điểm tương đồng CỤ THỂ theo tiêu chí hỏi (từng KS).
  CẤM câu chung chung kiểu "Có. Phần lớn khách… hài lòng với cả hai khách sạn"
  nếu câu hỏi là so sánh / đối chiếu (không phải yes/no về cả hai).
  Ví dụ đúng: "Về phục vụ, {A} được nhắc tích cực hơn trong dữ liệu đã thu thập;
  {B} có nhiều phản hồi hơn về ồn vào ban đêm."

### 3) Điểm nổi bật
Khi có đủ quotes cả hai nguồn: lấy ý cân bằng ~2 TripAdvisor + ~2 Chudu24 (3–5 bullet tổng).
So sánh ≥2 KS: nhóm theo tên khách sạn hoặc ghi rõ tên trong mỗi bullet.
Thiếu một nguồn thì lấy đủ từ nguồn còn lại. Viết tiếng Việt.

Mỗi bullet BẮT BUỘC kết thúc bằng markdown link — chữ hiện ra đúng "Xem chi tiết", click mở review:
`- {ý tiếng Việt} ({Nguồn}) [Xem chi tiết]({quotes[].review_url})`
Ví dụ đúng:
`- Vị trí thuận tiện, nhân viên tận tâm (Chudu24) [Xem chi tiết](https://www.chudu24.com/...)`
- Người dùng chỉ thấy chữ "Xem chi tiết" (đã là hyperlink). CẤM hiện URL trần.
- CẤM viết: "Xem chi tiết: https://...", "Xem thêm", hay để URL ngoài ngoặc markdown.
- `href` chỉ lấy nguyên văn `quotes[].review_url` của quote đúng ý — không bịa.
- Không có review_url: ghi `(Nguồn)` và bỏ phần link.

### 4) Điểm cần lưu ý
0–3 bullet điểm yếu nếu có trong dữ liệu đã thu thập.
Nếu không có: ghi đúng câu "Không thấy phản hồi tiêu cực rõ trong dữ liệu đã thu thập."
So sánh ≥2 KS: nêu rõ từng KS khi có khác biệt. Viết tiếng Việt.
Khi có review_url: cùng kiểu `({Nguồn}) [Xem chi tiết](url)` — chữ "Xem chi tiết" chính là link.

### 5) Nhận xét tiêu biểu
2–4 mục. Paraphrase tiếng Việt ≤1–2 câu (không copy nguyên văn dài).
Khi đủ 2 nguồn: 2 mục TripAdvisor + 2 mục Chudu24 (hoặc 1+1 nếu chỉ 2 mục).
So sánh ≥2 KS: phân bổ đều giữa các KS khi có quotes.

Mỗi mục BẮT BUỘC (KHÔNG ghi ngày) — "Xem chi tiết" chính là hyperlink:
`- "{paraphrase}" ({Nguồn}) [Xem chi tiết]({quotes[].review_url})`
Ví dụ đúng:
`- "Phòng sạch, nhân viên hữu ích." (TripAdvisor) [Xem chi tiết](https://www.tripadvisor.com/...)`
- Không ngày. Không "Xem thêm". Không hiện URL trần.
- Không có review_url → chỉ `(Nguồn)`.

### 6) Độ tin cậy
Một dòng: "Tổng hợp từ X review đã thu thập (Y Chudu24 + Z TripAdvisor)."
X/Y/Z lấy từ evidence sources[].sample.size (thiếu nguồn thì ghi 0).
≥2 KS: có thể tách theo từng KS nếu số liệu khác nhau.
Không giải thích thuật ngữ kỹ thuật.

### 7) Kết luận
2–3 câu tiếng Việt theo dữ liệu (phần lớn / ý trái chiều / khác biệt giữa các KS nếu so sánh).
Không viết "nên chọn / không nên chọn / đáng ở / nên đặt phòng".

## Quy tắc dữ liệu
- Tuân thủ response_policy (include_score_overview, hotel_count, compare_mode,
  scope, refuse_off_topic).
- "conversation_summary" + "history": ngữ cảnh nối tiếp; KHÔNG lặp nguyên văn trả lời trước.
- Dữ kiện chỉ từ evidence / quotes / web_findings — không bịa.
- URL chỉ lấy nguyên văn từ quotes[].review_url (ưu tiên), hoặc place / sources /
  web_findings.citations khi thật sự cần — không đưa URL vào phần mở đầu / bảng điểm.
- Ở Điểm nổi bật / Điểm cần lưu ý / Nhận xét tiêu biểu: chỉ hiện markdown
  `[Xem chi tiết](url)`, không dán URL trần, không "Xem thêm", không kèm ngày cạnh link.
- web_findings: fact ngoài corpus; diễn đạt tiếng Việt; reference_only = true → mục riêng cuối.

## Phạm vi hỗ trợ
- Trong phạm vi: đánh giá / trải nghiệm KS TP.HCM, tiện ích, vị trí, giá tham khảo,
  so sánh KS, **gợi ý KS theo quận/khu vực** (vd: Tân Bình) từ dữ liệu đã thu thập.
- response_policy.area_recommend = true: đây là câu hỏi chọn/gợi ý theo khu vực — VẪN trong phạm vi.
  Chỉ dùng KS có trong evidence / available_places. Nếu không có KS đúng khu vực trong dữ liệu:
  nói rõ chưa có KS đã thu thập tại khu vực đó, liệt kê ngắn các KS đang có (available_places),
  mời user chọn tên KS hoặc hỏi tiêu chí (ồn, ăn sáng…). Không bịa KS ngoài danh sách.
  Được nêu khác biệt theo điểm/review đã thu thập; tránh Imperative "bắt buộc phải chọn A".
- Ngoài phạm vi cứng (án mạng, tội phạm, chính trị, chứng khoán, lập trình, …):
  KHÔNG trả lời dù có nhắc tên khách sạn / hỏi lách.
  Lưu ý: "phạm vi tân bình/quận…" = khu vực địa lý, KHÔNG phải ngoài phạm vi hỗ trợ.
- Mixed (response_policy.refuse_off_topic = true hoặc scope = "mixed"):
  CHỈ trả lời phần liên quan khách sạn/review; BỎ QUA phần mua sắm / ngoài đề.
  Không tư vấn điện thoại, bóng đá, thời tiết, … dù user gắn thêm vào câu hỏi KS.
- Không bị “lách” bởi kiểu "tiện thể", "by the way", "không liên quan nhưng",
  "khách sạn X có vụ án…".

## CẤM
- Không địa chỉ, GPS/tọa độ, URL ở phần mở đầu.
- Không cột Score trùng / sample_mean trong bảng.
- Không chấm điểm tổng 8.5/10 hay hòa một rating duy nhất.
- Không "nên chọn / không nên chọn / đáng ở".
- Không bịa số liệu; không tự tạo URL.
- Không lộ thuật ngữ nội bộ (quotes, RAG, retrieval, sample_mean, JSON).
- Không để nguyên đoạn review tiếng Anh trong các mục nội dung.
- Không trả lời so sánh bằng câu "hài lòng với cả hai" chung chung.
- Không ghi ngày cạnh nguồn ở Nhận xét tiêu biểu; không dùng anchor "Xem thêm".
"""

SUMMARY_SYSTEM = """Bạn nén hội thoại cũ thành bộ nhớ ngắn cho trợ lý review khách sạn.
Gộp previous_summary với older_messages thành MỘT bản tóm tắt tiếng Việt <=150 từ.

Giữ lại: khách sạn / địa điểm đang bàn, tiêu chí user quan tâm (sạch sẽ, ồn, vị trí,
giá, gia đình...), ràng buộc user nêu (ngày đi, ngân sách, số người), kết luận đã chốt.
Bỏ: lời chào, câu dẫn, số liệu chi tiết, quote, URL.
Chỉ xuất bản tóm tắt, không thêm tiêu đề hay lời bình.
"""

GROUNDING_SYSTEM = """Bạn tìm thông tin khách sạn trên web, ưu tiên TripAdvisor.
Trả lời ngắn bằng tiếng Việt. Chỉ nêu fact có thể kiểm chứng; kèm gợi ý URL.
Không bịa điểm rating hay số lượng review. Không kết luận "nên ở / không nên ở".
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

    async def chat(
        self, payload: ReviewChatRequest, user: AuthUser | None = None
    ) -> ReviewChatResponse:
        place_ids = list(payload.place_ids)
        if payload.place_id:
            place_ids.append(payload.place_id)
        seen: set[str] = set()
        unique_ids: list[UUID] = []
        for pid in place_ids:
            key = str(pid)
            if key not in seen:
                seen.add(key)
                unique_ids.append(pid)

        context = ConversationContext()
        if payload.conversation_id:
            if user is None:
                raise ConversationAccessError(
                    "Sign in to continue an existing conversation"
                )
            context = await self._load_context(payload.conversation_id, user.id)
            if not unique_ids:
                unique_ids = context.place_ids

        catalog = self.places.list_places()
        area_recommend = is_area_recommend_intent(payload.message)

        # Message-named hotels win over a stale/auto-selected place_id
        # (e.g. user asks Park Hyatt while FE still has Lancaster selected).
        named = match_places_in_message(payload.message, catalog)
        if named:
            named_ids = [p.id for p in named]
            selected = {str(x) for x in unique_ids}
            if not unique_ids or any(str(i) not in selected for i in named_ids):
                logger.info(
                    "Resolving places from message names: %s",
                    [p.name for p in named],
                )
                unique_ids = named_ids

        # Area recommend ("KS nào ở Tân Bình"): resolve by address, don't keep
        # a stale conversation hotel that is outside the asked district.
        area_places = (
            match_places_in_area(payload.message, catalog, limit=5)
            if area_recommend
            else []
        )
        if area_recommend and not named:
            if area_places:
                unique_ids = [p.id for p in area_places]
                logger.info(
                    "Resolving places from area: %s",
                    [p.name for p in area_places],
                )
            else:
                unique_ids = []

        # Full refuse (hard out-of-scope, or soft with no hotel ask).
        # Mixed hotel+off-topic continues — answer hotel part only.
        if classify_scope(payload.message) == ScopeKind.OUT_OF_SCOPE:
            reply = abstain_message(RetrievalDecision.ABSTAIN_OUT_OF_SCOPE)
            conversation_id = payload.conversation_id
            if user is not None:
                conversation_id = self._persist(
                    user_id=user.id,
                    conversation_id=conversation_id,
                    message=payload.message,
                    reply=reply,
                    place_ids=unique_ids,
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
        else:
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

        gate = decide_retrieval(
            payload.message, all_quotes, evidences, settings=self.settings
        )
        available_places = [
            {
                "name": p.name,
                "address": p.address or "",
                "id": str(p.id),
            }
            for p in catalog
        ]
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
    ) -> ResolvedReply:
        if gate.decision in (
            RetrievalDecision.ABSTAIN_BEYOND_SAMPLE,
            RetrievalDecision.ABSTAIN_NO_DATA,
            RetrievalDecision.ABSTAIN_OUT_OF_SCOPE,
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
            )
            reply = self._apply_scope_notice(reply, gate.scope)
            return ResolvedReply(reply, "rag")

        policy = self._web_policy(gate.decision)
        web = await self._fetch_web_context(message, evidences, policy)
        if web is None:
            # No usable web → if we still have some evidence, answer from RAG;
            # otherwise abstain.
            if evidences or quotes or area_recommend:
                reply = await self._llm_from_evidence(
                    message,
                    evidences,
                    quotes,
                    None,
                    context=context,
                    scope=gate.scope,
                    area_recommend=area_recommend,
                    available_places=available_places,
                )
                reply = self._apply_scope_notice(reply, gate.scope)
                return ResolvedReply(reply, "rag")
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
        )
        if policy.reference_only:
            reply = f"{REFERENCE_NOTICE}\n\n{reply}"
        reply = self._apply_scope_notice(reply, gate.scope)
        return ResolvedReply(
            reply,
            source,
            citations=grounded.citations,
            search_suggestion_html=grounded.search_suggestion_html,
            reference_only=policy.reference_only,
        )

    @staticmethod
    def _apply_scope_notice(reply: str, scope: ScopeKind) -> str:
        if scope != ScopeKind.MIXED:
            return reply
        if MIXED_SCOPE_NOTICE in reply:
            return reply
        return f"{MIXED_SCOPE_NOTICE}\n\n{reply}"

    def _web_policy(self, decision: RetrievalDecision) -> WebPolicy:
        if decision == RetrievalDecision.NEED_GROUNDING_REFERENCE:
            # Price / promotions / live status: any source is acceptable as long
            # as it is cited, because the answer is explicitly informational.
            return WebPolicy(allowed_domains=(), reference_only=True)
        return WebPolicy(
            allowed_domains=tuple(self.settings.allowed_search_domains),
            reference_only=False,
        )

    async def _fetch_web_context(
        self,
        message: str,
        evidences: list[PlaceEvidenceResponse],
        policy: WebPolicy,
    ) -> tuple[WebContext, RetrievalSource] | None:
        if not evidences:
            # Without a resolved place there is no hotel name to search for;
            # a generic query would produce unverifiable results.
            logger.info("Skipping web grounding: no place resolved from request")
            return None

        hotel_name = evidences[0].place.name
        ta_url = evidences[0].place.tripadvisor_url
        domains = list(policy.allowed_domains)

        # 1) Gemini grounding
        if self.gemini.available:
            try:
                result = await self.gemini.grounded_complete(
                    GROUNDING_SYSTEM,
                    f"Khách sạn: {hotel_name}\nCâu hỏi: {message}",
                    prefer_url=ta_url if domains else None,
                    preferred_domains=domains,
                )
                if result.ok and result.text:
                    return result, "rag+gemini"
            except GeminiQuotaExhausted as exc:
                logger.warning("Gemini quota exhausted, falling back to web_search: %s", exc)
            except GeminiGroundingError as exc:
                logger.warning("Gemini grounding failed, falling back to web_search: %s", exc)

        # 2) OpenAI web_search under the same source policy
        search = await self.web_search.search(
            hotel_name=hotel_name,
            question=message,
            source_url=ta_url,
            allowed_domains=domains,
        )
        if search.ok and search.text:
            return (
                WebContext(text=search.text, citations=search.citations, ok=True),
                "rag+web_search",
            )
        return None

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
    ) -> str:
        # Order matters: memory first, then retrieved context, question last.
        user_payload: dict[str, Any] = {}
        if context and context.summary:
            user_payload["conversation_summary"] = context.summary
        if context and context.history:
            user_payload["history"] = context.history
        user_payload["response_policy"] = _response_policy(
            message,
            evidences,
            context,
            scope=scope,
            area_recommend=area_recommend,
            available_places=available_places if area_recommend else None,
        )
        user_payload["evidence"] = [e.model_dump(mode="json") for e in evidences]
        user_payload["quotes"] = [q.model_dump(mode="json") for q in quotes]
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
        return await self.llm.complete(
            SYSTEM_PROMPT,
            json.dumps(user_payload, ensure_ascii=False, indent=2),
        )

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
