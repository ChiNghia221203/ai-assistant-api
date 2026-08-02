"""Classify the question, then decide: RAG only, grounding, or abstain."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from core.config import Settings, get_settings
from domains.places.place_match import is_area_recommend_intent
from domains.places.schemas import PlaceEvidenceResponse, QuoteOut

# User wants more reviews than the ingested sample (~100 newest).
_BEYOND_SAMPLE_RE = re.compile(
    r"("
    r"\b[2-9]\d{2,}\s*review"
    r"|\b\d{3,}\s*đánh\s*giá"
    r"|hơn\s*100"
    r"|trên\s*100"
    r"|hơn\s*một\s*trăm"
    r"|tất\s*cả\s*(các\s*)?(review|đánh\s*giá)"
    r"|toàn\s*bộ\s*(các\s*)?(review|đánh\s*giá)"
    r"|mọi\s*(review|đánh\s*giá)"
    r"|phân\s*tích\s*\d{3,}"
    r"|lấy\s*\d{3,}\s*review"
    r")",
    re.IGNORECASE,
)

# Time-sensitive: never reliable from an offline review corpus.
_PRICE_LIVE_RE = re.compile(
    r"("
    # "giá" but not inside "đánh giá" (review) or "giá trị" (value)
    r"(?<!đánh\s)giá(?!\s*trị)"
    r"|bao\s*nhiêu\s*(tiền|đồng|vnd|usd|k|một\s*đêm|/\s*đêm)"
    r"|khuyến\s*mãi"
    r"|ưu\s*đãi"
    r"|giảm\s*giá"
    r"|voucher"
    r"|mã\s*giảm"
    r"|phụ\s*thu"
    r"|còn\s*(hoạt\s*động|mở\s*cửa)"
    r"|đóng\s*cửa"
    r"|(đang\s*)?(sửa\s*chữa|cải\s*tạo|tu\s*sửa)"
    r"|mới\s*(khai\s*trương|mở)"
    r"|\bprice\b"
    r"|\bpromotion\b"
    r"|\bdiscount\b"
    r"|\bdeal\b"
    r")",
    re.IGNORECASE,
)

# Subjective quality: the ingested corpus is the best source, web adds noise.
_EXPERIENCE_RE = re.compile(
    r"("
    r"\bồn\b"
    r"|yên\s*tĩnh"
    r"|sạch"
    r"|bẩn"
    r"|\bmùi\b"
    r"|thái\s*độ"
    r"|nhân\s*viên"
    r"|phục\s*vụ"
    r"|chất\s*lượng"
    r"|\btệ\b"
    r"|\bngon\b"
    r"|thoải\s*mái"
    r"|đáng"
    r"|trải\s*nghiệm"
    r"|giường"
    r"|\bnệm\b"
    r"|\bchật\b"
    r"|\bcũ\s*kỹ\b"
    r"|\bxuống\s*cấp\b"
    r")",
    re.IGNORECASE,
)

# Static facts about the property: web can confirm, trusted sources only.
_STATIC_FACT_RE = re.compile(
    r"("
    r"hồ\s*bơi"
    r"|bể\s*bơi"
    r"|\bgym\b"
    r"|phòng\s*tập"
    r"|\bspa\b"
    r"|đỗ\s*xe"
    r"|bãi\s*xe"
    r"|\bwifi\b"
    r"|ăn\s*sáng"
    r"|buffet"
    r"|đưa\s*đón"
    r"|sân\s*bay"
    r"|địa\s*chỉ"
    r"|ở\s*đâu"
    r"|\bgần\b"
    r"|bao\s*xa"
    r"|check\s*-?\s*(in|out)"
    r"|thú\s*cưng"
    r"|\bpet\b"
    r"|thang\s*máy"
    r"|nhà\s*hàng"
    r"|tiện\s*ích"
    r")",
    re.IGNORECASE,
)

# Explicit hotel-review ask (beyond amenity keywords).
_HOTEL_ASK_RE = re.compile(
    r"("
    r"khách\s*sạn|khach\s*san|\bhotel\b|\bresort\b"
    r"|đánh\s*giá|\breview\b|so\s*sánh|đối\s*chiếu"
    r"|\bphòng\b|đặt\s*phòng|nên\s*ở|trải\s*nghiệm\s*(ks|khách)"
    r")",
    re.IGNORECASE,
)

# Always refuse — even if a hotel name is used as camouflage ("lách").
# Keep patterns tight: "phạm vi tân bình" (area) must NEVER match tội phạm / vụ án.
_HARD_OUT_OF_SCOPE_RE = re.compile(
    r"("
    r"án\s*mạng|an\s*mang|giết\s*người|giet\s*nguoi|homicide|\bmurder\b"
    r"|\bvụ\s*án\b|\bvu\s*an\b|\btội\s*phạm\b|\btoi\s*pham\b|\bcrime\b"
    r"|cảnh\s*sát|canh\s*sat|\bpolice\b|bắt\s*cóc|bat\s*coc"
    r"|chết\s*người|chet\s*nguoi|xác\s*chết|xac\s*chet"
    r"|scandal\s*(tình|sex|tội)|dâm\s*ô"
    r"|chứng\s*khoán|chung\s*khoan|\bcrypto\b|\bbitcoin\b"
    r"|bầu\s*cử|bau\s*cu|\belection\b|chính\s*trị|chinh\s*tri"
    r"|lập\s*trình|lap\s*trinh|\bpython\b|\bjavascript\b|viết\s*code|viet\s*code"
    r")",
    re.IGNORECASE,
)

# Refuse when alone; if mixed with a real hotel ask → answer hotel only.
_SOFT_OUT_OF_SCOPE_RE = re.compile(
    r"("
    r"mua\s*(điện\s*thoại|dien\s*thoai|iphone|samsung|laptop|máy\s*tính|"
    r"may\s*tinh|xe\s*máy|xe\s*may|ô\s*tô|o\s*to|xe\s*hơi|ipad|máy\s*ảnh)"
    r"|(^|[^\w])(điện\s*thoại|dien\s*thoai|iphone|samsung|laptop)([^\w]|$)"
    r"|tư\s*vấn\s*mua|nen\s*mua\s*(con|cái|chiếc)|nên\s*mua\s*(con|cái|chiếc)"
    r"|bóng\s*đá|bong\s*da|world\s*cup|\bfootball\b|\bsoccer\b"
    r"|nấu\s*ăn|nau\s*an|\brecipe\b|công\s*thức\s*nấu"
    r"|thời\s*tiết\s*(hôm\s*nay|ngày\s*mai)|thoi\s*tiet"
    r"|by\s*the\s*way.{0,40}(mua|buy|phone|iphone)"
    r"|tiện\s*thể.{0,40}(mua|điện\s*thoại|iphone|laptop)"
    r"|ngoài\s*lề.{0,40}(mua|điện\s*thoại|iphone|laptop)"
    r"|không\s*liên\s*quan.{0,40}(mua|điện\s*thoại|iphone)"
    r"|hỏi\s*thêm.{0,40}(mua|điện\s*thoại|iphone|laptop)"
    r")",
    re.IGNORECASE,
)

ABSTAIN_BEYOND_SAMPLE = (
    "Hiện hệ thống chỉ có tối đa khoảng **100 review mới nhất** đã thu thập "
    "theo methodology từng nguồn (Chudu24 / TripAdvisor / Google Places mẫu nhỏ). "
    "Yêu cầu phân tích nhiều hơn (ví dụ 200 review / toàn bộ review) **chưa có trong dữ liệu**, "
    "nên mình chưa thể tổng hợp chính xác. "
    "Bạn có thể hỏi trong phạm vi evidence hiện có (điểm theo nguồn, quote liên quan, so sánh KS)."
)

ABSTAIN_NO_DATA = (
    "Hiện **chưa có đủ dữ liệu đáng tin** để trả lời câu hỏi này "
    "(thiếu evidence trong hệ thống và không lấy được nguồn web phù hợp). "
    "Vui lòng chọn khách sạn đã được ingest, hoặc hỏi lại khi dữ liệu được cập nhật."
)

ABSTAIN_OUT_OF_SCOPE = (
    "Câu hỏi này **không thuộc phạm vi** hỗ trợ của trợ lý. "
    "Mình chỉ hỗ trợ thông tin đánh giá / trải nghiệm khách sạn tại TP. Hồ Chí Minh "
    "từ dữ liệu đã thu thập (Chudu24, TripAdvisor) và các nhu cầu liên quan cơ bản "
    "(phòng, nhân viên, ăn sáng, vị trí, tiện ích, giá tham khảo, so sánh khách sạn). "
    "Bạn hãy hỏi lại trong phạm vi đó nhé."
)

MIXED_SCOPE_NOTICE = (
    "> **Phạm vi:** phần hỏi ngoài chủ đề khách sạn / review "
    "(ví dụ mua sắm, tin tức không liên quan) **không thuộc phạm vi hỗ trợ** — "
    "mình chỉ trả lời phần liên quan khách sạn bên dưới."
)

REFERENCE_NOTICE = (
    "> **Chỉ mang tính tham khảo:** thông tin về giá / khuyến mãi / tình trạng hoạt động "
    "dưới đây lấy từ web tại thời điểm truy vấn, không thuộc dữ liệu review đã kiểm chứng "
    "của hệ thống. Vui lòng kiểm tra lại trên trang đặt phòng chính thức."
)


class QuestionIntent(str, Enum):
    PRICE_OR_LIVE = "price_or_live"
    STATIC_FACT = "static_fact"
    EXPERIENCE = "experience"


class ScopeKind(str, Enum):
    IN_SCOPE = "in_scope"
    OUT_OF_SCOPE = "out_of_scope"
    # Hotel ask + off-topic in one message — answer hotel part only.
    MIXED = "mixed"


class RetrievalDecision(str, Enum):
    RAG_ONLY = "rag_only"
    # Static fact: grounding restricted to trusted domains
    NEED_GROUNDING = "need_grounding"
    # Price / live status: grounding on the open web, reference-only answer
    NEED_GROUNDING_REFERENCE = "need_grounding_reference"
    ABSTAIN_BEYOND_SAMPLE = "abstain_beyond_sample"
    ABSTAIN_NO_DATA = "abstain_no_data"
    ABSTAIN_OUT_OF_SCOPE = "abstain_out_of_scope"


@dataclass(frozen=True)
class GateResult:
    decision: RetrievalDecision
    intent: QuestionIntent = QuestionIntent.EXPERIENCE
    reason: str = ""
    scope: ScopeKind = ScopeKind.IN_SCOPE


def is_beyond_sample_intent(message: str) -> bool:
    return bool(_BEYOND_SAMPLE_RE.search(message or ""))


def has_hard_out_of_scope(message: str) -> bool:
    return bool(_HARD_OUT_OF_SCOPE_RE.search(message or ""))


def has_soft_out_of_scope(message: str) -> bool:
    return bool(_SOFT_OUT_OF_SCOPE_RE.search(message or ""))


def is_out_of_scope_intent(message: str) -> bool:
    """True when the whole message should be refused (no hotel answer)."""
    return classify_scope(message) == ScopeKind.OUT_OF_SCOPE


def has_in_scope_hotel_ask(message: str) -> bool:
    """True when the user is asking something the hotel assistant can answer."""
    text = message or ""
    if is_area_recommend_intent(text):
        return True
    if _PRICE_LIVE_RE.search(text):
        return True
    if _EXPERIENCE_RE.search(text):
        return True
    if _STATIC_FACT_RE.search(text):
        return True
    if _HOTEL_ASK_RE.search(text):
        return True
    return False


def classify_scope(message: str) -> ScopeKind:
    """Resolve in-scope / out-of-scope / mixed (hotel + off-topic).

    Hard topics (crime, politics, …) always refuse — even with a hotel name.
    Soft topics (shopping, sports, …) refuse alone; if mixed with a real hotel
    ask, allow RAG for the hotel part only.
    """
    text = message or ""
    hard = has_hard_out_of_scope(text)
    soft = has_soft_out_of_scope(text)
    hotel = has_in_scope_hotel_ask(text)

    if hard:
        return ScopeKind.OUT_OF_SCOPE
    if soft and hotel:
        return ScopeKind.MIXED
    if soft:
        return ScopeKind.OUT_OF_SCOPE
    return ScopeKind.IN_SCOPE


def classify_intent(message: str) -> QuestionIntent:
    """Experience is the default: cheapest and best served by the corpus."""
    text = message or ""
    if _PRICE_LIVE_RE.search(text):
        return QuestionIntent.PRICE_OR_LIVE
    if _EXPERIENCE_RE.search(text):
        return QuestionIntent.EXPERIENCE
    if _STATIC_FACT_RE.search(text):
        return QuestionIntent.STATIC_FACT
    return QuestionIntent.EXPERIENCE


def decide_retrieval(
    message: str,
    quotes: list[QuoteOut],
    evidences: list[PlaceEvidenceResponse],
    settings: Settings | None = None,
) -> GateResult:
    cfg = settings or get_settings()
    scope = classify_scope(message)

    if scope == ScopeKind.OUT_OF_SCOPE:
        return GateResult(
            RetrievalDecision.ABSTAIN_OUT_OF_SCOPE,
            QuestionIntent.EXPERIENCE,
            reason="question_outside_hotel_review_scope",
            scope=scope,
        )

    if is_beyond_sample_intent(message):
        return GateResult(
            RetrievalDecision.ABSTAIN_BEYOND_SAMPLE,
            QuestionIntent.EXPERIENCE,
            reason="user_requested_beyond_ingested_sample",
            scope=scope,
        )

    intent = classify_intent(message)

    # Price / promotions / operating status are never trustworthy from old
    # reviews, so ground them even when retrieval looks strong.
    if intent == QuestionIntent.PRICE_OR_LIVE:
        return GateResult(
            RetrievalDecision.NEED_GROUNDING_REFERENCE,
            intent,
            reason="time_sensitive_question",
            scope=scope,
        )

    max_sim = max((q.similarity or 0.0 for q in quotes), default=0.0)
    enough_quotes = (
        len(quotes) >= cfg.rag_min_quotes and max_sim >= cfg.rag_min_similarity
    )
    has_evidence = bool(evidences)

    if has_evidence and enough_quotes:
        return GateResult(
            RetrievalDecision.RAG_ONLY,
            intent,
            reason="rag_sufficient",
            scope=scope,
        )

    # Area recommend with no district match: still ask LLM using the place catalog
    # (honest "chưa có KS tại quận X trong dữ liệu") — do not abstain as out-of-scope.
    if is_area_recommend_intent(message) and not has_evidence and not quotes:
        return GateResult(
            RetrievalDecision.RAG_ONLY,
            intent,
            reason="area_recommend_catalog_only",
            scope=scope,
        )

    # Subjective questions: web search cannot replace the review corpus.
    if intent == QuestionIntent.EXPERIENCE:
        if has_evidence or quotes:
            return GateResult(
                RetrievalDecision.RAG_ONLY,
                intent,
                reason="experience_rag_only",
                scope=scope,
            )
        return GateResult(
            RetrievalDecision.ABSTAIN_NO_DATA,
            intent,
            reason="no_experience_data",
            scope=scope,
        )

    return GateResult(
        RetrievalDecision.NEED_GROUNDING,
        intent,
        reason="static_fact_rag_insufficient" if has_evidence else "no_place_evidence",
        scope=scope,
    )


def abstain_message(decision: RetrievalDecision) -> str:
    if decision == RetrievalDecision.ABSTAIN_BEYOND_SAMPLE:
        return ABSTAIN_BEYOND_SAMPLE
    if decision == RetrievalDecision.ABSTAIN_OUT_OF_SCOPE:
        return ABSTAIN_OUT_OF_SCOPE
    return ABSTAIN_NO_DATA
