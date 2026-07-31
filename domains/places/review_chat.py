"""Hotel review chat: evidence → RAG quotes → LLM (no verdict)."""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from core.config import Settings, get_settings
from domains.places.schemas import (
    PlaceEvidenceResponse,
    QuoteOut,
    ReviewChatRequest,
    ReviewChatResponse,
)
from domains.places.service import PlacesService, get_places_service
from infra.database import get_supabase
from infra.llm import LlmClient, get_llm_client

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là trợ lý tổng hợp đánh giá khách sạn đa nguồn.
Nhiệm vụ: trình bày rõ ràng evidence đã cung cấp để USER TỰ QUYẾT ĐỊNH.

BẮT BUỘC:
- Lặp lại methodology từng nguồn (Chudu24 100 mới / Google Places API / TripAdvisor meta).
- Hiện bảng hoặc bullet điểm theo TỪNG nguồn: site_overall, sample_mean, date_min→date_max, sample_size.
- Nêu contrast / lệch điểm giữa nguồn nếu có.
- Trích 2–5 quote kèm nguồn + ngày + link nếu có.
- So sánh 2 KS bằng bảng cùng tiêu chí nếu có nhiều place.

CẤM:
- Không chấm điểm tổng kiểu 8.5/10.
- Không viết "nên chọn / không nên chọn / đáng ở".
- Không bịa số liệu ngoài JSON evidence.
- Không hòa thành một rating duy nhất.
"""


class ReviewChatService:
    def __init__(
        self,
        places: PlacesService | None = None,
        llm: LlmClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.places = places or get_places_service()
        self.llm = llm or get_llm_client()
        self.settings = settings or get_settings()

    async def chat(self, payload: ReviewChatRequest) -> ReviewChatResponse:
        place_ids = list(payload.place_ids)
        if payload.place_id:
            place_ids.append(payload.place_id)
        # dedupe
        seen: set[str] = set()
        unique_ids: list[UUID] = []
        for pid in place_ids:
            key = str(pid)
            if key not in seen:
                seen.add(key)
                unique_ids.append(pid)

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
            # No place selected: try global quote search then map places
            quotes = await self.places.search_quotes(
                payload.message, top_k=payload.top_k
            )
            all_quotes = quotes
            place_from_quotes = []
            for q in quotes:
                if q.place_id and q.place_id not in place_from_quotes:
                    place_from_quotes.append(q.place_id)
            for pid in place_from_quotes[:3]:
                pq = [q for q in quotes if q.place_id == pid]
                evidences.append(self.places.get_evidence(pid, quotes=pq))

        user_payload = {
            "question": payload.message,
            "evidence": [e.model_dump(mode="json") for e in evidences],
            "quotes": [q.model_dump(mode="json") for q in all_quotes],
        }
        reply = await self.llm.complete(
            SYSTEM_PROMPT,
            json.dumps(user_payload, ensure_ascii=False, indent=2),
        )

        conversation_id = payload.conversation_id
        if payload.user_id:
            conversation_id = self._persist(
                user_id=payload.user_id,
                conversation_id=conversation_id,
                message=payload.message,
                reply=reply,
                place_ids=unique_ids,
                evidences=evidences,
                quotes=all_quotes,
            )

        return ReviewChatResponse(
            reply=reply,
            evidence=evidences,
            quotes=all_quotes,
            conversation_id=conversation_id,
            mock=self.settings.mock_llm or not self.settings.openai_api_key,
        )

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
            sb.table("conversations").update(
                {"place_ids": [str(p) for p in place_ids]}
            ).eq("id", str(conversation_id)).execute()

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
