"""Lightweight chat entry point — runs the same cascade as /chat/review."""

from __future__ import annotations

from core.config import Settings, get_settings
from core.security import AuthUser
from domains.chat.schemas import ChatRequest, ChatResponse
from domains.places.review_chat import ReviewChatService, get_review_chat_service
from domains.places.schemas import ReviewChatRequest


class ChatService:
    def __init__(
        self,
        review_chat: ReviewChatService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.review_chat = review_chat or get_review_chat_service()
        self.settings = settings or get_settings()

    async def chat(
        self, payload: ChatRequest, user: AuthUser | None = None
    ) -> ChatResponse:
        result = await self.review_chat.chat(
            ReviewChatRequest(
                message=payload.message,
                place_id=payload.place_id,
                place_ids=payload.place_ids,
                conversation_id=payload.conversation_id,
                top_k=payload.top_k,
            ),
            user=user,
        )
        return ChatResponse(
            reply=result.reply,
            mock=result.mock,
            conversation_id=result.conversation_id,
            retrieval_source=result.retrieval_source,
            web_citations=result.web_citations,
            search_suggestion_html=result.search_suggestion_html,
            reference_only=result.reference_only,
        )


def get_chat_service() -> ChatService:
    return ChatService()
