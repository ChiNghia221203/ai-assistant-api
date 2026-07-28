"""
Chat business logic.

NestJS tương đương: @Injectable() ChatService
"""

from app.core.config import Settings, get_settings
from app.domains.chat.schemas import ChatRequest, ChatResponse
from app.infra.llm import LlmClient, get_llm_client

DEFAULT_SYSTEM = (
    "Bạn là trợ lý AI ngắn gọn, trả lời bằng tiếng Việt. "
    "Giải thích khái niệm kỹ thuật rõ ràng cho người biết NestJS đang học Python."
)


class ChatService:
    def __init__(
        self,
        llm: LlmClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.llm = llm or get_llm_client()
        self.settings = settings or get_settings()

    async def chat(self, payload: ChatRequest) -> ChatResponse:
        system = payload.system_prompt or DEFAULT_SYSTEM
        reply = await self.llm.complete(system=system, user=payload.message)
        return ChatResponse(
            reply=reply,
            mock=self.settings.mock_llm or not self.settings.openai_api_key,
        )


def get_chat_service() -> ChatService:
    """≈ NestJS DI: constructor(private chatService: ChatService)."""
    return ChatService()
