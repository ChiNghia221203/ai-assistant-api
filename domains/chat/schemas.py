from uuid import UUID

from pydantic import BaseModel, Field

from domains.places.schemas import RetrievalSource, WebCitationOut


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Tin nhắn người dùng gửi cho AI",
        examples=["Khách sạn này có ồn không?"],
    )
    place_id: UUID | None = Field(
        default=None, description="Khách sạn đang xem (nếu có)"
    )
    place_ids: list[UUID] = Field(
        default_factory=list, description="Nhiều khách sạn để so sánh"
    )
    conversation_id: UUID | None = Field(
        default=None, description="Tiếp tục hội thoại đã có"
    )
    top_k: int = Field(default=8, ge=1, le=20)


class ChatResponse(BaseModel):
    reply: str
    mock: bool = Field(description="true nếu đang dùng MOCK_LLM")
    conversation_id: UUID | None = None
    retrieval_source: RetrievalSource = "rag"
    web_citations: list[WebCitationOut] = Field(default_factory=list)
    search_suggestion_html: str = ""
    reference_only: bool = False
