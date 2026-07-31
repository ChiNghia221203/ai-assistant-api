
from fastapi import APIRouter, Depends
from domains.chat.schemas import ChatRequest, ChatResponse
from domains.chat.service import ChatService, get_chat_service

router = APIRouter(tags=["Chat"])


@router.post(
    "/",
    response_model=ChatResponse,
    summary="Gửi tin nhắn tới AI assistant",
)
async def create_chat(
    body: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    return await service.chat(body)
