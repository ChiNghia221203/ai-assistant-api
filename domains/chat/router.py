from fastapi import APIRouter, Depends, HTTPException, status

from core.rate_limit import chat_rate_limit
from core.security import AuthUser, optional_user
from domains.chat.schemas import ChatRequest, ChatResponse
from domains.chat.service import ChatService, get_chat_service
from domains.places.review_chat import ConversationAccessError

router = APIRouter(tags=["Chat"])


@router.post(
    "/",
    response_model=ChatResponse,
    summary="Chat qua full cascade: RAG → grounding → web_search → abstain",
    dependencies=[Depends(chat_rate_limit)],
)
async def create_chat(
    body: ChatRequest,
    service: ChatService = Depends(get_chat_service),
    user: AuthUser | None = Depends(optional_user),
) -> ChatResponse:
    try:
        return await service.chat(body, user=user)
    except HTTPException:
        raise
    except ConversationAccessError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc
