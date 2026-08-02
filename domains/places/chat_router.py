from uuid import UUID

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from core.config import Settings, get_settings
from core.rate_limit import chat_rate_limit
from core.security import AuthUser, optional_user, require_user, user_from_token
from domains.places.review_chat import (
    ConversationAccessError,
    ReviewChatService,
    get_review_chat_service,
)
from domains.places.schemas import (
    ClaimGuestRequest,
    ClaimGuestResponse,
    ConversationCreate,
    ConversationOut,
    MessageOut,
    ReviewChatRequest,
    ReviewChatResponse,
)
from infra.database import get_supabase

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ReviewChat"])


def _owned_conversation(conversation_id: UUID, user: AuthUser) -> dict:
    """Load a conversation the caller owns, or fail with 403/404."""
    sb = get_supabase()
    rows = (
        sb.table("conversations")
        .select("*")
        .eq("id", str(conversation_id))
        .limit(1)
        .execute()
        .data
        or []
    )
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
        )
    if str(rows[0].get("user_id")) != str(user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Conversation belongs to another user",
        )
    return rows[0]


@router.post(
    "/review",
    response_model=ReviewChatResponse,
    summary="Review/compare hotels from multi-source evidence + RAG quotes",
    dependencies=[Depends(chat_rate_limit)],
)
async def chat_review(
    body: ReviewChatRequest,
    service: ReviewChatService = Depends(get_review_chat_service),
    user: AuthUser | None = Depends(optional_user),
) -> ReviewChatResponse:
    try:
        return await service.chat(body, user=user)
    except HTTPException:
        raise
    except ConversationAccessError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat/review failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/conversations",
    response_model=ConversationOut,
    summary="Create conversation for the authenticated user",
)
def create_conversation(
    body: ConversationCreate,
    user: AuthUser = Depends(require_user),
) -> ConversationOut:
    sb = get_supabase()
    row = (
        sb.table("conversations")
        .insert(
            {
                "user_id": str(user.id),
                "title": body.title,
                "place_ids": [str(p) for p in body.place_ids],
            }
        )
        .execute()
        .data[0]
    )
    return ConversationOut(
        id=row["id"],
        user_id=row["user_id"],
        title=row["title"],
        place_ids=row.get("place_ids") or [],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get(
    "/conversations",
    response_model=list[ConversationOut],
    summary="List conversations of the authenticated user",
)
def list_conversations(
    user: AuthUser = Depends(require_user),
) -> list[ConversationOut]:
    sb = get_supabase()
    rows = (
        sb.table("conversations")
        .select("*")
        .eq("user_id", str(user.id))
        .order("updated_at", desc=True)
        .execute()
        .data
        or []
    )
    return [
        ConversationOut(
            id=r["id"],
            user_id=r["user_id"],
            title=r["title"],
            place_ids=r.get("place_ids") or [],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]


@router.post(
    "/conversations/claim",
    response_model=ClaimGuestResponse,
    summary="Move an anonymous session's conversations to the caller's account",
)
def claim_guest_conversations(
    body: ClaimGuestRequest,
    user: AuthUser = Depends(require_user),
    settings: Settings = Depends(get_settings),
) -> ClaimGuestResponse:
    """Rescues guest history when the accounts could not be linked in place.

    `signInWithOAuth` on a guest session produces a *new* user id, orphaning
    whatever was chatted as a guest. Holding the anonymous access token is the
    proof of ownership needed to reassign it.
    """
    guest = user_from_token(body.guest_token, settings)
    if not guest.is_anonymous:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="guest_token must belong to an anonymous user",
        )
    if guest.id == user.id:
        # Identity linking already kept the id: nothing to move.
        return ClaimGuestResponse(claimed=0)

    sb = get_supabase()
    moved = (
        sb.table("conversations")
        .update({"user_id": str(user.id)})
        .eq("user_id", str(guest.id))
        .execute()
        .data
        or []
    )
    return ClaimGuestResponse(claimed=len(moved))


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageOut],
)
def list_messages(
    conversation_id: UUID,
    user: AuthUser = Depends(require_user),
) -> list[MessageOut]:
    _owned_conversation(conversation_id, user)
    sb = get_supabase()
    rows = (
        sb.table("messages")
        .select("*")
        .eq("conversation_id", str(conversation_id))
        .order("created_at")
        .execute()
        .data
        or []
    )
    return [
        MessageOut(
            id=r["id"],
            conversation_id=r["conversation_id"],
            role=r["role"],
            content=r["content"],
            sources=r.get("sources") or [],
            evidence=r.get("evidence"),
            created_at=r["created_at"],
        )
        for r in rows
    ]
