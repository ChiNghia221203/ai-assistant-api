from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from domains.places.review_chat import ReviewChatService, get_review_chat_service
from domains.places.schemas import (
    ConversationCreate,
    ConversationOut,
    MessageOut,
    ReviewChatRequest,
    ReviewChatResponse,
)
from infra.database import get_supabase

router = APIRouter(tags=["ReviewChat"])


@router.post(
    "/review",
    response_model=ReviewChatResponse,
    summary="Review/compare hotels from multi-source evidence + RAG quotes",
)
async def chat_review(
    body: ReviewChatRequest,
    service: ReviewChatService = Depends(get_review_chat_service),
) -> ReviewChatResponse:
    try:
        return await service.chat(body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/conversations",
    response_model=ConversationOut,
    summary="Create conversation (requires user_id from Supabase Auth)",
)
def create_conversation(body: ConversationCreate) -> ConversationOut:
    sb = get_supabase()
    row = (
        sb.table("conversations")
        .insert(
            {
                "user_id": str(body.user_id),
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
    summary="List conversations for a user",
)
def list_conversations(user_id: UUID) -> list[ConversationOut]:
    sb = get_supabase()
    rows = (
        sb.table("conversations")
        .select("*")
        .eq("user_id", str(user_id))
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


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageOut],
)
def list_messages(conversation_id: UUID) -> list[MessageOut]:
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
