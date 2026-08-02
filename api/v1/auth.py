from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from core.security import AuthUser, require_user

router = APIRouter(tags=["Auth"])


class MeResponse(BaseModel):
    id: UUID
    is_anonymous: bool
    email: str | None = None


@router.get(
    "/auth/me",
    response_model=MeResponse,
    summary="Who the bearer token belongs to (401 if the API rejects it)",
)
def me(user: AuthUser = Depends(require_user)) -> MeResponse:
    """Lets the frontend show session state and confirm the API accepts a token,
    whichever provider issued it (email, Google, anonymous)."""
    return MeResponse(
        id=user.id, is_anonymous=user.is_anonymous, email=user.email
    )
