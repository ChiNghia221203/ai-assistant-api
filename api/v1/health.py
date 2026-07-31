
from fastapi import APIRouter
from core.config import get_settings
from typing import Any
router = APIRouter(tags=["Health"])

@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Kiểm tra app còn sống — dùng cho Docker / load balancer."""
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "mock_llm": settings.mock_llm,
    }
