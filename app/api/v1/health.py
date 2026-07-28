"""
Health check endpoint.

NestJS tương đương:
  @Controller('health')
  @Get()
  health() { return { status: 'ok' }; }
"""

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> dict:
    """Kiểm tra app còn sống — dùng cho Docker / load balancer."""
    settings = get_settings()
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "mock_llm": settings.mock_llm,
    }
