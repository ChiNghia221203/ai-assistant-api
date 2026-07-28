
from fastapi import APIRouter
from app.api.v1 import health
from app.domains.chat.router import router as chat_router
from app.domains.rag.router import router as rag_router

api_router = APIRouter()

# ≈ app.use('/health', healthModule)
api_router.include_router(health.router)

# ≈ app.use('/chat', chatModule)
api_router.include_router(chat_router, prefix="/chat")

# ≈ app.use('/rag', ragModule)
api_router.include_router(rag_router, prefix="/rag")
