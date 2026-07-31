from fastapi import APIRouter

from api.v1 import health
from domains.chat.router import router as chat_router
from domains.places.chat_router import router as review_chat_router
from domains.places.router import router as places_router
from domains.rag.router import router as rag_router

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(chat_router, prefix="/chat")
api_router.include_router(review_chat_router, prefix="/chat")
api_router.include_router(places_router, prefix="/places")
api_router.include_router(rag_router, prefix="/rag")
