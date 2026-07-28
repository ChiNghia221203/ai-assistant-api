
from fastapi import APIRouter, Depends

from app.domains.rag.schemas import (
    IngestRequest,
    IngestResponse,
    RagQueryRequest,
    RagQueryResponse,
)
from app.domains.rag.service import RagService, get_rag_service

router = APIRouter(tags=["RAG"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Nạp tài liệu vào vector store (in-memory)",
)
async def ingest_documents(
    body: IngestRequest,
    service: RagService = Depends(get_rag_service),
) -> IngestResponse:
    return service.ingest(body)


@router.post(
    "/query",
    response_model=RagQueryResponse,
    summary="Hỏi AI với context lấy từ vector store",
)
async def query_rag(
    body: RagQueryRequest,
    service: RagService = Depends(get_rag_service),
) -> RagQueryResponse:
    return await service.query(body)
