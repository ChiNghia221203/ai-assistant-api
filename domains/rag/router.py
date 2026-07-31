from fastapi import APIRouter, Depends, HTTPException

from domains.places.schemas import RagSearchRequest, RagSearchResponse
from domains.places.service import PlacesService, get_places_service
from domains.rag.schemas import (
    IngestRequest,
    IngestResponse,
    RagQueryRequest,
    RagQueryResponse,
)
from domains.rag.service import RagService, get_rag_service

router = APIRouter(tags=["RAG"])


@router.post(
    "/search",
    response_model=RagSearchResponse,
    summary="Retrieve review quotes from pgvector (with date/source/url metadata)",
)
async def search_quotes(
    body: RagSearchRequest,
    service: PlacesService = Depends(get_places_service),
) -> RagSearchResponse:
    try:
        quotes = await service.search_quotes(
            query=body.query,
            place_id=body.place_id,
            place_ids=body.place_ids,
            top_k=body.top_k,
        )
        return RagSearchResponse(quotes=quotes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Legacy ingest helper (prefer scripts/ for hotel reviews)",
    deprecated=True,
)
async def ingest_documents(
    body: IngestRequest,
    service: RagService = Depends(get_rag_service),
) -> IngestResponse:
    return service.ingest(body)


@router.post(
    "/query",
    response_model=RagQueryResponse,
    summary="Legacy RAG Q&A (prefer /chat/review)",
    deprecated=True,
)
async def query_rag(
    body: RagQueryRequest,
    service: RagService = Depends(get_rag_service),
) -> RagQueryResponse:
    return await service.query(body)
