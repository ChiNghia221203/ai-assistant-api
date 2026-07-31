"""Legacy RAG service — hotel flow uses places + /rag/search + /chat/review."""

from __future__ import annotations

from core.config import Settings, get_settings
from domains.rag.prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt
from domains.rag.schemas import (
    IngestRequest,
    IngestResponse,
    RagQueryRequest,
    RagQueryResponse,
    RetrievedChunk,
)
from infra.llm import LlmClient, get_llm_client
from infra.vector_store import PgVectorStore, get_vector_store


class RagService:
    def __init__(
        self,
        store: PgVectorStore | None = None,
        llm: LlmClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.store = store or get_vector_store()
        self.llm = llm or get_llm_client()
        self.settings = settings or get_settings()

    def ingest(self, payload: IngestRequest) -> IngestResponse:
        # Hotel corpus is ingested via local crawl scripts into Supabase.
        return IngestResponse(ingested=0, total_in_store=0)

    async def query(self, payload: RagQueryRequest) -> RagQueryResponse:
        hits = await self.store.similarity_search(
            payload.question, top_k=payload.top_k
        )
        if not hits:
            return RagQueryResponse(
                answer=(
                    "Chưa có tài liệu trong pgvector. "
                    "Corpus khách sạn nằm trên Supabase — kiểm tra seed/embed "
                    "hoặc dùng POST /api/v1/chat/review."
                ),
                sources=[],
                mock=self.settings.mock_llm or not self.settings.openai_api_key,
            )

        context = "\n\n---\n\n".join(f"[{h.id}] {h.content}" for h in hits)
        user_prompt = build_rag_user_prompt(payload.question, context)
        answer = await self.llm.complete(RAG_SYSTEM_PROMPT, user_prompt)
        return RagQueryResponse(
            answer=answer,
            sources=[RetrievedChunk(id=h.id, content=h.content) for h in hits],
            mock=self.settings.mock_llm or not self.settings.openai_api_key,
        )


def get_rag_service() -> RagService:
    return RagService()
