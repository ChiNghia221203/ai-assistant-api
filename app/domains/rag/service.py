"""
RAG business logic: retrieve → augment prompt → generate.
"""

from app.core.config import Settings, get_settings
from app.domains.rag.prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt
from app.domains.rag.schemas import (
    IngestRequest,
    IngestResponse,
    RagQueryRequest,
    RagQueryResponse,
    RetrievedChunk,
)
from app.infra.llm import LlmClient, get_llm_client
from app.infra.vector_store import (
    DocumentChunk,
    InMemoryVectorStore,
    get_vector_store,
)


class RagService:
    def __init__(
        self,
        store: InMemoryVectorStore | None = None,
        llm: LlmClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.store = store or get_vector_store()
        self.llm = llm or get_llm_client()
        self.settings = settings or get_settings()

    def ingest(self, payload: IngestRequest) -> IngestResponse:
        chunks = [
            DocumentChunk(id=d.id, content=d.content, metadata=d.metadata)
            for d in payload.documents
        ]
        count = self.store.upsert(chunks)
        return IngestResponse(ingested=count, total_in_store=self.store.count)

    async def query(self, payload: RagQueryRequest) -> RagQueryResponse:
        hits = self.store.similarity_search(payload.question, top_k=payload.top_k)

        if not hits:
            return RagQueryResponse(
                answer=(
                    "Chưa có tài liệu liên quan trong vector store. "
                    "Hãy gọi POST /api/v1/rag/ingest trước."
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
