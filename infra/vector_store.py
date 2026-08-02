
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import httpx

from core.config import Settings, get_settings
from infra.database import get_supabase

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    place_id: str | None = None
    review_id: str | None = None
    source: str | None = None
    similarity: float | None = None


class EmbeddingClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.settings.mock_llm or not self.settings.openai_api_key:
            # Deterministic fake vectors for local smoke tests without OpenAI
            return [_mock_embedding(t) for t in texts]

        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.openai_embedding_model,
            "input": texts,
        }
        url = f"{self.settings.openai_base_url.rstrip('/')}/embeddings"
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0]


def _mock_embedding(text: str, dim: int = 1536) -> list[float]:
    """Tiny hash-based vector so local RAG smoke tests work without OpenAI."""
    vec = [0.0] * dim
    for i, ch in enumerate(text.encode("utf-8")[:dim]):
        vec[i % dim] += (ch / 255.0) * 0.01
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


class PgVectorStore:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: EmbeddingClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or EmbeddingClient(self.settings)

    async def similarity_search(
        self,
        query: str,
        top_k: int = 8,
        place_id: str | UUID | None = None,
    ) -> list[DocumentChunk]:
        query_embedding = await self.embedder.embed_one(query)
        sb = get_supabase()
        params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "match_count": top_k,
        }
        if place_id:
            params["filter_place_id"] = str(place_id)

        result = sb.rpc("match_documents", params).execute()
        rows = result.data or []
        return [
            DocumentChunk(
                id=str(row["id"]),
                content=row["content"],
                metadata=row.get("metadata") or {},
                place_id=str(row["place_id"]) if row.get("place_id") else None,
                review_id=str(row["review_id"]) if row.get("review_id") else None,
                source=row.get("source"),
                similarity=row.get("similarity"),
            )
            for row in rows
        ]

    async def upsert_document(
        self,
        *,
        place_id: str,
        review_id: str | None,
        source: str,
        content: str,
        metadata: dict[str, Any],
        document_id: str | None = None,
    ) -> str:
        embedding = await self.embedder.embed_one(content)
        sb = get_supabase()
        payload: dict[str, Any] = {
            "place_id": place_id,
            "review_id": review_id,
            "source": source,
            "content": content,
            "metadata": metadata,
            "embedding": embedding,
        }
        if document_id:
            payload["id"] = document_id
            sb.table("documents").upsert(payload).execute()
            return document_id

        result = sb.table("documents").insert(payload).execute()
        return str(result.data[0]["id"])


# Backward-compatible aliases used by older RAG sample code
InMemoryVectorStore = PgVectorStore


def get_vector_store() -> PgVectorStore:
    return PgVectorStore()


def get_embedding_client() -> EmbeddingClient:
    return EmbeddingClient()
