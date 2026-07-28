"""
Vector store in-memory — mẫu học RAG (không cần Chroma/Pinecone).

NestJS tương đương: repository/service lưu embedding + search.
Production: thay bằng Chroma, Qdrant, pgvector, ...
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DocumentChunk:
    id: str
    content: str
    metadata: dict = field(default_factory=dict)


class InMemoryVectorStore:
    """
    Search đơn giản bằng keyword overlap (không phải embedding thật).
    Đủ để hiểu flow RAG: ingest → retrieve → generate.
    """

    def __init__(self) -> None:
        self._docs: list[DocumentChunk] = []

    def clear(self) -> None:
        self._docs.clear()

    def upsert(self, chunks: list[DocumentChunk]) -> int:
        existing = {d.id: i for i, d in enumerate(self._docs)}
        for chunk in chunks:
            if chunk.id in existing:
                self._docs[existing[chunk.id]] = chunk
            else:
                self._docs.append(chunk)
        return len(chunks)

    def similarity_search(self, query: str, top_k: int = 3) -> list[DocumentChunk]:
        tokens = set(query.lower().split())
        if not tokens:
            return []

        scored: list[tuple[int, DocumentChunk]] = []
        for doc in self._docs:
            doc_tokens = set(doc.content.lower().split())
            score = len(tokens & doc_tokens)
            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    @property
    def count(self) -> int:
        return len(self._docs)


# Singleton trong process — ≈ NestJS provider scope DEFAULT
_vector_store = InMemoryVectorStore()


def get_vector_store() -> InMemoryVectorStore:
    return _vector_store
