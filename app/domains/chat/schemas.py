"""
DTO cho RAG — ingest documents + hỏi với context.
"""

from pydantic import BaseModel, Field


class IngestDocument(BaseModel):
    id: str = Field(..., min_length=1, description="ID duy nhất của chunk")
    content: str = Field(..., min_length=1, description="Nội dung văn bản")
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    ingested: int
    total_in_store: int


class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class RetrievedChunk(BaseModel):
    id: str
    content: str


class RagQueryResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
    mock: bool
