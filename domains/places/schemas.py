from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class PlaceOut(BaseModel):
    id: UUID
    slug: str
    name: str
    city: str
    address: str | None = None
    lat: float | None = None
    lng: float | None = None
    google_place_id: str | None = None
    chudu24_url: str | None = None
    tripadvisor_url: str | None = None


class ScoreBlock(BaseModel):
    value: float | None = None
    scale: int = 5
    n_total: int | None = None


class SampleBlock(BaseModel):
    size: int = 0
    date_min: date | None = None
    date_max: date | None = None


class SourceEvidence(BaseModel):
    source: str
    source_url: str = ""
    captured_at: datetime | None = None
    sample_policy: str = ""
    sample: SampleBlock = Field(default_factory=SampleBlock)
    scores: dict[str, Any] = Field(default_factory=dict)
    distribution: dict[str, int] = Field(default_factory=dict)
    reviews_available: bool = False


class QuoteOut(BaseModel):
    source: str
    review_date: date | None = None
    score: float | None = None
    text: str
    review_url: str | None = None
    similarity: float | None = None
    place_id: str | None = None


class PlaceEvidenceResponse(BaseModel):
    place: PlaceOut
    methodology: dict[str, str]
    sources: list[SourceEvidence]
    contrast: dict[str, Any]
    relevant_quotes: list[QuoteOut] = Field(default_factory=list)


class RagSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    place_id: UUID | None = None
    place_ids: list[UUID] = Field(default_factory=list)
    top_k: int = Field(default=8, ge=1, le=20)


class RagSearchResponse(BaseModel):
    quotes: list[QuoteOut]


class ReviewChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    place_id: UUID | None = None
    place_ids: list[UUID] = Field(default_factory=list)
    conversation_id: UUID | None = None
    user_id: UUID | None = None
    top_k: int = Field(default=8, ge=1, le=20)


class ReviewChatResponse(BaseModel):
    reply: str
    evidence: list[PlaceEvidenceResponse]
    quotes: list[QuoteOut]
    conversation_id: UUID | None = None
    mock: bool = False


class ConversationCreate(BaseModel):
    title: str = "New conversation"
    place_ids: list[UUID] = Field(default_factory=list)
    user_id: UUID


class ConversationOut(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    place_ids: list[UUID]
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    id: UUID
    conversation_id: UUID
    role: str
    content: str
    sources: list[Any] = Field(default_factory=list)
    evidence: Any | None = None
    created_at: datetime
