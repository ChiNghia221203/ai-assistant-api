"""Places + multi-source evidence service."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any
from uuid import UUID

from domains.places.schemas import (
    PlaceEvidenceResponse,
    PlaceOut,
    QuoteOut,
    SampleBlock,
    SourceEvidence,
)
from domains.places.stats import contrast_site_overall
from infra.database import get_supabase
from infra.vector_store import PgVectorStore, get_vector_store

logger = logging.getLogger(__name__)

METHODOLOGY = {
    "chudu24": (
        "100 review mới nhất theo ngày đăng (hoặc ít hơn nếu thiếu); "
        "sample_mean tính từ mẫu đó."
    ),
    "google": (
        "rating + reviews trả về từ Places API (thường ≤5); "
        "không phải 100 review mới nhất."
    ),
    "tripadvisor": (
        "100 review mới nhất theo ngày đăng (hoặc ít hơn nếu thiếu); "
        "sample_mean tính từ mẫu đó; điểm tổng site vẫn hiện."
    ),
}


class PlacesService:
    def __init__(self, store: PgVectorStore | None = None) -> None:
        self.store = store or get_vector_store()

    def list_places(self, city: str | None = "Ho Chi Minh") -> list[PlaceOut]:
        sb = get_supabase()
        q = sb.table("places").select("*").order("name")
        if city:
            q = q.eq("city", city)
        rows = q.execute().data or []
        return [self._place_out(r) for r in rows]

    def get_place(self, place_id: UUID | str) -> PlaceOut | None:
        sb = get_supabase()
        rows = (
            sb.table("places")
            .select("*")
            .eq("id", str(place_id))
            .limit(1)
            .execute()
            .data
            or []
        )
        if not rows:
            return None
        return self._place_out(rows[0])

    def get_place_by_slug(self, slug: str) -> PlaceOut | None:
        sb = get_supabase()
        rows = (
            sb.table("places").select("*").eq("slug", slug).limit(1).execute().data
            or []
        )
        if not rows:
            return None
        return self._place_out(rows[0])

    def get_evidence(
        self,
        place_id: UUID | str,
        quotes: list[QuoteOut] | None = None,
    ) -> PlaceEvidenceResponse:
        place = self.get_place(place_id)
        if not place:
            raise ValueError(f"Place not found: {place_id}")

        sb = get_supabase()
        snaps = (
            sb.table("source_snapshots")
            .select("*")
            .eq("place_id", str(place_id))
            .execute()
            .data
            or []
        )
        sources = [self._source_evidence(s) for s in snaps]
        # Stable order
        order = {"chudu24": 0, "google": 1, "tripadvisor": 2}
        sources.sort(key=lambda s: order.get(s.source, 99))

        return PlaceEvidenceResponse(
            place=place,
            methodology=METHODOLOGY,
            sources=sources,
            contrast=contrast_site_overall(snaps),
            relevant_quotes=quotes or [],
        )

    async def search_quotes(
        self,
        query: str,
        place_id: UUID | str | None = None,
        place_ids: list[UUID] | None = None,
        top_k: int = 8,
    ) -> list[QuoteOut]:
        ids = list(place_ids or [])
        if place_id:
            ids.append(UUID(str(place_id)) if not isinstance(place_id, UUID) else place_id)

        if not ids:
            chunks = await self.store.similarity_search(query, top_k=top_k)
            return [self._quote_from_chunk(c.__dict__) for c in chunks]

        # Search per place then merge by similarity
        merged: list[QuoteOut] = []
        per = max(2, top_k // max(len(ids), 1))
        for pid in ids:
            chunks = await self.store.similarity_search(
                query, top_k=per, place_id=pid
            )
            merged.extend(self._quote_from_chunk(c.__dict__) for c in chunks)

        merged.sort(key=lambda q: q.similarity or 0.0, reverse=True)
        return _diversify_quotes(merged, top_k)

    def _place_out(self, row: dict[str, Any]) -> PlaceOut:
        return PlaceOut(
            id=row["id"],
            slug=row["slug"],
            name=row["name"],
            city=row.get("city") or "Ho Chi Minh",
            address=row.get("address"),
            lat=row.get("lat"),
            lng=row.get("lng"),
            google_place_id=row.get("google_place_id"),
            chudu24_url=row.get("chudu24_url"),
            tripadvisor_url=row.get("tripadvisor_url"),
        )

    def _source_evidence(self, row: dict[str, Any]) -> SourceEvidence:
        date_min = row.get("date_min")
        date_max = row.get("date_max")
        if isinstance(date_min, str):
            date_min = date.fromisoformat(date_min)
        if isinstance(date_max, str):
            date_max = date.fromisoformat(date_max)

        captured = row.get("captured_at")
        if isinstance(captured, str):
            captured = datetime.fromisoformat(captured.replace("Z", "+00:00"))

        scores: dict[str, Any] = {
            "site_overall": {
                "value": float(row["site_overall"])
                if row.get("site_overall") is not None
                else None,
                "scale": row.get("site_overall_scale") or 5,
                "n_total": row.get("site_n_total"),
            },
            "sample_mean": {
                "value": float(row["sample_mean"])
                if row.get("sample_mean") is not None
                else None,
                "scale": row.get("sample_mean_scale") or 5,
                "n": row.get("sample_size") or 0,
            },
        }
        return SourceEvidence(
            source=row["source"],
            source_url=row.get("source_url") or "",
            captured_at=captured,
            sample_policy=row.get("sample_policy") or "",
            sample=SampleBlock(
                size=row.get("sample_size") or 0,
                date_min=date_min,
                date_max=date_max,
            ),
            scores=scores,
            distribution=row.get("distribution") or {},
            reviews_available=bool(row.get("reviews_available")),
        )

    def _quote_from_chunk(self, row: dict[str, Any]) -> QuoteOut:
        meta = row.get("metadata") or {}
        review_date = meta.get("review_date")
        if isinstance(review_date, str) and review_date:
            try:
                review_date = date.fromisoformat(review_date)
            except ValueError:
                review_date = None
        return QuoteOut(
            source=row.get("source") or meta.get("source") or "unknown",
            review_date=review_date,
            score=meta.get("score"),
            text=row.get("content") or "",
            review_url=meta.get("review_url"),
            similarity=row.get("similarity"),
            place_id=str(row["place_id"]) if row.get("place_id") else None,
        )


def _diversify_quotes(quotes: list[QuoteOut], top_k: int) -> list[QuoteOut]:
    """Prefer mix of higher and lower scores when available."""
    if len(quotes) <= top_k:
        return quotes
    high = [q for q in quotes if (q.score or 3) >= 4]
    low = [q for q in quotes if (q.score or 3) < 3.5]
    mid = [q for q in quotes if q not in high and q not in low]
    out: list[QuoteOut] = []
    while len(out) < top_k and (high or low or mid):
        if high and len(out) < top_k:
            out.append(high.pop(0))
        if low and len(out) < top_k:
            out.append(low.pop(0))
        if mid and len(out) < top_k:
            out.append(mid.pop(0))
    return out[:top_k]


def get_places_service() -> PlacesService:
    return PlacesService()
