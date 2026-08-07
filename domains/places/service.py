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
from core.config import Settings, get_settings
from domains.places.stats import contrast_site_overall
from infra.database import get_supabase
from infra.vector_store import DocumentChunk, PgVectorStore, get_vector_store

logger = logging.getLogger(__name__)

_TA_EN_BOOSTER = (
    "TripAdvisor hotel guest review staff service "
    "cleanliness location room breakfast"
)

METHODOLOGY = {
    "chudu24": (
        "Chudu24: tối đa 100 review mới nhất theo ngày đăng "
        "(hoặc ít hơn nếu nguồn chưa đủ)."
    ),
    "google": (
        "Google Places: rating + một số review API trả về (thường ≤5); "
        "không phải 100 review mới nhất."
    ),
    "tripadvisor": (
        "TripAdvisor: tối đa 100 review mới nhất theo ngày đăng "
        "(hoặc ít hơn nếu nguồn chưa đủ); Score trang vẫn hiện."
    ),
}


class PlacesService:
    def __init__(
        self,
        store: PgVectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.store = store or get_vector_store()
        self.settings = settings or get_settings()

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
        # TripAdvisor first in evidence payload so the model leads with it.
        order = {"tripadvisor": 0, "chudu24": 1, "google": 2}
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
        """Retrieve up to rag_per_source_k quotes per primary source, then cap.

        Per-source RPC filters avoid VI queries drowning TripAdvisor (EN).
        Request ``top_k`` is a total cap after merge, not a substitute for
        per-source quota.
        """
        ids = list(place_ids or [])
        if place_id:
            ids.append(
                UUID(str(place_id)) if not isinstance(place_id, UUID) else place_id
            )

        per_k = max(1, int(self.settings.rag_per_source_k))
        merged: list[QuoteOut] = []
        seen_ids: set[str] = set()

        if ids:
            for pid in ids:
                for source in _PRIMARY_SOURCES:
                    for chunk in await self._search_source_chunks(
                        query, per_k, place_id=pid, source=source
                    ):
                        key = chunk.id or f"{chunk.source}:{chunk.content[:40]}"
                        if key in seen_ids:
                            continue
                        seen_ids.add(key)
                        merged.append(self._quote_from_chunk(chunk.__dict__))
        else:
            for source in _PRIMARY_SOURCES:
                for chunk in await self._search_source_chunks(
                    query, per_k, place_id=None, source=source
                ):
                    key = chunk.id or f"{chunk.source}:{chunk.content[:40]}"
                    if key in seen_ids:
                        continue
                    seen_ids.add(key)
                    merged.append(self._quote_from_chunk(chunk.__dict__))

        out = _diversify_quotes(merged, top_k)
        by_source: dict[str, int] = {}
        for q in out:
            src = q.source or "unknown"
            by_source[src] = by_source.get(src, 0) + 1
        logger.info(
            "quotes_by_source=%s place_ids=%s top_k=%s per_source_k=%s pool=%d",
            by_source,
            [str(i) for i in ids] if ids else None,
            top_k,
            per_k,
            len(merged),
        )
        return out

    async def _search_source_chunks(
        self,
        query: str,
        per_k: int,
        *,
        place_id: UUID | None,
        source: str,
    ) -> list[DocumentChunk]:
        """Fetch up to per_k chunks for one source; EN booster for TripAdvisor."""
        queries = [query]
        if source == "tripadvisor" and any(ord(ch) > 127 for ch in (query or "")):
            queries.append(_TA_EN_BOOSTER)

        bucket: list[DocumentChunk] = []
        seen: set[str] = set()
        for q in queries:
            chunks = await self.store.similarity_search(
                q,
                top_k=per_k,
                place_id=place_id,
                source=source,
            )
            for chunk in chunks:
                key = chunk.id or f"{chunk.source}:{chunk.content[:40]}"
                if key in seen:
                    continue
                seen.add(key)
                bucket.append(chunk)

        bucket.sort(key=lambda c: c.similarity or 0.0, reverse=True)
        return bucket[:per_k]

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


_SOURCE_RANK = {"tripadvisor": 0, "chudu24": 1, "google": 2}
# Clear balance when both corpora exist (then fill remaining slots).
_PRIMARY_SOURCES = ("tripadvisor", "chudu24")
_PRIMARY_QUOTA = 2


def _mix_scores_within_source(items: list[QuoteOut]) -> list[QuoteOut]:
    """Interleave high / low scores so highlights are not all 5★."""
    high = [q for q in items if (q.score or 3) >= 4]
    low = [q for q in items if (q.score or 3) < 3.5]
    mid = [q for q in items if q not in high and q not in low]
    mixed: list[QuoteOut] = []
    while high or low or mid:
        if high:
            mixed.append(high.pop(0))
        if low:
            mixed.append(low.pop(0))
        if mid:
            mixed.append(mid.pop(0))
    return mixed


def _diversify_quotes(quotes: list[QuoteOut], top_k: int) -> list[QuoteOut]:
    """Balance quotes: up to 2 TripAdvisor + 2 Chudu24, then fill.

    VI queries otherwise drown TA under Chudu24 by similarity. Always
    reorders even when ``len(quotes) <= top_k``. Missing source → fill
    from the other.
    """
    if not quotes or top_k <= 0:
        return []

    by_source: dict[str, list[QuoteOut]] = {}
    for q in quotes:
        by_source.setdefault(q.source or "unknown", []).append(q)

    for source, items in by_source.items():
        by_source[source] = _mix_scores_within_source(items)

    cursors = {s: 0 for s in by_source}
    out: list[QuoteOut] = []

    # Pass 1: interleave up to 2 from each primary source.
    for _ in range(_PRIMARY_QUOTA):
        for source in _PRIMARY_SOURCES:
            if len(out) >= top_k:
                break
            bucket = by_source.get(source) or []
            i = cursors.get(source, 0)
            if i < len(bucket):
                out.append(bucket[i])
                cursors[source] = i + 1

    # Pass 2: round-robin remaining slots (TA → Chudu24 → Google → …).
    source_order = sorted(
        by_source.keys(),
        key=lambda s: _SOURCE_RANK.get(s, 9),
    )
    while len(out) < top_k:
        progressed = False
        for source in source_order:
            i = cursors.get(source, 0)
            bucket = by_source[source]
            if i < len(bucket):
                out.append(bucket[i])
                cursors[source] = i + 1
                progressed = True
                if len(out) >= top_k:
                    break
        if not progressed:
            break

    return out[:top_k]


def get_places_service() -> PlacesService:
    return PlacesService()
