"""Google Places API (Place Details) — no HTML scrape."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import httpx

from core.config import Settings, get_settings


@dataclass
class GoogleReview:
    external_id: str
    review_date: date | None
    score: float
    body: str
    author: str | None
    review_url: str | None


@dataclass
class GooglePlaceDetails:
    place_id: str
    name: str | None
    rating: float | None
    user_ratings_total: int | None
    maps_url: str
    address: str | None
    lat: float | None
    lng: float | None
    reviews: list[GoogleReview]


class GooglePlacesClient:
    """Places API (New) Place Details."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.google_places_api_key:
            raise RuntimeError("GOOGLE_PLACES_API_KEY is not set")

    async def get_place_details(self, google_place_id: str) -> GooglePlaceDetails:
        url = (
            f"https://places.googleapis.com/v1/places/{google_place_id}"
        )
        headers = {
            "X-Goog-Api-Key": self.settings.google_places_api_key,
            "X-Goog-FieldMask": (
                "id,displayName,rating,userRatingCount,googleMapsUri,"
                "formattedAddress,location,reviews"
            ),
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        reviews_raw = data.get("reviews") or []
        reviews: list[GoogleReview] = []
        for idx, item in enumerate(reviews_raw):
            text = (item.get("text") or {}).get("text") or item.get("originalText", {}).get("text") or ""
            if not text.strip():
                continue
            publish = item.get("publishTime")
            review_date = None
            if publish:
                try:
                    review_date = datetime.fromisoformat(
                        publish.replace("Z", "+00:00")
                    ).date()
                except ValueError:
                    review_date = None
            reviews.append(
                GoogleReview(
                    external_id=item.get("name") or f"google-{google_place_id}-{idx}",
                    review_date=review_date,
                    score=float(item.get("rating") or 0),
                    body=text.strip(),
                    author=(item.get("authorAttribution") or {}).get("displayName"),
                    review_url=(item.get("authorAttribution") or {}).get("uri"),
                )
            )

        location = data.get("location") or {}
        display = data.get("displayName") or {}
        return GooglePlaceDetails(
            place_id=data.get("id") or google_place_id,
            name=display.get("text"),
            rating=float(data["rating"]) if data.get("rating") is not None else None,
            user_ratings_total=(
                int(data["userRatingCount"])
                if data.get("userRatingCount") is not None
                else None
            ),
            maps_url=data.get("googleMapsUri")
            or f"https://www.google.com/maps/place/?q=place_id:{google_place_id}",
            address=data.get("formattedAddress"),
            lat=location.get("latitude"),
            lng=location.get("longitude"),
            reviews=reviews,
        )


def get_google_places_client() -> GooglePlacesClient:
    return GooglePlacesClient()
