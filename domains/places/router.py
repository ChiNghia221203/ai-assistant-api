from fastapi import APIRouter, Depends, HTTPException, Query

from domains.places.schemas import PlaceEvidenceResponse, PlaceOut
from domains.places.service import PlacesService, get_places_service

router = APIRouter(tags=["Places"])


@router.get(
    "",
    response_model=list[PlaceOut],
    summary="List hotels (default: Ho Chi Minh)",
)
@router.get(
    "/",
    response_model=list[PlaceOut],
    summary="List hotels (default: Ho Chi Minh)",
    include_in_schema=False,
)
def list_places(
    city: str | None = Query(default="Ho Chi Minh"),
    service: PlacesService = Depends(get_places_service),
) -> list[PlaceOut]:
    try:
        return service.list_places(city=city)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/{place_id}",
    response_model=PlaceOut,
    summary="Get one place",
)
def get_place(
    place_id: str,
    service: PlacesService = Depends(get_places_service),
) -> PlaceOut:
    place = service.get_place(place_id)
    if not place:
        raise HTTPException(status_code=404, detail="Place not found")
    return place


@router.get(
    "/{place_id}/evidence",
    response_model=PlaceEvidenceResponse,
    summary="Multi-source evidence card (methodology + scores + contrast)",
)
def get_place_evidence(
    place_id: str,
    service: PlacesService = Depends(get_places_service),
) -> PlaceEvidenceResponse:
    try:
        return service.get_evidence(place_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc
