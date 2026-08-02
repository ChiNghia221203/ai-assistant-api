"""Match hotel names / districts mentioned in a user message to known places."""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domains.places.schemas import PlaceOut

_STOP = frozenset(
    {
        "hotel",
        "hotels",
        "khach",
        "san",
        "and",
        "apartments",
        "apartment",
        "the",
        "tai",
        "tp",
        "ho",
        "chi",
        "minh",
        "saigon",
        "sai",
        "gon",
        "city",
        "review",
        "reviews",
        "muon",
        "ve",
        "toi",
        "cho",
        "cac",
        "nguon",
    }
)

# Folded aliases → match against folded place.address / name / city.
_DISTRICT_ALIASES: dict[str, tuple[str, ...]] = {
    "tan binh": ("tan binh", "quan tan binh", "district tan binh"),
    "binh thanh": ("binh thanh", "quan binh thanh", "district binh thanh"),
    "phu nhuan": ("phu nhuan", "quan phu nhuan", "district phu nhuan"),
    "quan 1": ("quan 1", "district 1", " q1 ", "q.1"),
    "quan 3": ("quan 3", "district 3", " q3 ", "q.3"),
    "quan 5": ("quan 5", "district 5", " q5 ", "q.5"),
    "quan 7": ("quan 7", "district 7", " q7 ", "q.7"),
    "go vap": ("go vap", "quan go vap", "district go vap"),
    "thu duc": ("thu duc", "tp thu duc", "thu duc city"),
}

_AREA_RECOMMEND_RE = re.compile(
    r"("
    r"chọn\s*khách\s*sạn\s*nào|khách\s*sạn\s*nào|nen\s*chon\s*khach\s*san"
    r"|gợi\s*ý\s*khách\s*sạn|goi\s*y\s*khach\s*san|recommend\s*(a\s*)?hotel"
    r"|khách\s*sạn\s*(ở|tai|tại|gần|trong)"
    r"|trong\s*(khu\s*vực|phạm\s*vi|quan|quận)"
    r"|phạm\s*vi\s*(tân|tan|quận|quan|q\d)"
    r")",
    re.IGNORECASE,
)


def fold_text(value: str) -> str:
    """Lowercase ASCII fold for loose Vietnamese / English matching."""
    text = unicodedata.normalize("NFD", value.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def place_core_name(name: str) -> str:
    """Strip leading catalog numbers like '36. ' then fold."""
    core = fold_text(name)
    return re.sub(r"^\d+\s+", "", core).strip()


def is_area_recommend_intent(message: str) -> bool:
    """User asks which / nearby hotel in a district or area — still in-scope."""
    return bool(_AREA_RECOMMEND_RE.search(message or ""))


def extract_area_aliases(message: str) -> list[str]:
    """Return folded district alias strings found in the message."""
    msg = f" {fold_text(message)} "
    found: list[str] = []
    for aliases in _DISTRICT_ALIASES.values():
        if any(f" {a.strip()} " in msg or a in msg for a in aliases):
            found.extend(aliases)
    return found


def match_places_in_area(
    message: str,
    places: list[PlaceOut],
    *,
    limit: int = 5,
) -> list[PlaceOut]:
    """Places whose address/name/city mention a district from the message."""
    aliases = extract_area_aliases(message)
    if not aliases or not places:
        return []

    out: list[PlaceOut] = []
    for place in places:
        blob = fold_text(
            f"{place.name} {place.address or ''} {place.city or ''}"
        )
        if any(a.strip() in blob for a in aliases):
            out.append(place)
        if len(out) >= limit:
            break
    return out


def match_places_in_message(
    message: str,
    places: list[PlaceOut],
    *,
    limit: int = 2,
) -> list[PlaceOut]:
    """Return places whose names appear in `message`, strongest first.

    Used so a selected sidebar place (e.g. auto-picked Lancaster) does not
    override an explicit hotel name in the question (e.g. Park Hyatt).
    """
    msg = fold_text(message)
    if not msg or not places:
        return []

    scored: list[tuple[int, PlaceOut]] = []
    for place in places:
        core = place_core_name(place.name)
        if len(core) < 4:
            continue
        if core in msg:
            scored.append((1000 + len(core), place))
            continue
        tokens = [t for t in core.split() if t not in _STOP and len(t) > 2]
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in msg.split() or t in msg)
        need = 2 if len(tokens) >= 2 else 1
        if hits >= need:
            scored.append((hits * 50 + len(core), place))

    scored.sort(key=lambda item: item[0], reverse=True)
    out: list[PlaceOut] = []
    seen: set[str] = set()
    for _, place in scored:
        key = str(place.id)
        if key in seen:
            continue
        seen.add(key)
        out.append(place)
        if len(out) >= limit:
            break
    return out
