"""Low-level string matchers for hotel names and districts.

Prefer EntityResolver (entity_resolve.py) for chat routing. This module stays
pure and side-effect free so tests / index builders can reuse it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from domains.places.geo_lexicon import DISTRICTS, district_aliases

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

# Hotel noun including common abbreviations (KS / ks).
_HOTEL_NOUN = r"(?:khách\s*sạn|khach\s*san|\bks\b|\bhotel\b|\bresort\b)"

_AREA_RECOMMEND_RE = re.compile(
    r"("
    rf"chọn\s*{_HOTEL_NOUN}\s*nào"
    rf"|nen\s*chon\s*{_HOTEL_NOUN}\s*nao"
    rf"|{_HOTEL_NOUN}\s*nào"
    rf"|{_HOTEL_NOUN}\s*nao"
    rf"|gợi\s*ý\s*{_HOTEL_NOUN}|goi\s*y\s*{_HOTEL_NOUN}|recommend\s*(a\s*)?hotel"
    rf"|{_HOTEL_NOUN}\s*(ở|o|tai|tại|gần|gan|trong)"
    rf"|nên\s*chọn\s*{_HOTEL_NOUN}|nen\s*chon\s*{_HOTEL_NOUN}"
    r"|trong\s*(khu\s*vực|khu\s*vuc|phạm\s*vi|pham\s*vi|quan|quận)"
    r"|phạm\s*vi\s*|pham\s*vi\s*"
    r"|ở\s*(quận|quan|phường|phuong)|o\s*(quan|quận)"
    r"|tại\s*(quận|quan)|tai\s*(quan|quận)"
    r")",
    re.IGNORECASE,
)

_HOTEL_CUE_RE = re.compile(
    rf"({_HOTEL_NOUN}|chọn|chon|gợi\s*ý|goi\s*y|nên|nen|nào|nao)",
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


def alias_hits_text(folded: str, aliases: tuple[str, ...]) -> bool:
    """True if any alias appears in folded text (phrase or compact form)."""
    if not folded:
        return False
    padded = f" {folded} "
    compact = folded.replace(" ", "")
    for raw in aliases:
        alias = raw.strip()
        if not alias:
            continue
        if f" {alias} " in padded:
            return True
        # Compact forms: "qtb", "tanbinh", "q1"
        ac = alias.replace(" ", "")
        if len(ac) >= 2 and ac in compact:
            # Guard ultra-short tokens (e.g. lone "q") — require digit or len>=3
            if len(ac) >= 3 or any(ch.isdigit() for ch in ac):
                return True
    return False


def district_keys_in_text(folded: str) -> list[str]:
    """Canonical district keys found in already-folded text."""
    found: list[str] = []
    for entry in DISTRICTS:
        if alias_hits_text(folded, entry.aliases):
            found.append(entry.key)
    return found


def district_keys_in_message(message: str) -> list[str]:
    """Canonical district keys mentioned in the message (e.g. 'tan binh')."""
    return district_keys_in_text(fold_text(message or ""))


def is_area_recommend_intent(message: str) -> bool:
    """User asks which / nearby hotel in a district or area — still in-scope."""
    text = message or ""
    if _AREA_RECOMMEND_RE.search(text):
        return True
    if district_keys_in_message(text) and _HOTEL_CUE_RE.search(text):
        return True
    return False


def extract_area_aliases(message: str) -> list[str]:
    """Return folded district alias strings for districts found in the message."""
    found: list[str] = []
    for key in district_keys_in_message(message):
        found.extend(district_aliases(key))
    return found


def infer_place_district_keys(place: PlaceOut) -> list[str]:
    """Districts inferred from place address/city only — never from hotel name.

    Name fallback caused false "Tân Bình" membership; empty address → no district.
    """
    blob = fold_text(f"{place.address or ''} {place.city or ''}")
    if not blob.strip():
        return []
    return district_keys_in_text(blob)


def match_places_in_area(
    message: str,
    places: list[PlaceOut],
    *,
    limit: int = 5,
) -> list[PlaceOut]:
    """Places whose address/city map to a district mentioned in the message."""
    keys = set(district_keys_in_message(message))
    if not keys or not places:
        return []

    out: list[PlaceOut] = []
    for place in places:
        place_keys = set(infer_place_district_keys(place))
        if place_keys & keys:
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

    Contiguous multi-token phrases (e.g. \"park hyatt\") outrank lone shared
    tokens (e.g. Wink … by Hyatt matching only \"hyatt\"). Near-ties at the top
    stay; weak secondary hits are dropped.
    """
    msg = fold_text(message)
    if not msg or not places:
        return []
    msg_words = set(msg.split())

    scored: list[tuple[int, PlaceOut]] = []
    for place in places:
        core = place_core_name(place.name)
        if len(core) < 4:
            continue
        if core in msg:
            scored.append((2000 + len(core), place))
            continue
        tokens = [t for t in core.split() if t not in _STOP and len(t) > 2]
        if not tokens:
            continue
        hits = sum(1 for t in tokens if t in msg_words)
        # Contiguous bigrams present in both message and place name.
        phrase_bonus = 0
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            if bigram in msg and bigram in core:
                phrase_bonus += 500
        strong_hit = any(len(t) >= 5 and t in msg_words for t in tokens)
        need = 2 if len(tokens) >= 2 else 1
        if phrase_bonus:
            scored.append((phrase_bonus + hits * 50 + len(core), place))
        elif hits >= need or (strong_hit and hits >= 1):
            scored.append((hits * 50 + len(core) + (20 if strong_hit else 0), place))

    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0][0]
    # Keep only near-ties with the best score (same brand branches), drop weak
    # secondary hits like \"by Hyatt\" when \"Park Hyatt\" already won.
    cluster = [(s, p) for s, p in scored if s >= best - 80]
    out: list[PlaceOut] = []
    seen: set[str] = set()
    for _, place in cluster:
        key = str(place.id)
        if key in seen:
            continue
        seen.add(key)
        out.append(place)
        if len(out) >= limit:
            break
    return out


def brand_candidate_places(
    message: str,
    places: list[PlaceOut],
    *,
    limit: int = 8,
) -> list[PlaceOut]:
    """Places whose *leading* brand matches a distinctive cue from the message.

    Avoids false hits on trailing parent-brand words (\"… by Hyatt\").
    """
    msg = fold_text(message)
    words = [t for t in msg.split() if t not in _STOP and len(t) > 2]
    if not words or not places:
        return []
    # Prefer multi-word brand phrases from the user message.
    phrases: list[str] = []
    for i in range(len(words) - 1):
        phrases.append(f"{words[i]} {words[i + 1]}")
    phrases.extend([w for w in words if len(w) >= 5])

    out: list[PlaceOut] = []
    seen: set[str] = set()
    for phrase in phrases:
        for place in places:
            core = place_core_name(place.name)
            # Phrase / cue must sit at the start of the place brand, not \"by X\".
            if phrase in core:
                # Reject mid/tail-only: require match near the beginning.
                idx = core.find(phrase)
                if idx < 0 or idx > 12:
                    continue
            else:
                # Unigram: must be among leading content tokens.
                if " " in phrase:
                    continue
                lead = [
                    t for t in core.split() if t not in _STOP and len(t) > 2
                ][:2]
                if phrase not in lead:
                    continue
            key = str(place.id)
            if key in seen:
                continue
            seen.add(key)
            out.append(place)
            if len(out) >= limit:
                return out
        # If a multi-word phrase already found ≥1 place, stop (don't widen to
        # weak unigram \"hyatt\" and pull in unrelated Hyatt soft-brands).
        if " " in phrase and out:
            return out
    return out


def shared_brand_token(places: list[PlaceOut], message: str) -> str | None:
    """Longest *leading* brand token shared by all places and present in message.

    \"Sheraton A\" + \"Sheraton B\" → sheraton. \"Park Hyatt\" + \"Wink by Hyatt\" → None.
    """
    if len(places) < 2:
        return None
    msg = fold_text(message)
    msg_tokens = set(msg.split())
    lead_sets: list[set[str]] = []
    for place in places:
        core = place_core_name(place.name)
        lead = [
            t for t in core.split() if t not in _STOP and len(t) >= 4
        ][:2]
        if not lead:
            return None
        lead_sets.append(set(lead))
    shared = set.intersection(*lead_sets)
    in_msg = [t for t in shared if t in msg_tokens or t in msg]
    if not in_msg:
        return None
    return max(in_msg, key=len)


def is_brand_ambiguous_named(
    message: str,
    named: list[PlaceOut],
    *,
    wants_compare: bool,
) -> bool:
    """True when ≥2 catalog hits look like one brand, not an explicit multi-hotel ask."""
    if wants_compare or len(named) < 2:
        return False
    return shared_brand_token(named, message) is not None
