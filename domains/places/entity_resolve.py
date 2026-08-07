"""Entity resolution for review chat: parse → bind → retrieval policy.

Chat must not invent place membership from global RAG. All place/area routing
goes through EntityResolver so new edge cases plug in here without scattering
regex in review_chat.

Explicit 2-hotel compare (labels → catalog hit/miss → rag/hybrid/web) lives
here; IntentExtract is primary for labels, short regex is fallback only.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from domains.places.geo_lexicon import district_label
from domains.places.place_match import (
    brand_candidate_places,
    district_keys_in_message,
    is_area_recommend_intent,
    is_brand_ambiguous_named,
    match_places_in_area,
    match_places_in_message,
)
from domains.places.retrieval_gate import (
    extract_specific_hotel_span,
    is_ambiguous_entity,
    looks_like_specific_hotel_ask,
)
from domains.places.schemas import PlaceOut

if TYPE_CHECKING:
    from domains.places.intent_extract import IntentExtract

logger = logging.getLogger(__name__)

MAX_COMPARE = 2

_COMPARE_INTENT_RE = re.compile(
    r"(so\s*sánh|đối\s*chiếu|khác\s*nhau|hơn\s*kém|cái\s*nào\s*hơn|vs\.?|versus|"
    r"tốt\s*hơn|đáng\s*ở\s*hơn|nào\s*.{0,40}hơn|chọn\s*cái\s*nào|cái\s*nào\s*ổn)",
    re.IGNORECASE,
)

# Fallback pair extract when IntentExtract has no hotels (LLM is primary).
_COMPARE_PAIR_RE = re.compile(
    r"(?:so\s*sánh|đối\s*chiếu)\s+"
    r"(?:(?:khách\s*sạn|khach\s*san|\bks\b|\bhotel\b)\s+)?"
    r"(?P<a>.+?)\s+(?:và|va|vs\.?|versus|với|voi)\s+"
    r"(?:(?:khách\s*sạn|khach\s*san|\bks\b|\bhotel\b)\s+)?"
    r"(?P<b>.+?)"
    r"(?=\s+về\b|\s+ve\b|\s+theo\b|\s+ở\b|\s+o\b|[?.!]|$)",
    re.IGNORECASE,
)

_VS_PAIR_RE = re.compile(
    r"(?:(?:khách\s*sạn|khach\s*san|\bks\b|\bhotel\b)\s+)?"
    r"(?P<a>[A-Za-zÀ-ỹ0-9][\wÀ-ỹ\.\-]*(?:\s+[A-Za-zÀ-ỹ0-9][\wÀ-ỹ\.\-]*){0,5})"
    r"\s+(?:vs\.?|versus)\s+"
    r"(?:(?:khách\s*sạn|khach\s*san|\bks\b|\bhotel\b)\s+)?"
    r"(?P<b>[A-Za-zÀ-ỹ0-9][\wÀ-ỹ\.\-]*(?:\s+[A-Za-zÀ-ỹ0-9][\wÀ-ỹ\.\-]*){0,5})",
    re.IGNORECASE,
)

_TRAILING_CRITERIA_RE = re.compile(
    r"\s+(về|ve|theo|ở|o|tại|tai|thì)\s+.+$",
    re.IGNORECASE,
)


def is_explicit_compare_ask(message: str) -> bool:
    return bool(_COMPARE_INTENT_RE.search(message or ""))


def _clean_compare_label(raw: str) -> str:
    text = (raw or "").strip(" \t\n\r,;:.-")
    text = _TRAILING_CRITERIA_RE.sub("", text).strip()
    text = re.sub(
        r"^(?:khách\s*sạn|khach\s*san|\bks\b|\bhotel\b)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(
        r"\s+(quận|quan|q\.?|district)\s*\d*\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text


def extract_compare_labels(message: str) -> list[str]:
    """Rule-based labels (fallback). Prefer IntentExtract hotels when available."""
    text = message or ""
    for pattern in (_COMPARE_PAIR_RE, _VS_PAIR_RE):
        match = pattern.search(text)
        if not match:
            continue
        labels = [
            _clean_compare_label(match.group("a")),
            _clean_compare_label(match.group("b")),
        ]
        labels = [x for x in labels if len(x) >= 2]
        if len(labels) >= 2:
            return labels[:MAX_COMPARE]
    return []


def _compare_kind(n_hit: int, n_miss: int) -> str:
    if n_hit >= 2 and n_miss == 0:
        return "rag_compare"
    if n_hit >= 1 and n_miss >= 1:
        return "hybrid_compare"
    if n_hit == 0 and n_miss >= 2:
        return "web_compare"
    return "incomplete"


@dataclass(slots=True)
class CompareBindResult:
    """Slim compare outcome (no per-side DTO)."""

    corpus_places: list[PlaceOut] = field(default_factory=list)
    reference_labels: list[str] = field(default_factory=list)
    kind: str = "incomplete"


def bind_compare(
    message: str,
    catalog: list[PlaceOut],
    *,
    intent: IntentExtract | None = None,
) -> CompareBindResult | None:
    """Bind a 2-hotel compare. Uses LLM intent hotels when present."""
    labels: list[str] = []
    wants = is_explicit_compare_ask(message)

    if intent is not None:
        if intent.wants_compare:
            wants = True
        if len(intent.hotel_mentions) >= 2:
            labels = [_clean_compare_label(h) for h in intent.hotel_mentions]
            labels = [x for x in labels if len(x) >= 2][:MAX_COMPARE]

    if len(labels) < 2:
        labels = extract_compare_labels(message)

    if not wants or len(labels) < 2:
        return None

    corpus: list[PlaceOut] = []
    refs: list[str] = []
    used_ids: set[str] = set()
    for label in labels[:MAX_COMPARE]:
        hits = match_places_in_message(label, catalog, limit=1)
        place = hits[0] if hits else None
        if place is not None and str(place.id) in used_ids:
            return None
        if place is not None:
            used_ids.add(str(place.id))
            corpus.append(place)
        else:
            refs.append(label)

    kind = _compare_kind(len(corpus), len(refs))
    if kind == "incomplete":
        return None
    return CompareBindResult(
        corpus_places=corpus,
        reference_labels=refs,
        kind=kind,
    )


@dataclass(frozen=True, slots=True)
class EntityHints:
    """Optional structured hints (rules today; LLM extract can fill later)."""

    district_keys: tuple[str, ...] = ()
    hotel_name_spans: tuple[str, ...] = ()
    area_recommend: bool | None = None


class EntityHintProvider(Protocol):
    def hints(self, message: str) -> EntityHints: ...


class RulesEntityHintProvider:
    """Default parser — cheap, deterministic, testable."""

    def hints(self, message: str) -> EntityHints:
        keys = tuple(district_keys_in_message(message))
        area = is_area_recommend_intent(message)
        return EntityHints(
            district_keys=keys,
            hotel_name_spans=(),
            area_recommend=area if area else None,
        )


@dataclass(slots=True)
class EntityResolveResult:
    """Outcome of binding a user message to catalog places + RAG constraints."""

    place_ids: list[UUID] = field(default_factory=list)
    named: list[PlaceOut] = field(default_factory=list)
    area_places: list[PlaceOut] = field(default_factory=list)
    district_keys: list[str] = field(default_factory=list)
    area_label: str | None = None
    hotel_label: str | None = None
    # Explicit compare: hotels not in catalog → web grounding (never peer-swap).
    reference_hotels: list[str] = field(default_factory=list)
    area_recommend: bool = False
    used_conversation_place_ids: bool = False
    allow_global_rag: bool = True
    is_ambiguous: bool = False
    # named | area | seed | conversation | none | unresolved_* |
    # ambiguous_name | rag_compare | hybrid_compare | web_compare
    source: str = "none"
    confidence: float = 0.0

    @property
    def named_matched(self) -> bool:
        return bool(self.named)

    @property
    def needs_web_compare(self) -> bool:
        return self.source in {"hybrid_compare", "web_compare"}


class EntityResolver:
    """Parse entities from the message, bind to catalog, emit RAG policy."""

    def __init__(
        self,
        hint_provider: EntityHintProvider | None = None,
        *,
        area_limit: int = 5,
        named_limit: int = 2,
    ) -> None:
        self.hints = hint_provider or RulesEntityHintProvider()
        self.area_limit = area_limit
        self.named_limit = named_limit

    def resolve(
        self,
        message: str,
        catalog: list[PlaceOut],
        *,
        seed_place_ids: list[UUID] | None = None,
        conversation_place_ids: list[UUID] | None = None,
        intent: IntentExtract | None = None,
    ) -> EntityResolveResult:
        hint = self.hints.hints(message)
        specific = looks_like_specific_hotel_ask(message)
        hotel_label = extract_specific_hotel_span(message) if specific else None
        if (
            not hotel_label
            and intent is not None
            and len(intent.hotel_mentions) == 1
            and not intent.wants_compare
        ):
            hotel_label = intent.hotel_mentions[0].strip() or None
            if hotel_label:
                specific = True
        raw_area = (
            hint.area_recommend
            if hint.area_recommend is not None
            else is_area_recommend_intent(message)
        )
        district_keys = list(hint.district_keys) or district_keys_in_message(message)
        if intent is not None and intent.area_mentions and not district_keys:
            # Soft: area strings from extract; district_keys_in_message re-scan
            # still preferred when lexicon hits.
            district_keys = district_keys_in_message(
                " ".join(intent.area_mentions)
            ) or district_keys
        label = district_label(district_keys[0]) if district_keys else None

        # --- Explicit 2-hotel compare (before single-hotel / area paths) ---
        compare = bind_compare(message, catalog, intent=intent)
        if compare is not None and compare.kind != "incomplete":
            return self._from_compare(compare, district_keys, label)

        # Named-hotel ask (with optional district qualifier) is NEVER area recommend.
        area_recommend = bool(raw_area) and not specific

        named = match_places_in_message(
            message, catalog, limit=max(self.named_limit, 5)
        )
        if not named and hotel_label:
            named = match_places_in_message(
                hotel_label, catalog, limit=max(self.named_limit, 5)
            )
        # Also try intent hotel labels when message match is empty.
        if (
            not named
            and intent is not None
            and intent.hotel_mentions
            and not intent.wants_compare
        ):
            named = match_places_in_message(
                intent.hotel_mentions[0], catalog, limit=max(self.named_limit, 5)
            )
        # Brand-token scan catches short queries like "Sheraton" across branches.
        if len(named) < 2:
            brand_hits = brand_candidate_places(message, catalog, limit=8)
            if len(brand_hits) >= 2:
                named = brand_hits
            elif not named and len(brand_hits) == 1:
                named = brand_hits

        area_places: list[PlaceOut] = []
        if area_recommend:
            area_places = match_places_in_area(
                message, catalog, limit=self.area_limit
            )

        seed = list(seed_place_ids or [])
        conversation = list(conversation_place_ids or [])
        wants_compare = bool(intent.wants_compare) if intent is not None else False

        # Same brand, multiple branches → ask user; do not auto-pick N hotels.
        if named and is_brand_ambiguous_named(
            message, named, wants_compare=wants_compare
        ):
            brand = hotel_label or (
                intent.hotel_mentions[0]
                if intent and intent.hotel_mentions
                else named[0].name
            )
            result = EntityResolveResult(
                place_ids=[],
                named=list(named),
                area_places=[],
                district_keys=district_keys,
                area_label=label,
                hotel_label=brand,
                area_recommend=False,
                used_conversation_place_ids=False,
                allow_global_rag=False,
                is_ambiguous=True,
                source="ambiguous_name",
                confidence=0.88,
            )
            logger.info(
                "EntityResolve source=%s places=%d refs=%s ambiguous=%s "
                "candidates=%s global_rag=%s",
                result.source,
                len(result.place_ids),
                result.reference_hotels,
                result.is_ambiguous,
                [p.name for p in result.named],
                result.allow_global_rag,
            )
            return result

        # Cap named hits for normal single/multi path after ambiguity check.
        named = named[: self.named_limit]

        if named:
            result = EntityResolveResult(
                place_ids=[p.id for p in named],
                named=named,
                area_places=area_places,
                district_keys=district_keys,
                area_label=label,
                hotel_label=hotel_label,
                area_recommend=False,
                used_conversation_place_ids=False,
                allow_global_rag=False,
                is_ambiguous=False,
                source="named",
                confidence=0.95,
            )
        elif specific:
            result = EntityResolveResult(
                place_ids=[],
                named=[],
                area_places=[],
                district_keys=district_keys,
                area_label=label,
                hotel_label=hotel_label,
                area_recommend=False,
                used_conversation_place_ids=False,
                allow_global_rag=False,
                is_ambiguous=True,
                source="unresolved_hotel",
                confidence=0.9,
            )
        elif area_recommend:
            if area_places:
                result = EntityResolveResult(
                    place_ids=[p.id for p in area_places],
                    named=[],
                    area_places=area_places,
                    district_keys=district_keys,
                    area_label=label,
                    hotel_label=None,
                    area_recommend=True,
                    used_conversation_place_ids=False,
                    allow_global_rag=False,
                    is_ambiguous=False,
                    source="area",
                    confidence=0.9,
                )
            else:
                result = EntityResolveResult(
                    place_ids=[],
                    named=[],
                    area_places=[],
                    district_keys=district_keys,
                    area_label=label,
                    hotel_label=None,
                    area_recommend=True,
                    used_conversation_place_ids=False,
                    allow_global_rag=False,
                    is_ambiguous=True,
                    source="unresolved_area",
                    confidence=0.85,
                )
        elif seed:
            result = EntityResolveResult(
                place_ids=seed,
                named=[],
                area_places=[],
                district_keys=district_keys,
                area_label=label,
                hotel_label=hotel_label,
                area_recommend=False,
                used_conversation_place_ids=False,
                allow_global_rag=False,
                is_ambiguous=False,
                source="seed",
                confidence=0.8,
            )
        elif conversation:
            result = EntityResolveResult(
                place_ids=conversation,
                named=[],
                area_places=[],
                district_keys=district_keys,
                area_label=label,
                hotel_label=hotel_label,
                area_recommend=False,
                used_conversation_place_ids=True,
                allow_global_rag=False,
                is_ambiguous=False,
                source="conversation",
                confidence=0.7,
            )
        else:
            result = EntityResolveResult(
                place_ids=[],
                named=[],
                area_places=[],
                district_keys=district_keys,
                area_label=label,
                hotel_label=hotel_label,
                area_recommend=False,
                used_conversation_place_ids=False,
                allow_global_rag=True,
                is_ambiguous=False,
                source="none",
                confidence=0.3,
            )

        if not result.is_ambiguous:
            result.is_ambiguous = is_ambiguous_entity(
                message,
                named_matched=result.named_matched,
                area_places_found=bool(result.area_places),
                used_conversation_place_ids=result.used_conversation_place_ids,
            )
            if result.is_ambiguous:
                result.allow_global_rag = False
                if result.source in {"none", "area"} and specific:
                    result.source = "unresolved_hotel"
                    result.place_ids = []
                    result.area_places = []
                    result.area_recommend = False
                elif result.is_ambiguous and result.area_recommend:
                    if result.source == "none":
                        result.source = "unresolved_area"

        logger.info(
            "EntityResolve source=%s places=%d refs=%s ambiguous=%s global_rag=%s",
            result.source,
            len(result.place_ids),
            result.reference_hotels,
            result.is_ambiguous,
            result.allow_global_rag,
        )
        return result

    def _from_compare(
        self,
        compare,
        district_keys: list[str],
        area_label: str | None,
    ) -> EntityResolveResult:
        corpus = compare.corpus_places
        refs = compare.reference_labels
        kind = compare.kind
        logger.info(
            "EntityResolve compare kind=%s corpus=%s refs=%s",
            kind,
            [p.name for p in corpus],
            refs,
        )
        return EntityResolveResult(
            place_ids=[p.id for p in corpus],
            named=list(corpus),
            area_places=[],
            district_keys=district_keys,
            area_label=area_label,
            hotel_label=refs[0] if refs and not corpus else None,
            reference_hotels=list(refs),
            area_recommend=False,
            used_conversation_place_ids=False,
            allow_global_rag=False,
            is_ambiguous=False,
            source=kind,
            confidence=0.92,
        )
