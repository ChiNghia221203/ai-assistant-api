"""HCMC geo lexicon — single source of truth for district keys / aliases / labels.

Add a new district by appending a DistrictEntry. Matching uses folded text
(see place_match.fold_text). Keep aliases explicit; do not invent membership
from RAG similarity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DistrictEntry:
    """Canonical district key used across parse → bind → user-facing labels."""

    key: str
    label: str
    # Folded aliases (ASCII, no diacritics). Prefer phrases over bare initials.
    aliases: tuple[str, ...]


# Order matters for display when multiple districts appear; matching is independent.
DISTRICTS: tuple[DistrictEntry, ...] = (
    DistrictEntry(
        key="tan binh",
        label="Tân Bình",
        aliases=(
            "tan binh",
            "quan tan binh",
            "district tan binh",
            "q tan binh",
            "q tb",
            "qtb",
            "tanbinh",
        ),
    ),
    DistrictEntry(
        key="binh thanh",
        label="Bình Thạnh",
        aliases=(
            "binh thanh",
            "quan binh thanh",
            "district binh thanh",
            "q binh thanh",
            "binhthanh",
        ),
    ),
    DistrictEntry(
        key="phu nhuan",
        label="Phú Nhuận",
        aliases=(
            "phu nhuan",
            "quan phu nhuan",
            "district phu nhuan",
            "q phu nhuan",
            "q pn",
            "phunhuan",
        ),
    ),
    DistrictEntry(
        key="quan 1",
        label="Quận 1",
        aliases=("quan 1", "district 1", "q 1", "q1", "d1", "quan1"),
    ),
    DistrictEntry(
        key="quan 3",
        label="Quận 3",
        aliases=("quan 3", "district 3", "q 3", "q3", "d3", "quan3"),
    ),
    DistrictEntry(
        key="quan 5",
        label="Quận 5",
        aliases=("quan 5", "district 5", "q 5", "q5", "d5", "quan5"),
    ),
    DistrictEntry(
        key="quan 7",
        label="Quận 7",
        aliases=("quan 7", "district 7", "q 7", "q7", "d7", "quan7"),
    ),
    DistrictEntry(
        key="go vap",
        label="Gò Vấp",
        aliases=("go vap", "quan go vap", "district go vap", "govap"),
    ),
    DistrictEntry(
        key="thu duc",
        label="Thủ Đức",
        aliases=("thu duc", "tp thu duc", "thu duc city", "thuduc"),
    ),
)

DISTRICT_BY_KEY: dict[str, DistrictEntry] = {d.key: d for d in DISTRICTS}


def district_label(key: str) -> str:
    entry = DISTRICT_BY_KEY.get(key)
    return entry.label if entry else key.title()


def district_aliases(key: str) -> tuple[str, ...]:
    entry = DISTRICT_BY_KEY.get(key)
    return entry.aliases if entry else ()
