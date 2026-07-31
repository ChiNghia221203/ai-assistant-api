"""Shared review sample statistics (no LLM)."""

from __future__ import annotations

from datetime import date
from typing import Any


def score_to_five(score: float | None, scale: int | float | None) -> float | None:
    if score is None or scale in (None, 0):
        return None
    return float(score) * (5.0 / float(scale))


def compute_sample_stats(
    reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    reviews items: {review_date: date|None, score: float|None, score_scale: int}
    """
    if not reviews:
        return {
            "sample_size": 0,
            "date_min": None,
            "date_max": None,
            "sample_mean": None,
            "distribution": {"1_2": 0, "3": 0, "4_5": 0},
        }

    dates = [r["review_date"] for r in reviews if r.get("review_date")]
    norms: list[float] = []
    dist = {"1_2": 0, "3": 0, "4_5": 0}
    for r in reviews:
        n = score_to_five(r.get("score"), r.get("score_scale") or 5)
        if n is None:
            continue
        norms.append(n)
        if n < 2.5:
            dist["1_2"] += 1
        elif n < 3.5:
            dist["3"] += 1
        else:
            dist["4_5"] += 1

    return {
        "sample_size": len(reviews),
        "date_min": min(dates).isoformat() if dates else None,
        "date_max": max(dates).isoformat() if dates else None,
        "sample_mean": round(sum(norms) / len(norms), 3) if norms else None,
        "distribution": dist,
    }


def contrast_site_overall(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    for s in snapshots:
        overall = s.get("site_overall")
        scale = s.get("site_overall_scale") or 5
        n = score_to_five(
            float(overall) if overall is not None else None,
            scale,
        )
        if n is not None:
            values.append(n)
    if len(values) < 2:
        return {"site_overall_spread": 0.0, "flags": []}
    spread = round(max(values) - min(values), 3)
    flags = []
    if spread >= 0.4:
        flags.append("rating_disagreement_between_sources")
    return {"site_overall_spread": spread, "flags": flags}


def truncate(text: str, max_len: int = 1200) -> str:
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"
