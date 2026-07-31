import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from domains.places.stats import compute_sample_stats, contrast_site_overall, score_to_five
from datetime import date


def test_score_to_five():
    assert score_to_five(10, 10) == 5.0
    assert score_to_five(4, 5) == 4.0


def test_compute_sample_stats():
    stats = compute_sample_stats(
        [
            {"review_date": date(2026, 1, 1), "score": 10, "score_scale": 10},
            {"review_date": date(2025, 6, 1), "score": 6, "score_scale": 10},
        ]
    )
    assert stats["sample_size"] == 2
    assert stats["date_min"] == "2025-06-01"
    assert stats["date_max"] == "2026-01-01"
    assert stats["sample_mean"] == 4.0


def test_contrast():
    c = contrast_site_overall(
        [
            {"site_overall": 4.5, "site_overall_scale": 5},
            {"site_overall": 3.8, "site_overall_scale": 5},
        ]
    )
    assert c["site_overall_spread"] == 0.7
    assert "rating_disagreement_between_sources" in c["flags"]
