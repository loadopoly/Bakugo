"""Grading and ratio-arithmetic tests."""

from __future__ import annotations

import math

import pytest

from cardcenter.grading import (
    all_grade_bands,
    available_graders,
    caveat_text,
    grade_band,
    load_standards,
)
from cardcenter.types import BorderPair, Measured


def _pair(low: float, high: float, sigma: float = 0.0) -> BorderPair:
    return BorderPair(
        axis="horizontal",
        low_name="left",
        high_name="right",
        low_mm=Measured(low, sigma),
        high_mm=Measured(high, sigma),
    )


def test_ratio_is_scale_invariant() -> None:
    """Card-size error cancels in a ratio, so it must not appear in the result."""
    a = _pair(3.0, 2.0).ratio_pct.value
    b = _pair(30.0, 20.0).ratio_pct.value
    assert a == pytest.approx(b)
    assert a == pytest.approx(60.0)


def test_ratio_always_reports_the_wider_side() -> None:
    assert _pair(3.0, 2.0).ratio_pct.value == pytest.approx(60.0)
    assert _pair(2.0, 3.0).ratio_pct.value == pytest.approx(60.0)
    assert _pair(3.0, 2.0).skew_toward == "left"
    assert _pair(2.0, 3.0).skew_toward == "right"


def test_perfect_centering_is_fifty() -> None:
    assert _pair(3.0, 3.0).ratio_pct.value == pytest.approx(50.0)


def test_ratio_uncertainty_matches_numeric_propagation() -> None:
    low, high, s = 3.4, 2.6, 0.05
    got = _pair(low, high, s).ratio_pct

    eps = 1e-8

    def f(a: float, b: float) -> float:
        return 100.0 * max(a, b) / (a + b)

    base = f(low, high)
    d_da = (f(low + eps, high) - base) / eps
    d_db = (f(low, high + eps) - base) / eps
    expected = math.hypot(d_da * s, d_db * s)
    assert got.value == pytest.approx(base)
    assert got.sigma == pytest.approx(expected, rel=1e-4)


def test_zero_total_border_degrades_gracefully() -> None:
    r = _pair(0.0, 0.0).ratio_pct
    assert r.value == 50.0
    assert r.sigma == 50.0  # maximally uninformative rather than a false 50/50


def test_measured_interval() -> None:
    m = Measured(60.0, 1.0)
    lo, hi = m.interval(1.96)
    assert lo == pytest.approx(58.04)
    assert hi == pytest.approx(61.96)


@pytest.mark.parametrize("grader", available_graders())
def test_thresholds_are_monotonic(grader: str) -> None:
    """Worse grades must never demand tighter centering than better ones."""
    tiers = load_standards()["graders"][grader]["tiers"]
    for key in ("front_strict", "front_lenient", "back_strict", "back_lenient"):
        vals = [t[key] for t in tiers]
        assert vals == sorted(vals), f"{grader} {key} is not monotonic"


@pytest.mark.parametrize("grader", available_graders())
def test_lenient_never_stricter_than_strict(grader: str) -> None:
    tiers = load_standards()["graders"][grader]["tiers"]
    for t in tiers:
        assert t["front_lenient"] >= t["front_strict"]
        assert t["back_lenient"] >= t["back_strict"]


def test_perfect_card_reaches_top_grade_everywhere() -> None:
    for grader in available_graders():
        band = grade_band(Measured(50.0, 0.05), grader, "front")
        tiers = load_standards()["graders"][grader]["tiers"]
        assert band.best == tiers[0]["grade"]
        assert band.worst == tiers[0]["grade"]


def test_badly_off_centre_card_falls_to_the_bottom() -> None:
    band = grade_band(Measured(97.0, 0.1), "PSA", "front")
    assert band.worst == load_standards()["graders"]["PSA"]["tiers"][-1]["grade"]


def test_band_widens_with_measurement_uncertainty() -> None:
    tight = grade_band(Measured(58.0, 0.05), "PSA", "front")
    loose = grade_band(Measured(58.0, 4.0), "PSA", "front")
    assert loose.measurement_span >= tight.measurement_span
    assert loose.measurement_span > 0


def test_attribution_identifies_standards_ambiguity() -> None:
    """At 57/43 PSA sources disagree (55 vs 60 for a 10). A precise measurement
    there must blame the standards, not itself."""
    band = grade_band(Measured(57.0, 0.05), "PSA", "front")
    assert band.standards_span > 0
    assert band.measurement_span == 0
    assert "standards ambiguity" in band.limited_by


def test_attribution_identifies_measurement_error() -> None:
    band = grade_band(Measured(62.0, 5.0), "PSA", "front")
    assert band.measurement_span > band.standards_span
    assert "measurement uncertainty" in band.limited_by


def test_back_face_is_more_lenient_than_front() -> None:
    """Back centering tolerances are looser at every grader and tier."""
    for grader in available_graders():
        for tier in load_standards()["graders"][grader]["tiers"]:
            assert tier["back_strict"] >= tier["front_strict"]


def test_unknown_grader_raises() -> None:
    with pytest.raises(KeyError):
        grade_band(Measured(55.0, 0.1), "NOT_A_GRADER")


def test_all_grade_bands_covers_every_grader() -> None:
    bands = all_grade_bands(Measured(58.0, 0.3))
    assert set(bands) == set(available_graders())


def test_low_confidence_graders_are_flagged() -> None:
    """Tables we could not source well must say so, or they mislead."""
    std = load_standards()["graders"]
    for name, g in std.items():
        if g.get("confidence") == "low":
            assert "indicative" in caveat_text(name).lower()


def test_non_subgrade_graders_disclose_the_ceiling_caveat() -> None:
    text = caveat_text("PSA")
    assert "ceiling" in text.lower()
