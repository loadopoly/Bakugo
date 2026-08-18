"""Tests for OCR gating and optimal liquidation.

Both modules exist mainly to refuse. The OCR refuses when the glyphs are not in
the pixels or when a reading does not uniquely identify a real printing; the
execution model refuses for single cards and for schedules the market cannot
absorb.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from cardcenter.catalog import MIN_PX_TO_READ, CatalogEntry
from cardcenter.execution import (
    ExecutionNotApplicable,
    ImpactParameters,
    check_feasibility,
    efficient_frontier,
    optimal_liquidation,
    solve_kappa,
)
from cardcenter.ocr import (
    NumberResult,
    OcrReading,
    TesseractEngine,
    levenshtein,
    number_region,
    preprocess_number_crop,
    read_collector_number,
)


def _cands(*numbers: str) -> list[CatalogEntry]:
    return [
        CatalogEntry(card_id=n, name="X", set_name="S", collector_number=n)
        for n in numbers
    ]


class FakeEngine:
    def __init__(self, text: str) -> None:
        self.text = text

    @property
    def name(self) -> str:
        return "fake"

    def read(self, image, charset: str) -> OcrReading:
        return OcrReading(text=self.text, confidence=90.0, engine="fake")


def _number_image(text: str = "263", glyph_px: int = 44) -> np.ndarray:
    """Render a collector number in a real typeface.

    OpenCV's Hershey fonts are stroke fonts and Tesseract reads them poorly,
    which makes them a misleading fixture: they understate the pipeline. Real
    cards are printed in ordinary typefaces, so the fixture uses one.
    """
    from PIL import Image, ImageDraw, ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            font = ImageFont.truetype(path, glyph_px)
            break
    else:  # pragma: no cover - environment without these fonts
        pytest.skip("no suitable TrueType font available")

    im = Image.new(
        "RGB",
        (int(glyph_px * len(text) * 0.8) + 20, int(glyph_px * 1.7)),
        (238, 236, 232),
    )
    ImageDraw.Draw(im).text((10, glyph_px * 0.2), text, font=font, fill=(22, 22, 26))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------
# The resolution gate
# --------------------------------------------------------------------------


def test_ocr_is_not_attempted_below_the_gate() -> None:
    """A better OCR engine cannot read digits that are not in the pixels."""
    res = read_collector_number(
        _number_image(), px_per_mm=6.0, candidates=_cands("263", "273"),
        engine=FakeEngine("263"),
    )
    assert res.gated_out
    assert not res.resolved
    assert res.engine == "none (gated)"
    assert any("plausible number rather than a refusal" in w for w in res.warnings)


def test_gate_opens_at_sufficient_resolution() -> None:
    res = read_collector_number(
        _number_image(), px_per_mm=18.0, candidates=_cands("263", "273"),
        engine=FakeEngine("263"),
    )
    assert not res.gated_out
    assert res.resolved
    assert res.snapped_to == "263"


def test_gate_boundary_matches_the_documented_floor() -> None:
    from cardcenter.catalog import FEATURE_SIZE_MM

    ppm = MIN_PX_TO_READ / FEATURE_SIZE_MM["collector_number"]
    below = read_collector_number(
        _number_image(), ppm * 0.99, _cands("1"), engine=FakeEngine("1")
    )
    above = read_collector_number(
        _number_image(), ppm * 1.01, _cands("1"), engine=FakeEngine("1")
    )
    assert below.gated_out
    assert not above.gated_out


# --------------------------------------------------------------------------
# The catalog constraint -- the part that makes OCR safe here
# --------------------------------------------------------------------------


def test_garbled_reading_snaps_to_the_real_number() -> None:
    """'2S3' is not a number; against the candidate list it is unambiguously 263."""
    res = read_collector_number(
        _number_image(), 20.0, _cands("263", "107", "5"), engine=FakeEngine("2s3")
    )
    assert res.snapped_to == "263"
    assert any("corrected to" in w for w in res.warnings)


def test_reading_that_matches_nothing_is_rejected() -> None:
    res = read_collector_number(
        _number_image(), 20.0, _cands("263", "107"), engine=FakeEngine("9999")
    )
    assert not res.resolved
    assert any("beyond the snap limit" in w for w in res.warnings)


def test_reading_equidistant_from_two_printings_stays_ambiguous() -> None:
    """'11' is one edit from both '1' and '111'. It disambiguates neither."""
    res = read_collector_number(
        _number_image(), 20.0, _cands("1", "111"), engine=FakeEngine("11")
    )
    assert not res.resolved
    assert set(res.ambiguous_matches) == {"1", "111"}


def test_exact_match_needs_no_correction_warning() -> None:
    res = read_collector_number(
        _number_image(), 20.0, _cands("263", "107"), engine=FakeEngine("263")
    )
    assert res.resolved
    assert not any("corrected" in w for w in res.warnings)


def test_empty_reading_is_not_resolved() -> None:
    res = read_collector_number(
        _number_image(), 20.0, _cands("263"), engine=FakeEngine("")
    )
    assert not res.resolved
    assert any("nothing legible" in w for w in res.warnings)


def test_no_candidates_means_no_constraint_and_no_resolution() -> None:
    res = read_collector_number(_number_image(), 20.0, [], engine=FakeEngine("263"))
    assert not res.resolved
    assert any("no catalog candidates" in w for w in res.warnings)


def test_levenshtein_basics() -> None:
    assert levenshtein("263", "263") == 0
    assert levenshtein("2s3", "263") == 1
    assert levenshtein("", "abc") == 3


def test_preprocessing_yields_dark_text_on_light() -> None:
    out = preprocess_number_crop(_number_image())
    assert out.ndim == 2
    assert float((out == 255).mean()) > 0.5  # background is the majority


def test_number_region_crops_bottom_left() -> None:
    img = np.zeros((880, 640, 3), dtype=np.uint8)
    crop = number_region(img, px_per_mm=10.0)
    assert crop.shape[0] < img.shape[0] * 0.2
    assert crop.shape[1] < img.shape[1] * 0.5


@pytest.mark.skipif(
    __import__("shutil").which("tesseract") is None, reason="tesseract not installed"
)
def test_real_tesseract_reads_a_clean_rendered_number() -> None:
    res = read_collector_number(
        _number_image("263"), 25.0, _cands("263", "107", "5", "42", "311"),
        engine=TesseractEngine(),
    )
    assert res.snapped_to == "263"


@pytest.mark.skipif(
    __import__("shutil").which("tesseract") is None, reason="tesseract not installed"
)
def test_catalog_constraint_recovers_a_misread_digit() -> None:
    """The point of the closed vocabulary: a wrong glyph can still yield the
    right printing, because the answer must be one that exists."""
    res = read_collector_number(
        _number_image("5"), 25.0, _cands("5", "263", "107", "42"),
        engine=TesseractEngine(),
    )
    assert res.snapped_to == "5"


def test_marginal_resolution_is_flagged_for_verification() -> None:
    from cardcenter.catalog import FEATURE_SIZE_MM, MARGINAL_PX_TO_READ

    ppm = (MARGINAL_PX_TO_READ - 2.0) / FEATURE_SIZE_MM["collector_number"]
    res = read_collector_number(
        _number_image(), ppm, _cands("263", "107"), engine=FakeEngine("263")
    )
    assert res.resolved
    assert any("marginal band" in w for w in res.warnings)


# --------------------------------------------------------------------------
# Almgren-Chriss
# --------------------------------------------------------------------------


def _impact(sigma: float = 2.0) -> ImpactParameters:
    return ImpactParameters(
        permanent_gamma=0.5, temporary_eta=2.0, daily_volatility=sigma,
        calibrated=True,
    )


def test_single_card_has_no_schedule() -> None:
    with pytest.raises(ExecutionNotApplicable, match="single decision"):
        optimal_liquidation(1, 30.0, _impact())


def test_small_order_is_flagged_as_mostly_rounding() -> None:
    s = optimal_liquidation(3, 30.0, _impact())
    assert any("rounding" in w for w in s.warnings)


def test_holdings_decrease_monotonically_to_zero() -> None:
    s = optimal_liquidation(20, 30.0, _impact(), risk_aversion=1e-3)
    assert s.holdings[0] == pytest.approx(20.0)
    assert s.holdings[-1] == pytest.approx(0.0, abs=1e-9)
    assert all(a >= b - 1e-9 for a, b in zip(s.holdings, s.holdings[1:]))


def test_risk_neutral_schedule_is_linear() -> None:
    s = optimal_liquidation(20, 20.0, _impact(), risk_aversion=0.0)
    mid = s.holdings[len(s.holdings) // 2]
    assert mid == pytest.approx(10.0, rel=1e-6)
    assert any("linear" in w for w in s.warnings)


def test_higher_risk_aversion_front_loads_the_schedule() -> None:
    calm = optimal_liquidation(20, 30.0, _impact(), risk_aversion=1e-4)
    urgent = optimal_liquidation(20, 30.0, _impact(), risk_aversion=1e-1)
    mid = len(calm.holdings) // 2
    assert urgent.holdings[mid] < calm.holdings[mid]
    assert urgent.half_life_days < calm.half_life_days


def test_urgency_costs_more_but_risks_less() -> None:
    """The whole point of the model: cost and risk trade off against each other."""
    calm = optimal_liquidation(30, 40.0, _impact(), risk_aversion=1e-4)
    urgent = optimal_liquidation(30, 40.0, _impact(), risk_aversion=1e-1)
    assert urgent.expected_cost > calm.expected_cost
    assert urgent.cost_stdev < calm.cost_stdev


def test_efficient_frontier_is_monotone() -> None:
    front = efficient_frontier(25, 30.0, _impact())
    costs = [c for _, c, _ in front]
    risks = [r for _, _, r in front]
    assert all(a <= b + 1e-9 for a, b in zip(costs, costs[1:]))
    assert all(a >= b - 1e-9 for a, b in zip(risks, risks[1:]))


def test_kappa_zero_when_risk_neutral() -> None:
    assert solve_kappa(0.0, 2.0, 1.0, 0.5) == 0.0
    assert solve_kappa(1e-3, 0.0, 1.0, 0.5) == 0.0


def test_kappa_increases_with_volatility() -> None:
    assert solve_kappa(1e-2, 4.0, 1.0, 0.5) > solve_kappa(1e-2, 1.0, 1.0, 0.5)


def test_degenerate_eta_is_rejected() -> None:
    bad = ImpactParameters(
        permanent_gamma=100.0, temporary_eta=0.1, daily_volatility=2.0, calibrated=True
    )
    with pytest.raises(ExecutionNotApplicable, match="non-positive"):
        optimal_liquidation(20, 30.0, bad, risk_aversion=1e-3)


def test_uncalibrated_impact_warns_that_dollars_are_meaningless() -> None:
    s = optimal_liquidation(20, 30.0, ImpactParameters.assumed(100.0, 2.0))
    assert not s.impact_calibrated
    assert any("dollar figures are not" in w for w in s.warnings)


# --------------------------------------------------------------------------
# Feasibility -- the check that keeps this honest for thin markets
# --------------------------------------------------------------------------


def test_schedule_exceeding_market_volume_is_infeasible() -> None:
    s = optimal_liquidation(30, 10.0, _impact(), risk_aversion=1e-2)
    checked = check_feasibility(s, sales_per_month=0.5)
    assert checked.feasible is False
    assert any("not executable" in w for w in checked.warnings)


def test_schedule_within_market_volume_is_feasible() -> None:
    s = optimal_liquidation(10, 90.0, _impact(), risk_aversion=1e-8)
    checked = check_feasibility(s, sales_per_month=60.0)
    assert checked.feasible is True


def test_zero_volume_market_is_never_feasible() -> None:
    s = optimal_liquidation(10, 30.0, _impact())
    checked = check_feasibility(s, sales_per_month=0.0)
    assert checked.feasible is False
    assert any("no recorded trades" in w for w in checked.warnings)


def test_large_share_of_volume_is_warned_even_when_feasible() -> None:
    s = optimal_liquidation(10, 30.0, _impact(), risk_aversion=1e-8)
    peak_per_day = max(s.trades) / (30.0 / s.n_intervals)
    checked = check_feasibility(s, sales_per_month=peak_per_day * 30.0 / 1.2)
    assert any("large fraction" in w or "not executable" in w for w in checked.warnings)
