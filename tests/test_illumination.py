"""Tests for illumination geometry.

The edge-shadow case is the highest-severity failure found in this project: a
perfectly centred card measuring 83/17 because a lamp was low. Most of these
tests exist to make sure it stays caught.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cardcenter.centering import measure_centering
from cardcenter.illumination import (
    SAFE_ELEVATION_DEG,
    TYPICAL_CARD_THICKNESS_MM,
    coherence_check,
    detect_edge_shadow,
    edge_shadow_mm,
    ratio_bias_pp,
)
from cardcenter.synth import render_capture
from cardcenter.types import CaptureSpec, DetectionError


# --------------------------------------------------------------------------
# Coherence: the formal answer to "is there phase structure to recover"
# --------------------------------------------------------------------------


def test_ordinary_light_is_incoherent_across_a_card_border() -> None:
    c = coherence_check(feature_mm=3.0)
    assert c.ratio > 50
    assert not c.phase_retrieval_possible
    assert "no extended phase relationship" in c.describe()


def test_narrowband_light_is_more_coherent() -> None:
    broad = coherence_check(bandwidth_nm=300.0)
    narrow = coherence_check(bandwidth_nm=1.0)
    assert narrow.temporal_um > broad.temporal_um


def test_smaller_angular_source_raises_spatial_coherence() -> None:
    assert (
        coherence_check(source_angular_radius_rad=0.001).spatial_um
        > coherence_check(source_angular_radius_rad=0.05).spatial_um
    )


def test_even_a_laser_pointer_is_incoherent_across_millimetres_of_a_scene() -> None:
    """Not a loophole: spatial coherence at the card still comes from the
    source's angular size, not its linewidth."""
    c = coherence_check(bandwidth_nm=0.01, source_angular_radius_rad=0.01, feature_mm=3.0)
    assert not c.phase_retrieval_possible


# --------------------------------------------------------------------------
# Edge shadow geometry
# --------------------------------------------------------------------------


def test_overhead_light_casts_no_edge_shadow() -> None:
    assert edge_shadow_mm(0.30, 90.0) == 0.0


def test_shadow_grows_as_the_light_drops() -> None:
    prev = 0.0
    for el in (75.0, 60.0, 45.0, 30.0, 15.0):
        s = edge_shadow_mm(0.30, el)
        assert s > prev
        prev = s


def test_shadow_matches_the_trigonometry() -> None:
    assert edge_shadow_mm(0.30, 45.0) == pytest.approx(0.30)
    assert edge_shadow_mm(0.30, 30.0) == pytest.approx(0.30 / math.tan(math.radians(30)))


def test_bias_does_not_cancel_in_the_ratio() -> None:
    """The whole reason this matters: it is one-sided."""
    assert ratio_bias_pp(0.52, 3.0, 3.0) > 3.9
    assert ratio_bias_pp(0.0, 3.0, 3.0) == pytest.approx(0.0)


def test_thicker_holder_casts_a_worse_shadow() -> None:
    assert edge_shadow_mm(3.2, 45.0) > edge_shadow_mm(0.30, 45.0)


# --------------------------------------------------------------------------
# Detection from the image
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lit_captures():
    out = {}
    for el in (90.0, 45.0, 30.0, 20.0):
        out[el] = render_capture(
            left_mm=3.0, right_mm=3.0, top_mm=3.0, bottom_mm=3.0,
            card_thickness_mm=0.30, light_elevation_deg=el, light_azimuth_deg=0.0,
        )
    return out


def test_even_lighting_is_not_flagged(lit_captures) -> None:
    from cardcenter.geometry import find_card_quad

    img, _, _ = lit_captures[90.0]
    quad, _, _ = find_card_quad(img)
    v = detect_edge_shadow(img, quad, 10.8)
    assert not v.directional
    assert "no edge shadow" in v.describe()


def test_low_light_is_detected_and_localised(lit_captures) -> None:
    from cardcenter.geometry import find_card_quad

    img, _, _ = lit_captures[30.0]
    quad, _, _ = find_card_quad(img)
    v = detect_edge_shadow(img, quad, 10.8)
    assert v.directional
    assert v.darker_side == "left"  # light came from the left
    assert v.severe


def test_estimated_elevation_is_close_to_truth(lit_captures) -> None:
    from cardcenter.geometry import find_card_quad

    for true_el in (30.0, 20.0):
        img, _, _ = lit_captures[true_el]
        quad, _, _ = find_card_quad(img)
        v = detect_edge_shadow(img, quad, 10.8)
        assert v.estimated_elevation_deg == pytest.approx(true_el, abs=8.0)


def test_severe_shadow_is_corrected_not_refused(lit_captures) -> None:
    """A perfectly centred card measured 82.8/17.2 at this elevation before the
    check existed. Refusing was also wrong: shop lighting is whatever the case
    has, so a tool that declines below 55 degrees declines nearly everything.
    The shadow is measurable, so it is subtracted."""
    img, gt, f = lit_captures[30.0]
    r = measure_centering(img, slab="raw", capture=CaptureSpec(focal_px=f))
    assert r.horizontal.ratio_pct.value == pytest.approx(50.0, abs=1.5)
    assert any("subtracted" in w for w in r.quality.warnings)


def test_correction_inflates_the_error_bar_on_the_affected_border(lit_captures) -> None:
    """The correction is itself a coarse measurement -- a penumbra is not a step
    -- so it must widen the interval rather than pretending to be exact."""
    clean, _, f_clean = lit_captures[90.0]
    shadowed, _, f_dark = lit_captures[30.0]
    a = measure_centering(clean, slab="raw", capture=CaptureSpec(focal_px=f_clean))
    b = measure_centering(shadowed, slab="raw", capture=CaptureSpec(focal_px=f_dark))
    assert b.horizontal.ratio_pct.sigma > 2 * a.horizontal.ratio_pct.sigma


def test_corrected_interval_still_covers_the_truth(lit_captures) -> None:
    for el in (30.0, 20.0):
        img, gt, f = lit_captures[el]
        r = measure_centering(img, slab="raw", capture=CaptureSpec(focal_px=f))
        lo, hi = r.horizontal.ratio_pct.interval()
        assert lo <= gt.h_ratio <= hi


def test_slabbed_card_in_a_case_is_unaffected() -> None:
    """The actual shop scenario. A card held against a slab's inner well has a
    ~0.15mm gap rather than 0.30mm, and the resulting shadow never gets large
    enough to matter at any lighting angle."""
    from cardcenter.synth import render_capture as rc

    for el in (90.0, 45.0, 30.0, 20.0):
        img, gt, f = rc(
            left_mm=3.0, right_mm=3.0, card_thickness_mm=0.15,
            light_elevation_deg=el, light_azimuth_deg=0.0,
        )
        r = measure_centering(img, slab="raw", capture=CaptureSpec(focal_px=f))
        assert r.horizontal.ratio_pct.value == pytest.approx(gt.h_ratio, abs=1.5)


def test_acceptable_lighting_still_measures(lit_captures) -> None:
    for el in (90.0, 45.0):
        img, gt, f = lit_captures[el]
        r = measure_centering(img, slab="raw", capture=CaptureSpec(focal_px=f))
        assert r.horizontal.ratio_pct.value == pytest.approx(gt.h_ratio, abs=1.5)


def test_uncorrectable_shadow_says_what_to_change() -> None:
    """Only when the band is wide enough to overlap the printed border does the
    tool give up -- and then it suggests a change that works in a shop, moving
    around the case, not moving the shop's lights."""
    from cardcenter.illumination import ShadowVerdict

    v = ShadowVerdict(
        asymmetry=0.9, darker_side="left", estimated_shadow_mm=2.4,
        estimated_elevation_deg=7.0, directional=True, severe=True,
        warnings=(), side_index=3, correctable=False,
    )
    assert not v.correctable


def test_correction_shifts_the_edge_inward() -> None:
    from cardcenter.illumination import correct_quad_for_shadow

    quad = np.array([[0.0, 0.0], [600.0, 0.0], [600.0, 840.0], [0.0, 840.0]])
    fixed = correct_quad_for_shadow(quad, 3, 0.5, 10.0)  # side 3 = left edge
    assert fixed[3][0] > quad[3][0]
    assert fixed[0][0] > quad[0][0]
    assert np.allclose(fixed[1], quad[1])


def test_safe_elevation_threshold_is_where_the_bias_becomes_material() -> None:
    """At the threshold the induced ratio bias should still be sub-point."""
    s = edge_shadow_mm(TYPICAL_CARD_THICKNESS_MM, SAFE_ELEVATION_DEG)
    assert ratio_bias_pp(s, 3.0, 3.0) < 2.0
