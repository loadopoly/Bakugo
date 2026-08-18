"""Tests for the AR session loop and caliper calibration."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
import pytest

from cardcenter.ar import (
    DEFAULT_DRIFT_PER_HOUR,
    MEASURE_LONG_SIDE,
    TRACK_LONG_SIDE,
    ARSession,
    ScaleCalibration,
    calibrate_from_points,
    detect_caliper_gap,
    track_quad,
    verify_calibration_against_card,
)
from cardcenter.geometry import find_card_quad
from cardcenter.synth import render_capture
from cardcenter.types import STANDARD_CARD_W_MM, DetectionError


@pytest.fixture(scope="module")
def small_card():
    img, gt, f = render_capture(left_mm=3.4, right_mm=2.6, tilt_deg=10.0)
    s = TRACK_LONG_SIDE / max(img.shape[:2])
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    quad, _, _ = find_card_quad(small)
    return small, quad, gt


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------


def test_tracker_agrees_with_full_detection(small_card) -> None:
    """The tracker searches along edge normals, so it locks onto the STRONGEST
    nearby gradient. On a bordered card the printed frame is often a stronger
    step than the die cut, so the tracker drifts toward it -- disagreement falls
    from 9.2 px to 3.2 px as the search radius narrows from 14 to 3.

    The search is therefore scaled to card size rather than fixed. A few pixels
    of tracker drift is acceptable because tracking only positions the overlay;
    every actual MEASUREMENT re-runs full detection at 1200 px."""
    small, quad, _ = small_card
    assert np.abs(track_quad(small, quad) - quad).max() < 7.0


def test_tracker_recovers_from_a_small_displacement(small_card) -> None:
    small, quad, _ = small_card
    nudged = quad + np.array([4.0, -3.0])
    assert np.abs(track_quad(small, nudged) - quad).max() < 9.0


def test_tracker_refuses_rather_than_drifting(small_card) -> None:
    """A tracker that has wandered onto something else is worse than one that
    admits it lost the card, because the session keeps averaging."""
    small, quad, _ = small_card
    far = quad + np.array([260.0, 200.0])
    with pytest.raises(DetectionError):
        track_quad(small, far)


def test_tracker_rejects_a_degenerate_quad(small_card) -> None:
    small, _, _ = small_card
    with pytest.raises(DetectionError):
        track_quad(small, np.array([[10.0, 10.0]] * 4))


# --------------------------------------------------------------------------
# Session behaviour
# --------------------------------------------------------------------------


def test_session_accumulates_and_settles() -> None:
    rng = np.random.default_rng(2)
    s = ARSession()
    now = 0.0
    status = None
    for i in range(16):
        img, gt, _ = render_capture(
            left_mm=3.4, right_mm=2.6, tilt_deg=float(rng.uniform(4, 18)),
            azimuth_deg=float(rng.uniform(0, 360)), noise_sigma=3.0, seed=i,
        )
        now += 0.12
        status = s.push(img, now=now)
    assert status.tracking
    assert status.measured_frames >= 3
    assert status.ratio.value == pytest.approx(gt.h_ratio, abs=1.5)


def test_session_throttles_measurement() -> None:
    """Measuring every frame would stall the overlay; the loop measures at a
    few Hz and tracks the rest of the time."""
    img, _, _ = render_capture(left_mm=3.4, right_mm=2.6)
    s = ARSession(measure_interval_s=1.0)
    for i in range(6):
        s.push(img, now=i * 0.1)
    assert s.measured <= 1
    assert s.seen == 6


def test_gate_uses_measurement_resolution_not_tracking_resolution() -> None:
    """Tracking runs at 540px where a card is ~5 px/mm, below the usable floor.
    Gating on that would reject every frame while the 1200px measurement would
    have been fine."""
    img, _, _ = render_capture(left_mm=3.4, right_mm=2.6)
    s = ARSession()
    st = s.push(img, now=1.0)
    assert not any("too far away" in g for g in st.guidance)


def test_session_reports_no_card_cleanly() -> None:
    s = ARSession()
    st = s.push(np.full((600, 800, 3), 120, dtype=np.uint8), now=1.0)
    assert not st.tracking
    assert any("point at a card" in g for g in st.guidance)


def test_reset_clears_accumulated_state() -> None:
    img, _, _ = render_capture(left_mm=3.4, right_mm=2.6)
    s = ARSession()
    s.push(img, now=1.0)
    s.push(img, now=2.0)
    s.reset()
    assert s.measured == 0 and s.seen == 0 and s.worst_ratio is None


# --------------------------------------------------------------------------
# Caliper calibration
# --------------------------------------------------------------------------


def test_calibration_recovers_a_known_scale() -> None:
    cal = calibrate_from_points((100.0, 300.0), (100.0 + 500.0, 300.0), 50.0)
    assert cal.px_per_mm == pytest.approx(10.0)
    assert cal.method == "caliper"


def test_caliper_precision_is_wasted_on_a_short_pixel_baseline() -> None:
    """The caliper reads to 0.02mm, but that is thrown away if its jaws cannot
    be located to better than a pixel or two. At 500 px the localisation term is
    0.42% -- worse than a bank card's published 0.152%."""
    cal = calibrate_from_points((100.0, 300.0), (600.0, 300.0), 50.0)
    rel = cal.sigma / cal.px_per_mm
    assert rel > 0.152 / 100
    assert any("would do as well" in w for w in cal.warnings)


def test_caliper_beats_a_bank_card_once_the_baseline_is_long_enough() -> None:
    cal = calibrate_from_points((100.0, 300.0), (100.0 + 3000.0, 300.0), 50.0)
    assert cal.sigma / cal.px_per_mm < 0.152 / 100
    assert not any("would do as well" in w for w in cal.warnings)


def test_short_baseline_is_refused() -> None:
    with pytest.raises(DetectionError, match="short baseline"):
        calibrate_from_points((100.0, 300.0), (110.0, 300.0), 50.0)


def test_depth_mismatch_is_the_dominant_risk_and_is_flagged() -> None:
    """Scale goes as 1/distance, so a caliper held 10% nearer is a 10% error --
    6.3mm on a card, a hundred times the thing being measured."""
    cal = calibrate_from_points(
        (100.0, 300.0), (600.0, 300.0), 50.0, depth_mismatch_frac=0.10
    )
    assert cal.sigma / cal.px_per_mm > 0.09
    assert any("same surface" in w for w in cal.warnings)


def test_missing_depth_information_is_disclosed() -> None:
    cal = calibrate_from_points((100.0, 300.0), (600.0, 300.0), 50.0)
    assert any("coplanarity is assumed" in w for w in cal.warnings)


def test_calibration_uncertainty_widens_with_age() -> None:
    cal = ScaleCalibration(
        px_per_mm=10.0, sigma=0.01, method="caliper", observed_at=0.0
    )
    fresh = cal.current(now=0.0)
    old = cal.current(now=6 * 3600.0)
    assert old.sigma > fresh.sigma
    assert cal.stale(now=6 * 3600.0)
    assert not cal.stale(now=600.0)


def test_verification_accepts_a_sound_calibration() -> None:
    ppm = 10.0
    cal = ScaleCalibration(ppm, 0.01, "caliper", time.time())
    quad = np.array(
        [[0, 0], [STANDARD_CARD_W_MM * ppm, 0],
         [STANDARD_CARD_W_MM * ppm, 88.9 * ppm], [0, 88.9 * ppm]], dtype=float
    )
    ok, msg = verify_calibration_against_card(cal, quad)
    assert ok and "looks sound" in msg


def test_verification_cannot_distinguish_bad_scale_from_a_trimmed_card() -> None:
    """An honest ambiguity, reported rather than resolved by guessing."""
    ppm = 10.0
    cal = ScaleCalibration(ppm, 0.01, "caliper", time.time())
    quad = np.array(
        [[0, 0], [(STANDARD_CARD_W_MM - 1.0) * ppm, 0],
         [(STANDARD_CARD_W_MM - 1.0) * ppm, 88.9 * ppm], [0, 88.9 * ppm]], dtype=float
    )
    ok, msg = verify_calibration_against_card(cal, quad)
    assert not ok
    assert "or this card is genuinely off-size" in msg


def test_wildly_wrong_calibration_is_named_as_such() -> None:
    cal = ScaleCalibration(8.0, 0.01, "caliper", time.time())
    quad = np.array([[0, 0], [635, 0], [635, 889], [0, 889]], dtype=float)
    ok, msg = verify_calibration_against_card(cal, quad)
    assert not ok
    assert "calibration is wrong" in msg


def test_caliper_gap_detected_from_a_synthetic_jaw_pair() -> None:
    img = np.full((300, 700, 3), 210, dtype=np.uint8)
    cv2.rectangle(img, (100, 60), (140, 240), (30, 30, 35), -1)   # left jaw
    cv2.rectangle(img, (500, 60), (540, 240), (30, 30, 35), -1)   # right jaw
    # The measurement is between the INNER jaw faces (x=140 and x=500), not the
    # outer edges: a caliper closes on the object.
    p1, p2 = detect_caliper_gap(img)
    assert abs(abs(p2[0] - p1[0]) - 360.0) < 25.0


def test_ambiguous_jaws_are_refused_not_guessed() -> None:
    """A mis-detected calibration corrupts every measurement taken after it."""
    with pytest.raises(DetectionError):
        detect_caliper_gap(np.full((300, 700, 3), 210, dtype=np.uint8))
