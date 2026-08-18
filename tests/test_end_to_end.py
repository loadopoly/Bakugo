"""End-to-end pipeline tests, and the refusal paths.

Half of these assert that the tool *declines* to answer. That is deliberate. A
measurement tool's failure mode is not returning a wrong number loudly, it is
returning a wrong number quietly, and the only defence is testing that bad
input produces a refusal instead of a plausible-looking float.
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from cardcenter.centering import measure_centering
from cardcenter.detect import detect_side_border
from cardcenter.render import annotate
from cardcenter.synth import render_capture
from cardcenter.types import STANDARD_CARD_H_MM, STANDARD_CARD_W_MM, CaptureSpec, DetectionError


@pytest.fixture(scope="module")
def straight_on():
    img, gt, f = render_capture(
        left_mm=3.4, right_mm=2.6, top_mm=3.1, bottom_mm=2.9, tilt_deg=0.0
    )
    return img, gt, f


def test_recovers_known_centering(straight_on) -> None:
    img, gt, f = straight_on
    res = measure_centering(img, slab="raw", capture=CaptureSpec(focal_px=f))
    assert res.horizontal.ratio_pct.value == pytest.approx(gt.h_ratio, abs=1.0)
    assert res.vertical.ratio_pct.value == pytest.approx(gt.v_ratio, abs=1.0)


def test_recovers_border_widths_in_mm(straight_on) -> None:
    img, gt, f = straight_on
    res = measure_centering(img, slab="raw", capture=CaptureSpec(focal_px=f))
    assert res.horizontal.low_mm.value == pytest.approx(gt.left_mm, abs=0.15)
    assert res.horizontal.high_mm.value == pytest.approx(gt.right_mm, abs=0.15)
    assert res.vertical.low_mm.value == pytest.approx(gt.top_mm, abs=0.15)
    assert res.vertical.high_mm.value == pytest.approx(gt.bottom_mm, abs=0.15)


def test_identifies_the_wider_side(straight_on) -> None:
    img, _, f = straight_on
    res = measure_centering(img, slab="raw", capture=CaptureSpec(focal_px=f))
    assert res.horizontal.skew_toward == "left"  # left was rendered wider
    assert res.vertical.skew_toward == "top"


@pytest.mark.parametrize("tilt", [0.0, 18.0, 32.0])
def test_tilt_is_corrected(tilt: float) -> None:
    img, gt, f = render_capture(
        left_mm=4.0, right_mm=2.4, top_mm=3.0, bottom_mm=3.0, tilt_deg=tilt
    )
    res = measure_centering(img, slab="raw", capture=CaptureSpec(focal_px=f))
    assert res.horizontal.ratio_pct.value == pytest.approx(gt.h_ratio, abs=1.2)


def test_confidence_interval_contains_truth() -> None:
    img, gt, f = render_capture(
        left_mm=3.8, right_mm=2.2, top_mm=3.0, bottom_mm=3.0, tilt_deg=20.0, slab="bgs"
    )
    res = measure_centering(img, slab="bgs", capture=CaptureSpec(focal_px=f))
    lo, hi = res.horizontal.ratio_pct.interval(1.96)
    assert lo <= gt.h_ratio <= hi


def test_survives_text_intruding_into_the_border() -> None:
    """A mean-based detector fails this; the median-based one must not."""
    img, gt, f = render_capture(
        left_mm=3.5, right_mm=2.5, top_mm=3.0, bottom_mm=3.0, add_border_text=True
    )
    res = measure_centering(img, slab="raw", capture=CaptureSpec(focal_px=f))
    assert res.horizontal.ratio_pct.value == pytest.approx(gt.h_ratio, abs=1.5)
    assert res.vertical.ratio_pct.value == pytest.approx(gt.v_ratio, abs=1.5)


def test_refraction_correction_is_applied_when_pose_is_known() -> None:
    img, _, f = render_capture(tilt_deg=25.0, slab="bgs")
    res = measure_centering(img, slab="bgs", capture=CaptureSpec(focal_px=f))
    assert res.quality.refraction_applied
    assert res.quality.max_refraction_shift_mm > 0.0


def test_missing_focal_length_inflates_uncertainty_for_slabs() -> None:
    """Without pose we cannot correct; the error bar must grow and say why."""
    img, _, f = render_capture(tilt_deg=25.0, slab="bgs")
    known = measure_centering(img, slab="bgs", capture=CaptureSpec(focal_px=f))
    unknown = measure_centering(img, slab="bgs", capture=CaptureSpec())
    assert unknown.horizontal.ratio_pct.sigma > known.horizontal.ratio_pct.sigma
    assert not unknown.quality.refraction_applied
    assert any("focal length" in w for w in unknown.quality.warnings)


def test_raw_card_needs_no_focal_length() -> None:
    """No slab means no refraction, so the warning must not fire."""
    img, gt, _ = render_capture(left_mm=3.4, right_mm=2.6)
    res = measure_centering(img, slab="raw", capture=CaptureSpec())
    assert not any("focal length" in w for w in res.quality.warnings)
    assert res.horizontal.ratio_pct.value == pytest.approx(gt.h_ratio, abs=1.2)


# --------------------------------------------------------------------------
# Refusal paths
# --------------------------------------------------------------------------


def test_refuses_full_bleed_card() -> None:
    """No printed border means nothing to measure. Say so; do not invent one."""
    img = np.full((1800, 1400, 3), 30, dtype=np.uint8)
    cv2.rectangle(img, (400, 450), (1000, 1290), (150, 90, 70), -1)
    with pytest.raises(DetectionError, match="full-bleed|no measurable border"):
        measure_centering(img, slab="raw", capture=CaptureSpec(focal_px=2400.0))


def test_refuses_pure_noise() -> None:
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (900, 700, 3), dtype=np.uint8)
    with pytest.raises(DetectionError):
        measure_centering(img, slab="raw")


def test_refuses_empty_image() -> None:
    with pytest.raises(DetectionError, match="empty"):
        measure_centering(np.zeros((0, 0, 3), dtype=np.uint8))


def test_refuses_when_no_card_shape_present() -> None:
    img = np.full((900, 700, 3), 120, dtype=np.uint8)
    with pytest.raises(DetectionError, match="card-shaped|no measurable"):
        measure_centering(img, slab="raw")


def test_unknown_slab_preset_raises() -> None:
    img, _, f = render_capture()
    with pytest.raises(KeyError, match="unknown slab preset"):
        measure_centering(img, slab="unobtanium", capture=CaptureSpec(focal_px=f))


# --------------------------------------------------------------------------
# Detector unit behaviour
# --------------------------------------------------------------------------


def _synthetic_rect(border_px: int, slope: float = 0.0, size=(900, 640)) -> np.ndarray:
    """Rectified card with a known left border, optionally slanted."""
    h, w = size
    img = np.full((h, w, 3), (60, 200, 240), dtype=np.uint8)
    for y in range(h):
        x0 = int(round(border_px + slope * y))
        img[y, x0 : w - border_px] = (120, 70, 45)
    img[:, : max(0, 0)] = (60, 200, 240)
    return img


def test_detects_a_known_border_width() -> None:
    ppm = 10.0
    img = _synthetic_rect(int(3.0 * ppm))
    prof = detect_side_border(img, "left", ppm)
    assert prof.depth_mm == pytest.approx(3.0, abs=0.15)
    assert prof.confidence > 0.5


@pytest.mark.parametrize("side", ["left", "top", "right", "bottom"])
def test_all_four_sides_use_consistent_orientation(side: str) -> None:
    """A sign error in strip extraction would show up as a wrong side reading."""
    ppm = 10.0
    h, w = 900, 640
    img = np.full((h, w, 3), (60, 200, 240), dtype=np.uint8)
    margins = {"left": 20, "right": 60, "top": 30, "bottom": 45}
    img[margins["top"] : h - margins["bottom"], margins["left"] : w - margins["right"]] = (
        120,
        70,
        45,
    )
    prof = detect_side_border(img, side, ppm)
    assert prof.depth_mm == pytest.approx(margins[side] / ppm, abs=0.2)


def test_flags_rotated_printing() -> None:
    ppm = 10.0
    img = _synthetic_rect(int(3.0 * ppm), slope=0.03)
    prof = detect_side_border(img, "left", ppm)
    assert abs(prof.slope_mm_per_mm) > 0.012


def test_reports_higher_sigma_on_a_noisier_edge() -> None:
    ppm = 10.0
    clean = detect_side_border(_synthetic_rect(30), "left", ppm)
    rng = np.random.default_rng(1)
    noisy_img = _synthetic_rect(30).astype(np.int16)
    noisy_img += rng.normal(0, 18, noisy_img.shape).astype(np.int16)
    noisy = detect_side_border(np.clip(noisy_img, 0, 255).astype(np.uint8), "left", ppm)
    assert noisy.sigma_mm >= clean.sigma_mm


# --------------------------------------------------------------------------
# Output surfaces
# --------------------------------------------------------------------------


def test_annotate_produces_an_image(straight_on) -> None:
    img, _, f = straight_on
    res = measure_centering(img, slab="raw", capture=CaptureSpec(focal_px=f))
    out = annotate(res)
    assert out.ndim == 3
    assert out.shape[1] > res.rectified.shape[1]  # card plus the text panel


def test_cli_json_round_trip(straight_on, tmp_path) -> None:
    from cardcenter.cli import main

    img, gt, f = straight_on
    src = tmp_path / "card.png"
    cv2.imwrite(str(src), img)
    out = tmp_path / "r.json"
    overlay = tmp_path / "o.png"
    rc = main(
        [
            str(src),
            "--focal-px",
            str(f),
            "--json",
            str(out),
            "--overlay",
            str(overlay),
            "--quiet",
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["schema"] == "cardcenter/1"
    assert data["horizontal"]["ratio_pct"] == pytest.approx(gt.h_ratio, abs=1.2)
    assert "disclaimer" in data
    assert len(data["worst_ratio_ci95"]) == 2
    assert overlay.exists()


def test_cli_refusal_returns_nonzero(tmp_path) -> None:
    from cardcenter.cli import main

    src = tmp_path / "flat.png"
    cv2.imwrite(str(src), np.full((600, 500, 3), 120, dtype=np.uint8))
    assert main([str(src), "--quiet"]) == 1


def test_disclaimer_states_the_ceiling_limitation() -> None:
    from cardcenter.cli import DISCLAIMER

    low = DISCLAIMER.lower()
    assert "centering only" in low
    assert "not a grade prediction" in low
