"""Framing: spend the pixel budget on the card, and refuse when it is too small.

The numbers in these tests come from synthetic captures at a fixed focal
length, so "distance" is a stand-in for how small the card is in frame: a
3000x4000 photo with the card 313 px wide is what a card across a shop
counter looks like on a phone.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from cardcenter.capture import MIN_PX_PER_MM
from cardcenter.centering import measure_centering
from cardcenter.framing import frame_card_for_measure
from cardcenter.synth import render_capture
from cardcenter.types import CaptureSpec, DetectionError, resolve_holder

FOCAL = 3200.0
SIZE = (3000, 4000)          # render_capture takes (width, height)
WIDTH, HEIGHT = SIZE
FOV = 2 * math.degrees(math.atan(WIDTH / 2 / FOCAL))


def _shot(distance_mm, **kw):
    return render_capture(left_mm=3.4, right_mm=2.6, distance_mm=distance_mm,
                          focal_px=FOCAL, image_size=SIZE, noise_sigma=2.0, **kw)


def _cap_only(img, max_side=2400):
    s = min(1.0, max_side / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return small, CaptureSpec(focal_px=FOCAL * s)


def test_crop_keeps_the_cards_pixels():
    img, _, _ = _shot(650.0)
    framed = frame_card_for_measure(img, fov_deg=FOV, max_side=2400)
    assert framed.cropped
    # the card is ~313 px wide in the source; capping the whole frame at 2400
    # would leave it 188 px, the crop leaves it untouched
    assert framed.px_per_mm == pytest.approx(313 / 63.5, rel=0.1)
    assert max(framed.image.shape[:2]) < 900
    assert framed.scale == 1.0


def test_crop_beats_capping_on_a_distant_card():
    img, gt, _ = _shot(650.0)
    framed = frame_card_for_measure(img, fov_deg=FOV, max_side=2400)
    cropped = measure_centering(framed.image, slab=resolve_holder("raw"),
                                capture=framed.capture, card_quad=framed.quad,
                                quad_residual_px=framed.residual_px)
    small, cap = _cap_only(img)
    try:
        capped_err = abs(measure_centering(small, slab=resolve_holder("raw"),
                                           capture=cap).worst_ratio.value - gt.worst_ratio)
    except DetectionError:
        capped_err = float("inf")       # capping loses the card entirely
    assert abs(cropped.worst_ratio.value - gt.worst_ratio) < 1.0
    assert abs(cropped.worst_ratio.value - gt.worst_ratio) < capped_err


@pytest.mark.parametrize("distance", [300.0, 400.0, 650.0])
def test_measurement_stays_true_across_distances(distance):
    img, gt, _ = _shot(distance)
    f = frame_card_for_measure(img, fov_deg=FOV, max_side=2400)
    r = measure_centering(f.image, slab=resolve_holder("raw"), capture=f.capture,
                          card_quad=f.quad, quad_residual_px=f.residual_px)
    assert abs(r.worst_ratio.value - gt.worst_ratio) < 1.0


def test_intrinsics_follow_the_crop():
    img, _, _ = _shot(650.0)
    f = frame_card_for_measure(img, fov_deg=FOV, max_side=2400)
    # cropping does not change the lens: focal scales with the resize only
    assert f.capture.focal_px == pytest.approx(FOCAL * f.scale, rel=1e-6)
    cx, cy = f.capture.principal_point
    # the optical axis is the source centre, moved by the crop origin
    assert cx == pytest.approx((WIDTH / 2 - f.origin[0]) * f.scale, abs=0.5)
    assert cy == pytest.approx((HEIGHT / 2 - f.origin[1]) * f.scale, abs=0.5)
    # and a frame built from the crop's own width would be badly wrong
    assert f.capture.focal_px > 3 * CaptureSpec.from_fov(FOV, f.image.shape).focal_px


def test_located_outline_maps_into_the_framed_image():
    img, _, _ = _shot(500.0)
    f = frame_card_for_measure(img, fov_deg=FOV, max_side=2400)
    assert f.quad is not None
    h, w = f.image.shape[:2]
    assert f.quad[:, 0].min() > 0 and f.quad[:, 1].min() > 0
    assert f.quad[:, 0].max() < w and f.quad[:, 1].max() < h
    back = f.to_source(f.quad)
    assert np.abs(back.mean(axis=0) - np.array([WIDTH / 2, HEIGHT / 2])).max() < 40


def test_no_card_falls_back_to_a_plain_cap():
    blank = np.full((HEIGHT, WIDTH, 3), 110, np.uint8)
    f = frame_card_for_measure(blank, fov_deg=FOV, max_side=2400)
    assert not f.cropped and f.quad is None and f.px_per_mm is None
    assert max(f.image.shape[:2]) == 2400
    assert f.capture.focal_px == pytest.approx(FOCAL * 2400 / HEIGHT, rel=1e-6)


def test_an_outline_supplied_by_the_caller_is_used():
    img, _, _ = _shot(500.0)
    from cardcenter.geometry import find_card_quad

    quad, _, _ = find_card_quad(img, prefer_point=(WIDTH / 2, HEIGHT / 2))
    f = frame_card_for_measure(img, quad=quad, fov_deg=FOV, max_side=1200)
    assert f.cropped
    assert np.abs(f.to_source(f.quad) - quad).max() < 1.5


def test_a_card_below_the_resolution_floor_is_refused_by_the_server():
    from cardcenter.serve import _measure_payload

    img, _, _ = _shot(800.0)          # card ~254 px wide: located, but too small
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    assert ok
    with pytest.raises(DetectionError) as err:
        _measure_payload(jpg.tobytes(), "raw", "main")
    msg = str(err.value)
    assert "px/mm" in msg and ("zoom" in msg or "closer" in msg)
    assert f"{MIN_PX_PER_MM:.1f}" in msg


# ---- Shop floor: big sensors, phone-side crops, identify at distance ----

def test_min_area_follows_the_pixel_floor_not_the_frame():
    from cardcenter.framing import DEFAULT_MIN_AREA_FRAC, min_area_frac_for

    assert min_area_frac_for((600, 800)) == DEFAULT_MIN_AREA_FRAC
    assert min_area_frac_for((3000, 4000)) <= DEFAULT_MIN_AREA_FRAC
    # a 50 MP frame: a card at the measurable floor is ~0.2% of it
    assert min_area_frac_for((6120, 8160)) < 0.003


def test_a_measurable_card_on_a_big_sensor_is_found():
    # 27 MP frame, card ~317 px wide (5 px/mm): above the pixel floor but only
    # 0.5% of the frame, which the old fixed 0.8% area limit threw away.
    size = (4500, 6000)
    img, gt, _ = render_capture(left_mm=3.4, right_mm=2.6, distance_mm=640.0,
                                focal_px=FOCAL, image_size=size, noise_sigma=2.0)
    fov = 2 * math.degrees(math.atan(size[0] / 2 / FOCAL))
    f = frame_card_for_measure(img, fov_deg=fov, max_side=2400)
    assert f.cropped and f.quad is not None
    assert f.px_per_mm == pytest.approx(317 / 63.5, rel=0.12)
    r = measure_centering(f.image, slab=resolve_holder("raw"), capture=f.capture,
                          card_quad=f.quad, quad_residual_px=f.residual_px)
    assert abs(r.worst_ratio.value - gt.worst_ratio) < 1.0


def test_a_phone_side_crop_keeps_the_lens():
    img, gt, _ = _shot(650.0)
    full = frame_card_for_measure(img, fov_deg=FOV, max_side=2400)
    x0, y0 = 900, 1300                       # card is centred at (1500, 2000)
    sub = np.ascontiguousarray(img[y0:y0 + 1400, x0:x0 + 1200])
    f = frame_card_for_measure(sub, fov_deg=FOV, max_side=2400,
                               parent=(x0, y0, WIDTH, HEIGHT))
    # same lens as the whole photo, not a lens as wide as the crop
    assert f.capture.focal_px == pytest.approx(full.capture.focal_px, rel=1e-6)
    cx, cy = f.capture.principal_point
    assert cx / f.scale + f.origin[0] + x0 == pytest.approx(WIDTH / 2, abs=0.5)
    assert cy / f.scale + f.origin[1] + y0 == pytest.approx(HEIGHT / 2, abs=0.5)
    r = measure_centering(f.image, slab=resolve_holder("raw"), capture=f.capture,
                          card_quad=f.quad, quad_residual_px=f.residual_px)
    assert abs(r.worst_ratio.value - gt.worst_ratio) < 1.0


def test_server_reads_the_crop_placement():
    from cardcenter.serve import _parent_crop

    img, _, _ = _shot(650.0)
    ok, jpg = cv2.imencode(".jpg", np.ascontiguousarray(img[1300:2700, 900:2100]))
    b = jpg.tobytes()
    good = {"crop_x": b"900", "crop_y": b"1300", "full_w": b"3000", "full_h": b"4000"}
    assert _parent_crop(good, b) == (900, 1300, 3000, 4000)
    assert _parent_crop({}, b) is None
    assert _parent_crop(dict(good, crop_x=b"x"), b) is None
    # a crop that does not fit inside the photo it claims to come from
    assert _parent_crop(dict(good, full_w=b"1500"), b) is None
    assert _parent_crop(dict(good, crop_y=b"-5"), b) is None


def test_measure_payload_on_a_phone_side_crop_matches_the_whole_photo():
    from cardcenter.serve import _measure_payload

    img, gt, _ = _shot(650.0)
    ok, whole = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    ok2, part = cv2.imencode(".jpg", np.ascontiguousarray(img[1300:2700, 900:2100]),
                             [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    a = _measure_payload(whole.tobytes(), "raw", "main")
    b = _measure_payload(part.tobytes(), "raw", "main", (900, 1300, WIDTH, HEIGHT))
    assert abs(a["ratio"] - b["ratio"]) < 0.6
    assert abs(b["ratio"] - gt.worst_ratio) < 1.0
    assert len(part) < len(whole) / 4          # what the phone no longer uploads


def test_identify_finds_a_card_across_the_counter(tmp_path, monkeypatch):
    from cardcenter.serve import _identify_payload

    monkeypatch.setenv("CARDCENTER_DB", str(tmp_path / "t.db"))
    img, _, _ = _shot(650.0)                 # card ~313 px wide in 3000x4000
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    d = _identify_payload(jpg.tobytes(), "dev-shop", False)
    assert d["ok"] and d["image_size"] == [WIDTH, HEIGHT]
    quad = np.asarray(d["quad"])
    # located in full-photo coordinates, on the card, at the card's scale
    assert np.abs(quad.mean(axis=0) - np.array([WIDTH / 2, HEIGHT / 2])).max() < 40
    assert d["px_per_mm"] == pytest.approx(313 / 63.5, rel=0.12)
