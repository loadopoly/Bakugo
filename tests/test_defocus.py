"""Tests for depth-from-defocus coplanarity checking."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from cardcenter.defocus import (
    LensModel, check_coplanarity, edge_blur_px,
)
from cardcenter.types import DetectionError

LENS = LensModel()


def _blurred_rect(blur_sigma: float, size=(700, 900)) -> tuple[np.ndarray, np.ndarray]:
    h, w = size
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    x0, y0, x1, y1 = 150, 150, 550, 750
    cv2.rectangle(img, (x0, y0), (x1, y1), (220, 220, 220), -1)
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    return img, quad


# --- the lens model -------------------------------------------------------


def test_blur_is_zero_at_the_focal_plane() -> None:
    assert LENS.blur_px(250.0, 250.0) == pytest.approx(0.0)


def test_blur_grows_with_depth_offset() -> None:
    prev = -1.0
    for dz in (2.0, 5.0, 10.0, 25.0):
        b = LENS.blur_px(200.0, 200.0 + dz)
        assert b > prev
        prev = b


def test_close_working_distance_has_real_discrimination() -> None:
    """10mm at 200mm gives ~4px -- measurable. The same 10mm at 500mm gives
    <1px, which is why this only works close up."""
    assert LENS.blur_px(200.0, 210.0) > 3.0
    assert LENS.blur_px(500.0, 510.0) < 1.0


def test_beyond_hyperfocal_there_is_no_depth_information() -> None:
    """The reason this technique was useless for the filament work."""
    h = LENS.hyperfocal_mm
    assert LENS.blur_px(h * 3, h * 3 + 100.0) < 0.5


def test_blur_inversion_round_trips() -> None:
    for dz in (5.0, 10.0, 20.0):
        b = LENS.blur_px(200.0, 200.0 + dz)
        assert LENS.depth_offset_mm(200.0, b) == pytest.approx(dz, rel=0.05)


def test_blur_is_symmetric_so_sign_is_unrecoverable() -> None:
    """Same two-to-one structure as the tangent-sign problem: nearer and
    further produce the same circle."""
    near = LENS.blur_px(200.0, 190.0)
    far = LENS.blur_px(200.0, 210.0)
    assert abs(near - far) / max(near, far) < 0.15


# --- measuring blur from an image ----------------------------------------


def test_measured_blur_tracks_applied_blur() -> None:
    widths = [edge_blur_px(*_blurred_rect(s)) for s in (0.5, 2.0, 5.0)]
    assert widths[0] < widths[1] < widths[2]


def test_refuses_when_there_is_no_contrast() -> None:
    flat = np.full((700, 900, 3), 128, dtype=np.uint8)
    quad = np.array([[150, 150], [550, 150], [550, 750], [150, 750]], dtype=float)
    with pytest.raises(DetectionError, match="high-contrast"):
        edge_blur_px(flat, quad)


# --- the coplanarity check ------------------------------------------------


def _two_object_scene(card_blur: float, ref_blur: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = 800, 1400
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    card = np.array([[100, 150], [520, 150], [520, 700], [100, 700]], dtype=float)
    ref = np.array([[800, 150], [1300, 150], [1300, 500], [800, 500]], dtype=float)

    a = np.zeros_like(img); cv2.fillConvexPoly(a, card.astype(np.int32), (220, 220, 220))
    b = np.zeros_like(img); cv2.fillConvexPoly(b, ref.astype(np.int32), (200, 205, 210))
    if card_blur > 0:
        a = cv2.GaussianBlur(a, (0, 0), card_blur)
    if ref_blur > 0:
        b = cv2.GaussianBlur(b, (0, 0), ref_blur)
    return np.clip(img.astype(int) + a.astype(int) + b.astype(int), 0, 255).astype(np.uint8), card, ref


def test_equal_blur_reads_as_coplanar() -> None:
    img, card, ref = _two_object_scene(1.0, 1.0)
    assert check_coplanarity(img, card, ref, working_distance_mm=200.0).coplanar


def test_unequal_blur_is_caught() -> None:
    """The failure that silently corrupts every dimension downstream."""
    img, card, ref = _two_object_scene(0.6, 5.0)
    c = check_coplanarity(img, card, ref, working_distance_mm=200.0)
    assert not c.coplanar
    assert c.depth_separation_mm is not None and c.depth_separation_mm > 0


def test_detected_mismatch_reports_its_scale_consequence() -> None:
    img, card, ref = _two_object_scene(0.6, 6.0)
    c = check_coplanarity(img, card, ref, working_distance_mm=200.0)
    assert c.scale_error_frac is not None
    assert "same surface" in " ".join(c.warnings)


def test_reports_its_own_resolution() -> None:
    img, card, ref = _two_object_scene(1.0, 1.0)
    c = check_coplanarity(img, card, ref, working_distance_mm=200.0)
    assert c.resolution_mm > 0
    assert "resolves depth to about" in c.describe()


def test_far_working_distance_warns_that_the_check_is_weak() -> None:
    img, card, ref = _two_object_scene(1.0, 1.0)
    c = check_coplanarity(img, card, ref, working_distance_mm=5000.0)
    assert any("hyperfocal" in w for w in c.warnings)
