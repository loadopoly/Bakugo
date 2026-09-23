"""Border widths are measured from the outline, so the outline must be ON the
card's edge. From a 2.18.0 field frame: a penny-sleeved Meganium whose top
side sat 0.4 mm off the card, on the start of a soft edge; the top border
then measured 0.7 mm where it is ~2.5-3, and the live session settled on
74.2% (~PSA 6) for a card near 51/54."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cardcenter.centering import _pick_scale, measure_centering, seat_outer_edges
from cardcenter.edge_information import locate_card
from cardcenter.synth import _camera_matrices, render_capture
from cardcenter.types import STANDARD_CARD_H_MM as H, STANDARD_CARD_W_MM as W

PREVIEW = Path(__file__).parent / "fixtures" / "field" / "previews" / "meganium_penny.jpg"


def _quad(tilt, size, off=(0, 0, 0, 0), az=30, dist=260, f=1400):
    l, t, r, b = off
    K, R, C = _camera_matrices(tilt, az, dist, f, size)
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], -R @ C])
    pts = np.array([[-l, -t, 1], [W + r, -t, 1], [W + r, H + b, 1], [-l, H + b, 1]]) @ Hm.T
    return pts[:, :2] / pts[:, 2:]


def _render(borders, tilt, blur, bg):
    L, R, T, B = borders
    img, _, _ = render_capture(left_mm=L, right_mm=R, top_mm=T, bottom_mm=B, tilt_deg=tilt,
                               azimuth_deg=30, distance_mm=260, focal_px=1400,
                               image_size=(1080, 1440), noise_sigma=1.5, background_bgr=bg)
    return cv2.GaussianBlur(img, (0, 0), blur) if blur else img


@pytest.mark.parametrize("off", [(0, 0.5, 0, 0), (0.4, 0, 0, 0), (0, 0, 0.45, 0)])
@pytest.mark.parametrize("tilt,blur", [(0, 0.0), (20, 1.5)])
def test_an_outline_off_the_card_is_seated_on_its_edge(off, tilt, blur):
    """Before: 3.6-11 points out on these (median 4), with nothing refused."""
    borders = (3.4, 2.6, 3.0, 3.0)
    img = _render(borders, tilt, blur, (35, 35, 38))
    res = measure_centering(img, card_quad=_quad(tilt, (1080, 1440), off), keep_rectified=False)
    L, R, T, B = borders
    assert abs(res.horizontal.ratio_pct.value - 100 * max(L, R) / (L + R)) < 2.5
    assert abs(res.vertical.ratio_pct.value - 100 * max(T, B) / (T + B)) < 2.5
    # a gap of background colour between outline and card on one side is
    # also what the edge-shadow check looks for, and it runs first: either
    # way the side ends up on the card, and the result says which moved it
    assert any("outline moved" in w or "shadow" in w for w in res.quality.warnings)


@pytest.mark.parametrize("borders", [(3.4, 2.6, 3.0, 3.0), (1.0, 5.0, 3.0, 3.0)])
@pytest.mark.parametrize("blur", [0.0, 1.5, 3.0])
def test_an_outline_on_the_card_is_left_where_it_is(borders, blur):
    """Including thin borders under heavy blur, where the card's edge ramp
    runs into the border's own and the half-way point slides outward."""
    img = _render(borders, 20, blur, (35, 35, 38))
    q = _quad(20, (1080, 1440))
    _, shift = seat_outer_edges(img, q, _pick_scale(q))
    assert all(v == 0.0 for v in shift.values()), shift


def test_the_settled_field_frame_measures_near_centred():
    img = cv2.imread(str(PREVIEW))
    assert img is not None
    q, _, r, _, _ = locate_card(img)
    res = measure_centering(img, slab="penny_sleeve", card_quad=q, quad_residual_px=r,
                            keep_rectified=False)
    assert 48.0 <= res.horizontal.ratio_pct.value <= 56.0
    assert 48.0 <= res.vertical.ratio_pct.value <= 60.0, res.vertical.ratio_pct
    assert res.vertical.low_mm.value > 2.0            # the top border, not 0.7
    assert any("outline moved" in w and "top" in w for w in res.quality.warnings)
