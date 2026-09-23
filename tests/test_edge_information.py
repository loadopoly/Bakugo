"""The information floor along a detected outline, and its use in AR."""

import cv2
import numpy as np
import pytest

from cardcenter.ar import ARSession, _resize_long, track_quad
from cardcenter.edge_information import (capture_advice, locate_card, quad_information,
                                         refine_quad_by_information, snap_to_information)
from cardcenter.geometry import find_card_quad
from cardcenter.synth import render_capture


def _scene(pad=0.0):
    img = np.full((600, 800, 3), 40, np.uint8)
    q = np.array([[250, 120], [520, 120], [520, 500], [250, 500]], float)
    cv2.fillConvexPoly(img, q.astype(np.int32), (230, 225, 215))
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
    halo = q + np.array([[-pad, -pad], [pad, -pad], [pad, pad], [-pad, pad]])
    return img, q, halo


def test_floor_on_a_clean_outline():
    img, q, _ = _scene()
    info = quad_information(img, q)
    assert info.resolvable and not info.unresolved_sides
    assert all(abs(s.offset_px) < 1.0 for s in info.sides)
    assert all(s.channel.contrast > 150 for s in info.sides)
    assert all(np.isfinite(c) and c < 0.1 for c in info.corner_sigma_px)
    d = info.to_dict()
    assert set(d) >= {"resolvable", "corner_sigma_px", "sigma_cr_pp", "advice", "sides"}


def test_halo_is_found_and_removed():
    img, q, halo = _scene(pad=7.0)
    info = quad_information(img, halo)
    assert all(-8.0 < s.offset_px < -6.0 for s in info.sides)
    fixed = refine_quad_by_information(halo, info)
    assert np.abs(fixed - q).max() < 1.5
    snapped, info2, moved = snap_to_information(img, halo)
    assert moved and np.abs(snapped - q).max() < 1.5
    assert all(abs(s.offset_px) < 1.5 for s in info2.sides)


def test_invisible_side_blocks_refinement():
    img, q, _ = _scene()
    # paint the right edge region the card colour: no edge there any more
    img[100:520, 505:560] = (230, 225, 215)
    info = quad_information(img, q)
    assert "right" in info.unresolved_sides
    assert refine_quad_by_information(q, info) is None
    assert any(a.key == "contrast" for a in info.advice)


def test_noise_and_blur_raise_the_floor_and_rank_advice():
    img, q, _ = _scene()
    base = quad_information(img, q)
    rng = np.random.default_rng(3)
    noisy = np.clip(img.astype(float) + rng.normal(0, 30, img.shape), 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img, (0, 0), 3.0)
    n = quad_information(noisy, q)
    b = quad_information(blurred, q)
    assert n.worst_corner_sigma_px > 3 * base.worst_corner_sigma_px
    assert b.worst_corner_sigma_px > base.worst_corner_sigma_px
    assert any(a.key == "steady" for a in b.advice)
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    s = quad_information(small, q * 0.25)
    assert s.advice and s.advice[0].key == "closer"
    gains = [a.gain for a in s.advice]
    assert gains == sorted(gains, reverse=True)


def test_locate_card_fixes_tracking_resolution_halo():
    # Synthetic captures at the 540 px tracking size: find_card_quad's outline
    # lands about one border width outside the card; the snap brings it back.
    rng = np.random.default_rng(0)
    before, after = [], []
    for seed in range(4):
        img, _, _ = render_capture(left_mm=3.4, right_mm=2.6, tilt_deg=float(rng.uniform(4, 18)),
                                   azimuth_deg=float(rng.uniform(0, 360)), noise_sigma=3.0, seed=seed)
        full, _, _ = find_card_quad(img)
        small, s = _resize_long(img, 540)
        centre = (small.shape[1] / 2, small.shape[0] / 2)
        raw, _, _ = find_card_quad(small, prefer_point=centre)
        snapped, _, _, info, _ = locate_card(small, prefer_point=centre)
        before.append(np.abs(raw / s - full).max())
        after.append(np.abs(snapped / s - full).max())
    assert max(after) < 5.0
    assert np.mean(after) < np.mean(before)


def test_track_quad_reports_achieved_uncertainty():
    img, q, _ = _scene()
    out, stats = track_quad(img, q + 1.0, return_stats=True)
    assert np.abs(out - q).max() < 2.0
    assert len(stats.side_rms_px) == 4 and all(n >= 6 for n in stats.side_points)
    assert 0 < stats.corner_sigma_px < 1.0


def test_ar_status_carries_information_and_decision():
    s = ARSession()
    now = 0.0
    status = None
    for i in range(8):
        img, _, _ = render_capture(left_mm=3.4, right_mm=2.6, tilt_deg=6.0, noise_sigma=2.0, seed=i)
        now += 0.4
        status = s.push(img, now=now)
    assert status.tracking and status.information is not None
    assert status.information["resolvable"] is True
    assert 3.0 <= status.information["search_px"] <= 14.0
    assert status.measured_frames >= 2
    assert status.decision in ("accept", "review", "reject")
    assert status.decision_reason


def test_ar_refuses_to_measure_an_unresolvable_outline(monkeypatch):
    from cardcenter import edge_information as ei

    real = ei.quad_information

    def fake(image, quad, **kw):
        info = real(image, quad, **kw)
        sides = tuple(type(sd)(**{**sd.__dict__, "resolvable": sd.side != "left"}) for sd in info.sides)
        return ei.QuadInformation(sides, info.corner_sigma_px, info.px_per_mm,
                                  info.sigma_cr_pp, info.shot_ratio, info.advice)

    monkeypatch.setattr(ei, "quad_information", fake)
    s = ARSession()
    now = 0.0
    for i in range(4):
        img, _, _ = render_capture(left_mm=3.4, right_mm=2.6, tilt_deg=6.0, noise_sigma=2.0, seed=i)
        now += 0.4
        status = s.push(img, now=now)
    assert status.measured_frames == 0
    assert any("left edge not visible" in g for g in status.guidance)
