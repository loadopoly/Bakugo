"""Edge strips from the full-resolution frame (cardcenter/detail.py)."""

import cv2
import numpy as np

from cardcenter.ar import ARSession
from cardcenter.detail import build_mosaic, covers
from cardcenter.synth import render_capture


def _scene():
    # 1080x1920 camera frame, card ~half the width: 8.6 px/mm in the camera
    # frame, 4.3 in the 540 px live frame (below the 4.5 live needs)
    img, gt, f = render_capture(left_mm=3.4, right_mm=2.6, top_mm=3.0, bottom_mm=3.0,
                                distance_mm=175, focal_px=1500, image_size=(1080, 1920),
                                background_bgr=(35, 35, 38), texture_px_per_mm=12, seed=3)
    return img, gt, f


def _strips(full, quad_full, fw, out_mm=6.0, in_mm=11.0):
    """What the phone sends: strips around quad_full (camera pixels), with
    their rectangles in live-frame pixels."""
    H, W = full.shape[:2]
    s = W / fw
    q = np.asarray(quad_full, float)
    ppm = 0.5 * (np.linalg.norm(q[1] - q[0]) / 63 + np.linalg.norm(q[3] - q[0]) / 88)
    c = q.mean(0)
    out = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        t = (b - a) / np.linalg.norm(b - a)
        n = np.array([-t[1], t[0]])
        if n @ ((a + b) / 2 - c) < 0:
            n = -n
        e = 3 * ppm
        pts = np.array([a - t * e + n * out_mm * ppm, b + t * e + n * out_mm * ppm,
                        a - t * e - n * in_mm * ppm, b + t * e - n * in_mm * ppm])
        x0, y0 = np.maximum(np.floor(pts.min(0)), 0)
        x1, y1 = np.minimum(np.ceil(pts.max(0)), [W, H])
        out.append((full[int(y0):int(y1), int(x0):int(x1)].copy(),
                    [x0 / s, y0 / s, (x1 - x0) / s, (y1 - y0) / s]))
    return out, s                                  # m = 1 * s (strips at camera scale)


def test_mosaic_places_strips_and_checks_coverage():
    img, _, _ = _scene()
    H, W = img.shape[:2]
    frame = cv2.resize(img, (540, int(round(H * 540 / W))), interpolation=cv2.INTER_AREA)
    s = ARSession()
    for i in range(3):
        s.push(frame, now=1.0 + 0.3 * i)
    q_frame = s._last_quad
    strips, m = _strips(img, q_frame * (W / 540), 540)
    mos = build_mosaic(frame, strips, m)
    assert mos is not None and abs(mos.m - 2.0) < 1e-6
    assert covers(mos, q_frame)
    # the strips hold the camera's pixels, not the upsampled frame's
    x, y, w, h = strips[0][1]
    ox, oy = mos.origin
    tile = mos.image[int((y - oy) * m) + 4:int((y - oy) * m) + 24, int((x - ox) * m) + 4:int((x - ox) * m) + 24]
    src = img[int(y * 2) + 4:int(y * 2) + 24, int(x * 2) + 4:int(x * 2) + 24]
    assert np.abs(tile.astype(int) - src.astype(int)).mean() < 2.0
    # an outline 15 mm off is not covered
    ppm = np.linalg.norm(q_frame[1] - q_frame[0]) / 63
    assert not covers(mos, q_frame + np.array([15 * ppm, 0]))


def test_live_measures_on_strips_where_the_frame_is_too_coarse():
    img, gt, f = _scene()
    H, W = img.shape[:2]
    frame = cv2.resize(img, (540, int(round(H * 540 / W))), interpolation=cv2.INTER_AREA)
    fov = 2 * np.degrees(np.arctan(W / 2 / f))
    plain = ARSession(fov_deg=fov)
    for i in range(8):
        st_plain = plain.push(frame, now=1.0 + 0.4 * i, source_scale=W / 540)
    assert st_plain.px_per_mm < 4.5 and st_plain.measured_frames == 0
    strips_s = ARSession(fov_deg=fov)
    st = None
    for i in range(8):
        det = None
        if strips_s._last_quad is not None:
            det = _strips(img, strips_s._last_quad * (W / 540), 540)
        st = strips_s.push(frame, now=1.0 + 0.4 * i, source_scale=W / 540, detail=det)
    assert strips_s.detail_used
    assert st.measured_frames >= 1
    truth = max(gt.h_ratio, gt.v_ratio)
    assert abs(st.ratio.value - truth) < 2.0


def test_malformed_strips_fall_back_to_the_frame():
    img, _, _ = _scene()
    H, W = img.shape[:2]
    frame = cv2.resize(img, (540, int(round(H * 540 / W))), interpolation=cv2.INTER_AREA)
    assert build_mosaic(frame, [], 2.0) is None
    assert build_mosaic(frame, [(np.zeros((0, 0, 3), np.uint8), [0, 0, 10, 10])], 2.0) is None
    assert build_mosaic(frame, [(img[:40, :40], [0, 0, 20, 20])], 50.0) is None   # m out of range
