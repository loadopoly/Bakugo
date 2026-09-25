"""The reader on real shop-style frames (tests/fixtures/field).

These are the owner's own frames from live sessions: cards on a light wood
counter at 17-40 degrees, sleeved, touching, one under another, a thumb over an
edge, motion blur. The bar is set at what the scene search achieves on them
today, with a little room; a change that loses cards here is a regression in
the thing the app is for, whatever the synthetic tests say.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from cardcenter.ar import ARSession
from cardcenter.edge_information import locate_card
from cardcenter.geometry import find_card_quad, order_quad, enforce_portrait
from cardcenter.scene import SceneSearch, card_pose, find_card_at
from cardcenter.types import DetectionError, STANDARD_CARD_H_MM, STANDARD_CARD_W_MM

FIELD = Path(__file__).parent / "fixtures" / "field"
ANN = json.loads((FIELD / "annotations.json").read_text())["frames"]


def _img(name):
    img = cv2.imread(str(FIELD / ANN[name]["file"]))
    assert img is not None, name
    return img


def _iou(a, b):
    a = np.asarray(a, np.float32).reshape(4, 2)
    b = np.asarray(b, np.float32).reshape(4, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = abs(cv2.contourArea(a)) + abs(cv2.contourArea(b)) - inter
    return inter / union if union > 0 else 0.0


def _cards(name, states=("complete", "finger_over_edge", "corner_hidden"), known_limits=False):
    return {n: np.array(c["quad"], float) for n, c in ANN[name]["cards"].items()
            if c["state"] in states and None not in c["quad"]
            and (known_limits or not c.get("known_limit"))}


def _floor(frame, sharp=0.8):
    """IoU a found outline must reach. The counter_* frames are badly out of
    focus and their outlines were drawn to about +/-6 px, not +/-3."""
    return 0.75 if ANN[frame].get("soft") else sharp


def _aims(quad):
    """Twelve points spread over the card face: where a user's reticle lands."""
    for u in (0.25, 0.5, 0.75):
        for v in (0.2, 0.4, 0.6, 0.8):
            top = quad[0] + u * (quad[1] - quad[0])
            bot = quad[3] + u * (quad[2] - quad[3])
            yield top + v * (bot - top)


ALL = [(f, c) for f in ANN for c in _cards(f)]


def test_the_contour_detector_alone_misses_these():
    """Why the scene search exists: record the old detector's result, so a
    future 'simplification' back to it is seen for what it is."""
    hits = 0
    # the shop-counter frames; on a black mat (the 'reader' frames of
    # 2.18.0) the contour detector is at home and finds the card
    for frame, card in [fc for fc in ALL if not ANN[fc[0]].get("reader")]:
        q = _cards(frame)[card]
        try:
            found, _, _ = find_card_quad(_img(frame), prefer_point=tuple(q.mean(0)))
            hits += _iou(found, q) >= 0.85
        except DetectionError:
            pass
    assert hits <= 2, hits


@pytest.mark.parametrize("frame,card", ALL)
def test_locate_card_finds_the_card_it_is_aimed_at(frame, card):
    q = _cards(frame)[card]
    found, _, _, _, _ = locate_card(_img(frame), prefer_point=tuple(q.mean(0)))
    # the left card of desk_pair is the softest frame in the set (motion
    # blur on a sleeve edge against wood)
    floor = 0.7 if (frame, card) == ("desk_pair_inset", "meganium_sleeved") else _floor(frame)
    assert _iou(found, q) >= floor


def test_aiming_anywhere_on_the_card_finds_the_whole_card():
    """The reticle lands on the artwork as often as not, and the art window
    of most cards is itself a clean card-shaped rectangle."""
    good = total = 0
    worst = {}
    for frame, card in ALL:
        S = SceneSearch(_img(frame))
        q = _cards(frame)[card]
        n = 0
        for p in _aims(q):
            hit = S.card_at(tuple(p))
            ok = hit is not None and _iou(hit.quad, q) >= _floor(frame)
            good += ok
            n += ok
            total += 1
        worst[(frame, card)] = n
    assert good >= 0.85 * total, (good, total, worst)
    # every card is found from most of its face, except the blurriest
    for key, n in worst.items():
        if key != ("desk_pair_inset", "meganium_sleeved"):
            assert n >= 9, (key, n)


def test_the_card_under_another_is_not_merged_with_it():
    """desk_spread: a card lies on top of another, aligned. The pair makes a
    bigger card-shaped quad; aiming at the top card must give the top card.
    Holds at the full frame size; at the live 540 px size it does not (see
    test_live_path_finds_each_card_from_anywhere_on_it)."""
    q = _cards("desk_spread_inset")["grass_energy_top"]
    hit = find_card_at(_img("desk_spread_inset"), tuple(q.mean(0)))
    assert _iou(hit.quad, q) >= 0.85


def test_reticle_on_the_table_takes_the_nearest_card():
    img = _img("desk_spread_inset")          # centre of the frame is bare wood
    h, w = img.shape[:2]
    hit = find_card_at(img)
    q = _cards("desk_spread_inset")["meganium_sleeved"]
    assert _iou(hit.quad, q) >= 0.85


def test_all_cards_on_the_counter():
    for frame in ("desk_spread_inset", "desk_close_inset", "hand_ar1"):
        found = SceneSearch(_img(frame)).all_cards()
        for card, q in _cards(frame, ("complete",)).items():
            assert any(_iou(h.quad, q) >= 0.8 for h in found), (frame, card)


def test_tilt_is_reported_for_cards_on_the_counter():
    """desk_spread was taken with the phone at ~40 degrees (its gyro)."""
    q = _cards("desk_spread_inset")["meganium_sleeved"]
    hit = find_card_at(_img("desk_spread_inset"), tuple(q.mean(0)))
    assert 15 <= hit.tilt_deg <= 50
    assert abs(hit.aspect - STANDARD_CARD_H_MM / STANDARD_CARD_W_MM) < 0.16


@pytest.mark.parametrize("frame", ["desk_spread_inset", "desk_close_inset", "hand_ar1", "hand_ar2"])
def test_live_session_tracks_a_card_on_real_frames(frame):
    img = _img(frame)
    s = ARSession()
    st = None
    for i in range(4):
        st = s.push(img, now=1.0 + 0.2 * i)
    assert st.tracking and st.quad is not None
    assert max(_iou(st.quad, q) for q in _cards(frame).values()) >= 0.8


@pytest.mark.parametrize("quality", [75, 92])
def test_live_path_finds_each_card_from_anywhere_on_it(quality):
    """End to end the way the phone does it: the frame JPEG-encoded at the
    app's quality (0.75) or better, a tap somewhere on the card, two pushes.
    Nine tap points per card."""
    per = {}
    for frame in ANN:
        img = _img(frame)
        ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        im = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        for card, q in _cards(frame).items():
            n = 0
            for u in (0.25, 0.5, 0.75):
                for v in (0.2, 0.5, 0.8):
                    top = q[0] + u * (q[1] - q[0])
                    bot = q[3] + u * (q[2] - q[3])
                    p = top + v * (bot - top)
                    s = ARSession()
                    s.select((p[0] / w, p[1] / h))
                    s.push(im, now=1.0)
                    st = s.push(im, now=1.3)
                    n += bool(st.tracking and st.quad is not None and _iou(st.quad, q) >= 0.75)
            per[(frame, card)] = n
    # KNOWN LIMIT: desk_spread's top card lies exactly on another with the
    # sides aligned. At the live 540 px size the pair is a card-shaped quad
    # that is better supported than the top card alone, and nothing local
    # separates them (see the changelog for 2.17.0). Recorded, not asserted.
    stacked = per.pop(("desk_spread_inset", "grass_energy_top"))
    assert all(n >= 8 for n in per.values()), (per, stacked)


def test_tap_picks_the_card():
    """Tap the card on the left of desk_pair while the reticle is between."""
    img = _img("desk_pair_inset")
    h, w = img.shape[:2]
    q = _cards("desk_pair_inset")["meganium_sleeved"]
    c = q.mean(0)
    s = ARSession()
    s.push(img, now=1.0)
    s.select((c[0] / w, c[1] / h))
    st = s.push(img, now=1.2)
    assert st.tracking and _iou(st.quad, q) >= 0.75


def test_too_far_for_live_points_at_the_photo():
    """At a counter the 540 px live frame is too coarse, but the phone's
    camera has 4x the pixels: the guidance should say to take the photo,
    not to walk closer."""
    img = _img("desk_spread_inset")
    s = ARSession()
    st = None
    for i in range(3):
        st = s.push(img, now=1.0 + 0.2 * i, source_scale=4.0)
    text = " ".join(st.guidance)
    assert "Freeze" in text and "photo" in text and "zoom in" in text, st.guidance
    s2 = ARSession()
    for i in range(3):
        st2 = s2.push(img, now=1.0 + 0.2 * i)
    assert any(g.startswith("too far away") for g in st2.guidance)


# ---- 2.17.0 at the counter: what the app drew, and why -------------------
#
# The counter_* frames are the owner's screenshots of 2.17.0 at a counter
# (tests/fixtures/field/README.md). In every one the app said TRACKING over an
# outline that was not the card -- and nearly the same outline in four of
# them, over three different cards on two different surfaces: the tracker was
# holding a place on the screen, not a card.

COUNTER = [f for f in ANN if "shown" in ANN[f]]


def _shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


@pytest.mark.parametrize("frame", COUNTER)
def test_an_outline_that_is_not_a_card_is_let_go(frame):
    """Start the session where 2.17.0 was -- holding the outline it drew --
    and give it the frame: it must drop that outline, and whatever it shows
    instead must be a card that is there (or nothing)."""
    img = _img(frame)
    shown = np.array(ANN[frame]["shown"], float)
    s = ARSession()
    s._last_raw = shown.copy()
    s._last_quad = shown.copy()
    s._last_valid_at = s._last_detect_at = 0.9
    st = s.push(img, now=1.0)
    if not st.tracking:
        assert st.quad is None and st.guidance
        return
    assert _iou(st.quad, shown) < 0.7
    cards = _cards(frame, known_limits=True)
    assert cards and max(_iou(st.quad, q) for q in cards.values()) >= 0.7, st.quad.tolist()


@pytest.mark.parametrize("frame", ["counter_meganium_a", "cloth_terapagos"])
def test_half_a_card_is_not_a_card(frame):
    """Too close, the bottom of the card is out of view, and the top border,
    the sides and the bottom of the art window make a clean quad that is
    card-shaped lying sideways. That is not a card to measure; the user is
    told to show all of it."""
    img = _img(frame)
    h, w = img.shape[:2]
    S = SceneSearch(img)
    assert S.card_at((w / 2, h / 2)) is None
    assert "whole card not in view" in S.refusal
    with pytest.raises(DetectionError, match="whole card not in view"):
        locate_card(img, prefer_point=(w / 2, h / 2))
    # and live, the way the phone sends it (JPEG 0.75, a few pushes)
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    live = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    s = ARSession()
    for i in range(3):
        st = s.push(live, now=1.0 + 0.3 * i, source_scale=3.6)
        assert not st.tracking and "whole card not in view" in " ".join(st.guidance)


@pytest.mark.parametrize("frame,card", [("counter_stack", "meganium_sleeved"),
                                        ("desk_close_inset", "meganium_sleeved"),
                                        ("desk_pair_inset", "dreepy_card")])
def test_the_outline_follows_the_card_when_the_phone_moves(frame, card):
    """A few pushes a second from a hand-held phone: the card moves tens of
    pixels between frames, past the edge search's reach. The outline must go
    with the card -- not stay where it was, and not jump to its neighbour."""
    img = _img(frame)
    q = _cards(frame)[card]
    h, w = img.shape[:2]
    c = q.mean(axis=0)
    s = ARSession()
    s.select((c[0] / w, c[1] / h))
    t = 1.0
    st = s.push(img, now=t)
    assert st.tracking and _iou(st.quad, q) >= 0.75
    for dx, dy in [(25, 0), (50, 10), (80, 20), (40, -30), (0, 0)]:
        t += 0.35
        st = s.push(_shift(img, dx, dy), now=t)
        moved = q + np.array([dx, dy], float)
        assert st.tracking, (dx, dy, st.guidance)
        assert _iou(st.quad, moved) >= 0.75, (dx, dy, st.quad.tolist())


def test_a_tap_names_a_card_once():
    """The tap is a point in the frame it was made on. Once it has found its
    card, the session follows the card; it does not keep re-aiming at that
    spot of the screen as the phone moves."""
    img = _img("desk_pair_inset")
    q = _cards("desk_pair_inset")["dreepy_card"]
    h, w = img.shape[:2]
    c = q.mean(axis=0)
    s = ARSession()
    s.select((c[0] / w, c[1] / h))
    st = s.push(img, now=1.0)
    assert st.tracking and s._aim_norm is None


def test_a_soft_frame_of_a_close_card_says_to_lift_the_phone():
    # desk_close: the card is ~60% of the frame wide and the frame is soft
    img = _img("desk_close_inset")
    st = None
    s = ARSession()
    for i in range(2):
        st = s.push(img, now=1.0 + 0.3 * i)
    assert st.tracking
    assert any("too close to focus" in g for g in st.guidance), st.guidance


# ---- synthetic: the scene search is as precise as it needs to be ----------

@pytest.mark.parametrize("tilt", [0, 20, 35, 45])
@pytest.mark.parametrize("background", [(35, 35, 38), (120, 150, 170)])
def test_scene_search_corners_on_synthetic_cards(tilt, background):
    from cardcenter.synth import _camera_matrices, render_capture

    size, f, dist, az = (1080, 1440), 1400.0, 300.0, 30.0
    img, _, _ = render_capture(left_mm=3.4, right_mm=2.6, tilt_deg=tilt, azimuth_deg=az,
                               distance_mm=dist, focal_px=f, image_size=size,
                               noise_sigma=2.0, background_bgr=background)
    K, R, C = _camera_matrices(tilt, az, dist, f, size)
    H = K @ np.column_stack([R[:, 0], R[:, 1], -R @ C])
    W, Hm = STANDARD_CARD_W_MM, STANDARD_CARD_H_MM
    pts = np.array([[0, 0, 1], [W, 0, 1], [W, Hm, 1], [0, Hm, 1]]) @ H.T
    gt = enforce_portrait(order_quad(pts[:, :2] / pts[:, 2:]))
    hit = find_card_at(img, tuple(gt.mean(0)))
    q = enforce_portrait(order_quad(hit.quad))
    err = min(np.abs(np.roll(q, k, 0) - gt).max() for k in range(4))
    assert err < 4.0, err                     # 1440 px frame, 720 px working size
    assert abs(hit.tilt_deg - tilt) < 12 or tilt == 0


def test_card_pose_recovers_aspect_through_the_tilt():
    from cardcenter.synth import _camera_matrices

    for tilt in (0, 25, 45):
        K, R, C = _camera_matrices(tilt, 40.0, 350.0, 900.0, (540, 632))
        H = K @ np.column_stack([R[:, 0], R[:, 1], -R @ C])
        W, Hm = STANDARD_CARD_W_MM, STANDARD_CARD_H_MM
        pts = np.array([[0, 0, 1], [W, 0, 1], [W, Hm, 1], [0, Hm, 1]]) @ H.T
        q = pts[:, :2] / pts[:, 2:]
        aspect, est_tilt = card_pose(q, (632, 540), fov_deg=2 * math.degrees(math.atan(270 / 900)))
        assert abs(aspect - Hm / W) < 0.03
        assert abs(est_tilt - tilt) < 3.0
