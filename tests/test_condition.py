"""Corners, edges, surface and the whole-card grade (2.24).

The synthetic cards here have rounded die-cut corners and known wear
(``synth.apply_wear``). As with every synthetic test in this repository they
check the geometry and the bookkeeping, not real-card accuracy: print,
foil, JPEG and a phone's sharpening are absent. ``data/condition_standards.json``
says so too.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cardcenter.centering import measure_centering
from cardcenter.condition import (
    CORNERS,
    SIDES,
    ConditionView,
    CornerReading,
    EdgeReading,
    SurfaceReading,
    analyse_view,
    classify_face,
    side_face_mm,
)
from cardcenter.condition_evidence import ConditionEvidence
from cardcenter.geometry import find_card_quad
from cardcenter.grading import (
    AspectGrade,
    compose,
    corner_grade_rule,
    predict_overall_grade,
)
from cardcenter.synth import render_capture
from cardcenter.types import CaptureSpec, Measured

WEAR = dict(
    corner_loss_mm={"top_right": 0.9},
    corner_white_mm={"bottom_left": 0.4},
    edge_white=[("left", 20.0, 60.0, 0.3)],
    nicks=[("bottom", 30.0, 1.0, 0.5)],
)


def _view(wear=None, tilt=15.0, az=40.0, focal=2600.0, distance=200.0, size=(1600, 2000),
          geometry_from_clean=False, **kw):
    render = dict(left_mm=3.0, right_mm=3.0, tilt_deg=tilt, azimuth_deg=az, image_size=size,
                  focal_px=focal, distance_mm=distance, **kw)
    img, gt, f = render_capture(wear=wear, **render)
    cap = CaptureSpec(focal_px=f)
    if geometry_from_clean:
        # a big highlight defeats the centering measurement; the card's
        # outline is the same as in the same view without it
        clean_img, _, _ = render_capture(wear={}, **render)
        res = measure_centering(clean_img, capture=cap)
    else:
        res = measure_centering(img, capture=cap)
    return analyse_view(img, res.corners_px, None, res.inner_rect_mm, capture=cap), res


@pytest.fixture(scope="module")
def worn():
    return _view(WEAR)


@pytest.fixture(scope="module")
def clean():
    return _view({})


def test_a_clean_card_reads_clean(clean) -> None:
    v, res = clean
    assert v.px_per_mm > 10
    for name, c in v.corners.items():
        assert c.resolvable, (name, c.reason)
        assert c.loss_mm < 0.15, name
        assert c.whitening_mm < 0.1, name
    for side, e in v.edges.items():
        assert e.resolvable, (side, e.reason)
        assert e.whitening_frac < 0.05, side
        assert e.nick_max_mm < 0.12, side


def test_a_rounded_corner_is_measured(worn) -> None:
    v, _ = worn
    tr = v.corners["top_right"]
    assert tr.resolvable
    assert tr.loss_mm == pytest.approx(0.9, abs=0.2)
    assert v.corners["top_left"].loss_mm < 0.15


def test_corner_whitening_is_measured(worn) -> None:
    v, _ = worn
    bl = v.corners["bottom_left"]
    assert bl.whitening_resolvable
    assert bl.whitening_mm == pytest.approx(0.4, abs=0.15)


def test_edge_whitening_and_a_nick_are_measured(worn) -> None:
    v, _ = worn
    # 40 mm of the left side's 80.9 mm read length
    assert v.edges["left"].whitening_frac == pytest.approx(40 / 80.9, abs=0.12)
    assert v.edges["bottom"].nick_count >= 1
    assert v.edges["bottom"].nick_max_mm == pytest.approx(0.5, abs=0.15)
    assert v.edges["top"].nick_count == 0


def test_too_coarse_is_not_assessed_rather_than_clean() -> None:
    v, res = _view(WEAR, size=(800, 1000), focal=1300.0, distance=400.0)
    # the rectification scale is clipped at 6; the card's own is lower
    assert v.px_per_mm < 6
    assert not any(c.resolvable for c in v.corners.values())
    assert all("px/mm" in c.reason for c in v.corners.values())
    pred = predict_overall_grade(res.worst_ratio, condition=v)
    assert not pred.complete
    assert pred.estimated_corners is None and pred.estimated_edges is None
    assert pred.grade_label == "PSA not graded"
    assert not pred.graded
    assert "corners" in pred.missing


def test_unassessed_aspects_are_never_assumed_clean() -> None:
    """The 2.23 behaviour this replaces: corners/edges/surface started at 10
    and were docked by detection quality, capped at 2 points -- so every
    soft frame read PSA 8."""
    pred = predict_overall_grade(Measured(51.0, 0.2), grader="PSA")
    assert pred.estimated_corners is None
    assert pred.estimated_edges is None
    assert pred.estimated_surface is None
    assert not pred.complete
    assert set(pred.missing) == {"corners", "edges", "surface"}
    assert "not assessed" in pred.describe()


def test_measured_wear_pulls_the_grade_down(worn, clean) -> None:
    vw, rw = worn
    vc, rc = clean
    pw = predict_overall_grade(rw.worst_ratio, condition=vw)
    pc = predict_overall_grade(rc.worst_ratio, condition=vc)
    assert pw.aspects["corners"].modal <= 5.0
    assert pc.aspects["corners"].modal >= 9.0
    assert pw.grade_score < pc.grade_score
    assert pw.estimated_corners is not None


def test_probabilities_are_a_distribution(worn) -> None:
    v, res = worn
    p = predict_overall_grade(res.worst_ratio, condition=v)
    assert sum(p.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert p.confidence == pytest.approx(max(p.probabilities.values()), abs=0.011)


# -- composition ---------------------------------------------------------------


def test_weakest_aspect_governs_exactly() -> None:
    a = AspectGrade("a", "front", {10.0: 0.5, 9.0: 0.5})
    b = AspectGrade("b", "front", {10.0: 0.8, 8.0: 0.2})
    d = compose([a, b], "PSA")
    # P(min = 10) = .5*.8 ; P(min = 9) = .5*.8 ; P(min = 8) = .2
    assert d[10.0] == pytest.approx(0.4)
    assert d[9.0] == pytest.approx(0.4)
    assert d[8.0] == pytest.approx(0.2)


def test_corner_rule_follows_the_published_wording() -> None:
    assert corner_grade_rule([4, 0, 0, 0, 0, 0]) == 10.0   # four sharp corners
    assert corner_grade_rule([3, 1, 0, 0, 0, 0]) == 9.0    # one minor flaw
    assert corner_grade_rule([2, 2, 0, 0, 0, 0]) == 8.0    # fraying at one or two
    assert corner_grade_rule([0, 4, 0, 0, 0, 0]) == 7.0    # some corners
    assert corner_grade_rule([3, 0, 0, 1, 0, 0]) == 5.0    # rounding evident


# -- many views ----------------------------------------------------------------


def _corner(name, white, res=12.0):
    return CornerReading(name=name, resolvable=True, px_per_mm=res, loss_mm=0.0,
                         whitening_mm=white, whitening_resolvable=True, sigma_mm=0.06)


def _edge(side, frac=0.0, nick=0.0):
    return EdgeReading(side=side, resolvable=True, px_per_mm=12.0, whitening_frac=frac,
                       whitening_resolvable=True, usable_frac=1.0, nick_max_mm=nick)


def _fake_view(face="front", white=None, edge_frac=None):
    white = white or {}
    edge_frac = edge_frac or {}
    return ConditionView(
        face=face, face_confidence=1.0, px_per_mm=12.0,
        corners={c: _corner(c, white.get(c, 0.0)) for c in CORNERS},
        edges={s: _edge(s, edge_frac.get(s, 0.0)) for s in SIDES},
        surface=SurfaceReading(resolvable=False, reason="test"))


def test_glare_in_one_view_does_not_become_whitening() -> None:
    """Whitening can only be ADDED by glare or a side face, so the pooled
    estimate follows the low end of the views, not the mean."""
    ev = ConditionEvidence()
    for w in (0.0, 0.0, 0.0, 0.7):
        ev.add(_fake_view(white={"top_left": w}, edge_frac={"top": 0.9 if w else 0.0}))
    est = ev.faces["front"].corner("top_left")
    assert est.value < 0.1
    assert ev.faces["front"].edge_whitening("top").value < 0.1


def test_back_corners_pair_with_the_mirrored_front_corner() -> None:
    ev = ConditionEvidence()
    ev.add(_fake_view("front"))
    # the back's top-LEFT is the front's top-RIGHT
    ev.add(_fake_view("back", white={"top_left": 0.6}))
    pred = predict_overall_grade(Measured(51.0, 0.2), condition=ev)
    corners = pred.aspects["corners"]
    assert corners.assessed == "both"
    assert "TR fraying" in corners.detail or "TR minor rounding" in corners.detail
    assert corners.modal <= 7.0


def test_surface_needs_coverage_before_it_is_assessed() -> None:
    ev = ConditionEvidence()
    ev.add(_fake_view())
    summ = ev.summary()
    assert not summ["front"]["surface"]["assessed"]
    pred = predict_overall_grade(Measured(51.0, 0.2), condition=ev)
    assert "surface" in pred.missing
    assert "tilt" in ev.next_action()


# -- surface: a scratch is in the reflection, not in the print -------------------


def _surface_views(scratch=True, printed=False):
    from cardcenter.synth import apply_wear  # noqa: F401  (documented entry point)

    views = []
    scratches = [(20.0, 40.0, 34.0, 44.0)] if scratch else []
    for gx in (22.0, 30.0, 40.0):
        wear = {"glare": [(gx, 42.0, 34.0, 1.0)], "scratches": scratches}
        v, _ = _view(wear, tilt=8.0, az=30.0, geometry_from_clean=True)
        views.append(v)
    return views


def test_a_scratch_seen_under_two_highlights_is_a_defect() -> None:
    ev = ConditionEvidence()
    for v in _surface_views():
        ev.add(v)
    defects = ev.faces["front"].surface.defects()
    assert any(abs(d["x_mm"] - 27) < 6 and abs(d["y_mm"] - 42) < 4 for d in defects)


def test_printed_ink_under_a_highlight_is_not_a_scratch() -> None:
    """Dark ink inside a highlight looks like a scratch in that view. The
    same place seen without the highlight shows the ink, so it is print."""
    ev = ConditionEvidence()
    line = [(20.0, 40.0, 34.0, 44.0)]
    for g in ((24.0, 42.0), (30.0, 42.0), (45.0, 78.0), (12.0, 75.0)):
        v, _ = _view({"glare": [(g[0], g[1], 26.0, 1.0)], "ink": line}, tilt=8.0, az=30.0,
                     geometry_from_clean=True)
        ev.add(v)
    assert ev.faces["front"].surface.defects() == []


def test_no_scratch_no_defect() -> None:
    ev = ConditionEvidence()
    for v in _surface_views(scratch=False):
        ev.add(v)
    assert ev.faces["front"].surface.defects() == []


# -- face and pose ---------------------------------------------------------------


def _canon_back():
    import cv2

    R, M = 8.0, 2.5
    W, H = 63.5, 88.9
    img = np.zeros((int((H + 2 * M) * R), int((W + 2 * M) * R), 3), np.uint8)
    img[:] = (60, 60, 60)
    img[int(M * R):int((M + H) * R), int(M * R):int((M + W) * R)] = (170, 70, 20)   # blue
    cx, cy = int((M + W / 2) * R), int((M + H / 2) * R)
    cv2.ellipse(img, (cx, cy), (int(14 * R), int(14 * R)), 0, 180, 360, (30, 30, 210), -1)  # red top half
    cv2.ellipse(img, (cx, cy), (int(14 * R), int(14 * R)), 0, 0, 180, (240, 240, 240), -1)
    return img, R, M


def test_the_back_is_recognised_and_blue_art_is_not() -> None:
    img, R, M = _canon_back()
    assert classify_face(img, R, M)[0] == "back"
    front = img.copy()
    # same blue face, but a yellow border ring and no red disc: a water-type front
    b = int((M + 2.5) * R)
    front[int(M * R):int((M + 88.9) * R), int(M * R):int((M + 63.5) * R)] = (60, 200, 240)
    front[b:int((M + 86.4) * R), b:int((M + 61.0) * R)] = (170, 70, 20)
    assert classify_face(front, R, M)[0] == "front"


def test_side_face_and_tilt_come_from_the_outline_even_with_a_wrong_focal() -> None:
    img, gt, f = render_capture(tilt_deg=30.0, azimuth_deg=90.0, focal_px=2400.0)
    q, _, _ = find_card_quad(img)
    sf, tilt = side_face_mm(q, CaptureSpec(focal_px=1500.0), img.shape)
    assert tilt == pytest.approx(30.0, abs=3.0)
    # camera beyond the bottom edge: only the bottom side face shows
    assert sf["bottom"] > 0.05
    assert sf["top"] == 0.0


# -- the live session ------------------------------------------------------------


def test_ar_session_reports_every_aspect() -> None:
    from cardcenter.ar import ARSession

    rng = np.random.default_rng(1)
    s = ARSession()
    st = None
    for i in range(5):
        img, _, _ = render_capture(left_mm=3.0, right_mm=3.0,
                                   tilt_deg=float(rng.uniform(5, 18)),
                                   azimuth_deg=float(rng.uniform(0, 360)),
                                   noise_sigma=2.0, seed=i, wear={"corner_loss_mm": {"top_right": 0.9}})
        st = s.push(img, now=0.4 * (i + 1))
    assert st.aspects is not None
    assert set(st.aspects) == {"centering", "corners", "edges", "surface"}
    assert st.aspects["corners"]["assessed"] == "front"
    assert float(st.aspects["corners"]["grade"]) <= 5.0
    assert st.aspects["surface"]["assessed"] == "no"
    assert not st.grade_complete
    assert st.grade_graded                      # corners and edges were read
    assert st.grade_estimate.endswith("max")    # surface was not
    assert st.condition_hint
    s.reset()
    assert s.condition.n_views == 0


def test_a_card_turned_over_keeps_its_front_evidence() -> None:
    """Turning a card over moves its outline enough to read as a new card;
    the first view of the OTHER face within FLIP_WINDOW_S carries the
    evidence across, so both faces grade together."""
    from cardcenter.ar import ARSession

    s = ARSession()
    s.add_condition(_fake_view("front"), now=10.0)
    s.horizontal.add(Measured(52.0, 0.5))
    s._new_card(now=11.0)
    assert s.condition.n_views == 0 and s.worst_ratio is None
    s.add_condition(_fake_view("back"), now=13.0)
    assert s.condition.has("front") and s.condition.has("back")
    assert s.worst_ratio is not None


def test_a_different_card_does_not_inherit_evidence() -> None:
    from cardcenter.ar import FLIP_WINDOW_S, ARSession

    s = ARSession()
    s.add_condition(_fake_view("front"), now=10.0)
    s._new_card(now=11.0)
    s.add_condition(_fake_view("front"), now=12.0)        # same face: a new card
    assert s.condition.n_views == 1
    s._new_card(now=13.0)
    s.add_condition(_fake_view("back"), now=13.0 + FLIP_WINDOW_S + 1.0)   # too late
    assert not s.condition.has("front")


def test_centering_alone_is_never_shown_as_a_grade(clean) -> None:
    """2.24.0 labelled a card whose corners and edges were unreadable
    "PSA 10 max"; read as a grade of 10, it was wrong about a card nobody
    had seen the corners of. Without corners and edges it is "not graded";
    with them (surface still missing) it is a ceiling with a number."""
    v, res = clean
    bare = predict_overall_grade(Measured(51.6, 1.2))
    assert bare.grade_label == "PSA not graded" and not bare.graded
    assert "says nothing about wear" in bare.describe()
    seen = predict_overall_grade(res.worst_ratio, condition=v)
    assert seen.graded and not seen.complete
    assert seen.grade_label.endswith("max") and seen.grade_label != "PSA not graded"
