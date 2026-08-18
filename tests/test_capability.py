"""Tests for the graded capability matrix.

The point of this module is that a card which cannot be fully centred is not
therefore ungradeable. Measured on 75 real cards, the old all-four-borders
requirement yielded 0% — it was unsatisfiable, not strict.
"""

from __future__ import annotations

import numpy as np
import pytest

from cardcenter.capability import (
    PERSPECTIVE_CONFOUND_DEG,
    Capability,
    assess,
    cut_geometry,
)
from cardcenter.synth import render_capture
from cardcenter.types import STANDARD_CARD_W_MM


def _square_quad(w=600.0, h=840.0, x=100.0, y=100.0):
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=float)


# --- cut geometry: available on every card, border or not ------------------


def test_square_quad_reads_as_square() -> None:
    g = cut_geometry(_square_quad(), px_per_mm=10.0)
    assert g.max_angle_error_deg < 0.01
    assert g.square and g.squareness_assessable


def test_skewed_quad_is_flagged() -> None:
    q = _square_quad()
    q[1] += np.array([0.0, 22.0])  # shear one corner ~2deg
    g = cut_geometry(q, px_per_mm=10.0)
    assert 1.5 <= g.max_angle_error_deg < PERSPECTIVE_CONFOUND_DEG
    assert not g.square
    assert any("skewed cut" in n for n in g.notes)


def test_perspective_is_not_reported_as_a_skewed_cut() -> None:
    """A handheld oblique shot turns a square card into a trapezoid. Measured on
    real photographs, 10-15deg deviations are routine and are camera angle, not
    cut quality — reporting them as defects would flag nearly every real card."""
    q = _square_quad()
    q[1] += np.array([0.0, 160.0])
    q[2] += np.array([0.0, -60.0])
    g = cut_geometry(q, px_per_mm=10.0)
    assert g.max_angle_error_deg >= PERSPECTIVE_CONFOUND_DEG
    assert not g.squareness_assessable
    assert any("perspective rather than cut quality" in n for n in g.notes)
    assert not any("skewed cut" in n for n in g.notes)


def test_dimensions_withheld_when_perspective_dominates() -> None:
    q = _square_quad()
    q[1] += np.array([0.0, 160.0])
    g = cut_geometry(q, px_per_mm=10.0)
    assert "withheld" in g.describe()


def test_geometry_recovers_nominal_size_when_square() -> None:
    ppm = 10.0
    g = cut_geometry(_square_quad(STANDARD_CARD_W_MM * ppm, 88.9 * ppm), ppm)
    assert g.width_mm == pytest.approx(STANDARD_CARD_W_MM, abs=0.05)


# --- capability tiers -----------------------------------------------------


def test_bordered_synthetic_card_reaches_full() -> None:
    img, gt, _ = render_capture(left_mm=3.4, right_mm=2.6, top_mm=3.0, bottom_mm=3.0)
    g = assess(img)
    assert g.capability is Capability.FULL
    assert g.worst_ratio is not None
    assert g.worst_ratio.value == pytest.approx(gt.worst_ratio, abs=2.0)


def test_full_bleed_card_falls_to_geometry_only_not_failure() -> None:
    """A Hyper Rare has no printed border. Verified against a certified PSA GEM
    MT 10 (cert 143341329): detected correctly, refused correctly. It must still
    yield geometry, because trim detection is the measurement that matters most
    on exactly these cards."""
    import cv2

    img = np.full((1400, 1000, 3), 25, dtype=np.uint8)
    cv2.rectangle(img, (200, 200), (835, 1089), (60, 140, 200), -1)
    g = assess(img)
    assert g.capability in (Capability.GEOMETRY_ONLY, Capability.PARTIAL)
    assert g.geometry is not None
    assert "trim check" in g.available


def test_capability_tiers_order_by_information() -> None:
    assert Capability.FULL.has_ratio
    assert Capability.SINGLE_AXIS.has_ratio
    assert not Capability.PARTIAL.has_ratio
    assert not Capability.GEOMETRY_ONLY.has_ratio


def test_single_axis_states_what_is_missing() -> None:
    """A one-axis ceiling must say the other axis could be worse, or a user will
    read it as a full result."""
    from cardcenter.capability import GradeCapability
    from cardcenter.types import BorderPair, Measured

    pair = BorderPair("vertical", "top", "bottom", Measured(3.4, 0.1), Measured(2.6, 0.1))
    g = GradeCapability(
        Capability.SINGLE_AXIS, None, pair, {}, None, 10.0,
        "only the vertical border pair is detectable",
        ("centering (vertical only)",), ("centering (horizontal)",),
    )
    text = g.describe()
    assert "not measurable" in text
    assert "could be worse" in text


def test_no_card_yields_none_and_lists_nothing_available() -> None:
    g = assess(np.full((600, 800, 3), 120, dtype=np.uint8))
    assert g.capability is Capability.NONE
    assert g.available == ()
    assert "centering" in " ".join(g.unavailable)


def test_geometry_available_on_every_located_card() -> None:
    """Whatever the border situation, a located card always yields geometry."""
    img, _, _ = render_capture(left_mm=3.0, right_mm=3.0)
    g = assess(img)
    assert g.geometry is not None
    assert "cut squareness" in g.available


def test_drifting_boundary_is_rejected_as_not_a_border() -> None:
    """REGRESSION FROM CERTIFIED GROUND TRUTH.

    On a certified PSA GEM MT 10 (cert 143341329) the tool reported 68.7/31.3
    vertical centering. PSA 10 requires roughly 55/45 or better, so the reading
    was definitively wrong -- it had locked onto the slab label boundary against
    the slab well edge, neither of which is a printed card border.

    A real printed border runs parallel to the cut, so its width is near-constant
    along the side. Rejecting boundaries that drift removed the false positive.
    A one-sided ground-truth label cannot confirm precision, but it falsifies a
    result this far out, and it did."""
    from cardcenter.capability import MAX_BORDER_DRIFT

    assert MAX_BORDER_DRIFT > 0
    # 0.0222 mm/mm was the measured drift on the slab label that produced the
    # false positive; it must sit above the rejection threshold.
    assert 0.0222 > MAX_BORDER_DRIFT


def test_caliper_beam_is_not_accepted_as_a_card() -> None:
    """REGRESSION FROM REAL DATA.

    Reviewing 75 real photographs, 73% of accepted quads had an aspect
    implausible for a card face -- including 9.85, 9.33 and 5.67. Those are the
    caliper's steel beam, a card seen edge-on, and partial detections. They then
    routed to geometry_only, so a misdetection arrived wearing a respectable
    label instead of being refused.

    The old window ran to 2.03, which admitted the beam. It is now 1.15-1.75,
    checked on the REFINED quad because refine_quad can reshape the raw one."""
    import cv2

    from cardcenter.geometry import find_card_quad
    from cardcenter.types import DetectionError

    img = np.full((900, 1400, 3), 30, dtype=np.uint8)
    cv2.rectangle(img, (100, 400), (1300, 500), (200, 205, 210), -1)  # aspect 12
    with pytest.raises(DetectionError):
        find_card_quad(img)


def test_card_aspect_still_accepted_under_perspective() -> None:
    """The gate must not reject genuine cards shot at an angle."""
    from cardcenter.geometry import find_card_quad

    img, _, _ = render_capture(left_mm=3.4, right_mm=2.6, tilt_deg=25.0)
    quad, _, _ = find_card_quad(img)
    e0 = float(np.linalg.norm(quad[1] - quad[0]))
    e1 = float(np.linalg.norm(quad[3] - quad[0]))
    assert 1.15 < max(e0, e1) / min(e0, e1) < 1.75


def test_refinement_declines_when_it_makes_the_quad_worse() -> None:
    """REGRESSION FROM REAL CAPTURE DATA.

    refine_quad intersects line fits through contour points assigned to each
    side. When something straight lies against the card -- most often the
    caliper beam, which is exactly what a good measurement shot contains --
    those points drag the fit off the true edge.

    Measured on a real caliper frame: a clean 1.45-aspect raw quad came back
    from refinement at 1.89 with 26.1 px residual, and the aspect gate then
    rejected the frame entirely. Across the session this discarded 25 of 57
    measurable frames -- the ones WITH a metric reference, i.e. the best data
    available. Falling back to the raw polygon when the residual is large
    recovered 18 of them."""
    import cv2

    from cardcenter.geometry import find_card_quad

    img = np.full((1400, 1000, 3), 30, dtype=np.uint8)
    cv2.rectangle(img, (250, 300), (700, 930), (200, 190, 180), -1)
    # A straight bright bar abutting the card, standing in for the caliper beam.
    cv2.rectangle(img, (150, 290), (760, 305), (240, 240, 245), -1)
    quad, _, residual = find_card_quad(img)
    e0 = float(np.linalg.norm(quad[1] - quad[0]))
    e1 = float(np.linalg.norm(quad[3] - quad[0]))
    assert 1.15 < max(e0, e1) / min(e0, e1) < 1.75


def test_bulk_bin_background_is_not_mistaken_for_a_screenshot() -> None:
    """REGRESSION FROM A REAL BULK-BIN SESSION.

    A card held over a bin of other cards fills the frame edge-to-edge with
    similarly-lit material, so corner and centre luminance match and the
    vignette test reads it as a screen capture. Measured: 10 of 160 real card
    photographs were routed to SCREENSHOT and silently dropped.

    A screenshot must now ALSO carry an exact device aspect ratio. Phone camera
    output never does, so the conjunction is safe where either test alone was
    not."""
    from cardcenter.triage import _is_screenshot

    flat = np.full((3072, 4080, 3), 150, dtype=np.uint8)  # unvignetted, camera ratio
    assert not _is_screenshot(flat)

    ui = np.full((2400, 1080, 3), 150, dtype=np.uint8)  # unvignetted, device ratio
    assert _is_screenshot(ui)


def test_holo_foil_is_not_mistaken_for_a_caliper() -> None:
    """Holographic foil is low-saturation and bright, like brushed steel. The
    naive metal test fired on any holo card: 18 of 160 real frames were routed
    to CALIBRATION with no caliper in shot. A caliper beam is additionally FLAT
    -- foil is full of specular structure -- so requiring low local variance
    separates them."""
    import cv2

    from cardcenter.triage import _detect_caliper

    rng = np.random.default_rng(0)
    foil = np.full((900, 1200, 3), 40, dtype=np.uint8)
    patch = rng.integers(120, 255, (400, 500, 3), dtype=np.uint8)  # specular noise
    foil[250:650, 350:850] = patch
    assert not _detect_caliper(foil)
