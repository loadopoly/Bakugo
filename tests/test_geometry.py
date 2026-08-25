"""Geometry tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

import cv2

from cardcenter.geometry import (
    apply_h,
    card_plane_corners_mm,
    enforce_portrait,
    find_card_quad,
    fit_line_robust,
    fit_line_tls,
    intersect_lines,
    order_quad,
    refine_quad,
    snap_quad_to_edges,
)
from cardcenter.types import STANDARD_CARD_H_MM, STANDARD_CARD_W_MM, DetectionError


def _rotate(pts: np.ndarray, deg: float, centre=(0.0, 0.0)) -> np.ndarray:
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c = np.asarray(centre)
    return (pts - c) @ R.T + c


def test_order_quad_is_rotation_stable() -> None:
    """The x+y / x-y heuristic breaks past ~30 degrees; angular sort must not."""
    base = np.array([[10.0, 10.0], [90.0, 10.0], [90.0, 150.0], [10.0, 150.0]])
    centre = base.mean(axis=0)
    for deg in (0, 5, 20, 44, -30, -15):
        rotated = _rotate(base, deg, centre)
        for roll in range(4):
            shuffled = np.roll(rotated, roll, axis=0)
            ordered = order_quad(shuffled)
            # Whatever the input order, the output must be the same cycle.
            ref = order_quad(rotated)
            assert np.allclose(ordered, ref, atol=1e-9), f"unstable at {deg} deg"
        # Clockwise: positive shoelace in y-down image coordinates.
        area = sum(
            ordered[i][0] * ordered[(i + 1) % 4][1]
            - ordered[(i + 1) % 4][0] * ordered[i][1]
            for i in range(4)
        )
        assert area > 0


def test_enforce_portrait() -> None:
    landscape = order_quad(
        np.array([[0.0, 0.0], [150.0, 0.0], [150.0, 90.0], [0.0, 90.0]])
    )
    fixed = enforce_portrait(landscape)
    short = np.linalg.norm(fixed[1] - fixed[0])
    long = np.linalg.norm(fixed[2] - fixed[1])
    assert short < long


def test_fit_line_tls_recovers_known_line() -> None:
    xs = np.linspace(0, 100, 200)
    ys = 0.5 * xs + 7.0
    line, resid = fit_line_tls(np.column_stack([xs, ys]))
    assert resid < 1e-9
    a, b, c = line
    for x, y in zip(xs[::20], ys[::20]):
        assert abs(a * x + b * y + c) < 1e-8


def test_fit_line_tls_handles_vertical() -> None:
    """A least-squares y=mx+b fit would blow up here; TLS must not."""
    ys = np.linspace(0, 100, 100)
    xs = np.full_like(ys, 42.0)
    line, resid = fit_line_tls(np.column_stack([xs, ys]))
    assert resid < 1e-9
    a, b, c = line
    assert abs(a * 42.0 + b * 50.0 + c) < 1e-8


def test_fit_line_residual_reports_noise() -> None:
    rng = np.random.default_rng(0)
    xs = np.linspace(0, 100, 500)
    ys = 2.0 * xs + rng.normal(0, 0.5, xs.size)
    _, resid = fit_line_tls(np.column_stack([xs, ys]))
    perpendicular_sigma = 0.5 / math.sqrt(1 + 2.0**2)
    assert resid == pytest.approx(perpendicular_sigma, rel=0.25)


def test_intersect_lines() -> None:
    l1, _ = fit_line_tls(np.array([[0.0, 0.0], [10.0, 0.0]]))
    l2, _ = fit_line_tls(np.array([[5.0, -5.0], [5.0, 5.0]]))
    assert np.allclose(intersect_lines(l1, l2), [5.0, 0.0], atol=1e-9)


def test_intersect_parallel_raises() -> None:
    l1, _ = fit_line_tls(np.array([[0.0, 0.0], [10.0, 0.0]]))
    l2, _ = fit_line_tls(np.array([[0.0, 5.0], [10.0, 5.0]]))
    with pytest.raises(DetectionError):
        intersect_lines(l1, l2)


def test_refine_quad_beats_its_input() -> None:
    """Subpixel refinement must improve on quantised corner estimates."""
    truth = np.array([[20.3, 30.7], [220.9, 30.1], [221.4, 310.6], [19.8, 311.2]])
    truth = order_quad(truth)

    pts = []
    for i in range(4):
        a, b = truth[i], truth[(i + 1) % 4]
        ts = np.linspace(0, 1, 300)
        pts.append(a[None, :] + ts[:, None] * (b - a)[None, :])
    contour = np.vstack(pts)

    quantised = np.round(truth)  # pixel-quantised, as a contour vertex would be
    refined, residual = refine_quad(contour, quantised)
    refined = order_quad(refined)

    err_before = np.abs(quantised - truth).max()
    err_after = np.abs(refined - truth).max()
    assert err_after < err_before
    assert err_after < 1e-6
    assert residual < 1e-6


def test_refine_quad_rejects_sparse_side() -> None:
    contour = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    quad = order_quad(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(DetectionError):
        refine_quad(contour, quad)


# ---------------------------------------------------------------------------
# Robust line fitting: a card side is a MIXTURE of surfaces, not noisy samples
# ---------------------------------------------------------------------------


def test_fit_line_robust_ignores_a_parallel_second_edge() -> None:
    """The penny-sleeve case: a second straight edge alongside the card's own.

    Least squares splits the difference between the two and lands between them,
    which is where no edge is. Rejection must keep the dominant one.
    """
    x = np.linspace(0, 200, 120)
    card = np.column_stack([x, np.full_like(x, 50.0)])
    sleeve = np.column_stack([x[:40], np.full(40, 62.0)])  # 12 px out, parallel
    pts = np.vstack([card, sleeve])

    _, plain_rms = fit_line_tls(pts)
    line, rms, rejected = fit_line_robust(pts)

    offset = float(np.median(pts[:120] @ line[:2] + line[2]))
    assert abs(offset) < 0.5, "robust fit did not land on the card edge"
    assert rms < plain_rms
    assert 0.0 < rejected < 0.5


def test_fit_line_robust_matches_tls_on_clean_points() -> None:
    """With no contamination it must not throw data away or move the line."""
    x = np.linspace(0, 100, 60)
    pts = np.column_stack([x, 3.0 + 0.0 * x])
    line, rms, rejected = fit_line_robust(pts)
    assert rejected == pytest.approx(0.0)
    assert rms < 1e-9


# ---------------------------------------------------------------------------
# Edge snapping: correcting a halo boundary
# ---------------------------------------------------------------------------


def test_snap_quad_pulls_a_halo_onto_the_real_edge() -> None:
    """A quad sitting outside the card must be pulled back onto its cut edge."""
    img = np.full((700, 560, 3), 30, dtype=np.uint8)
    x0, y0, x1, y1 = 130, 150, 430, 570  # the true card
    cv2.rectangle(img, (x0, y0), (x1, y1), (225, 225, 225), -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )

    halo = order_quad(
        np.array(
            [[x0 - 22.0, y0 - 20.0], [x1 + 21.0, y0 - 19.0],
             [x1 + 20.0, y1 + 22.0], [x0 - 19.0, y1 + 21.0]]
        )
    )
    snapped, _ = snap_quad_to_edges(grad, halo)
    snapped = order_quad(snapped)

    truth = order_quad(
        np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    )
    assert np.abs(snapped - truth).max() < 2.0
    assert np.abs(snapped - truth).max() < np.abs(halo - truth).max()


def test_snap_quad_refuses_a_degenerate_input() -> None:
    grad = np.zeros((80, 80), dtype=np.float32)
    tiny = order_quad(np.array([[1.0, 1.0], [6.0, 1.0], [6.0, 8.0], [1.0, 8.0]]))
    with pytest.raises(DetectionError):
        snap_quad_to_edges(grad, tiny)


def test_apply_h_identity_and_scaling() -> None:
    pts = np.array([[1.0, 2.0], [30.0, 40.0]])
    assert np.allclose(apply_h(np.eye(3), pts), pts)
    S = np.diag([2.0, 3.0, 1.0])
    assert np.allclose(apply_h(S, pts), pts * np.array([2.0, 3.0]))


def test_card_plane_corners_are_portrait_and_standard() -> None:
    c = card_plane_corners_mm()
    assert c[1][0] - c[0][0] == pytest.approx(STANDARD_CARD_W_MM)
    assert c[3][1] - c[0][1] == pytest.approx(STANDARD_CARD_H_MM)
    assert STANDARD_CARD_W_MM < STANDARD_CARD_H_MM


# ---------------------------------------------------------------------------
# find_card_quad: the multi-card container guard
# ---------------------------------------------------------------------------


def _draw_tray() -> tuple[np.ndarray, list[tuple[int, int]]]:
    """A 3x3 display-case tray: a strong outer rim around faint-gapped cells.

    Built so the rim itself is a clean, card-aspect-ish rectangle that would
    otherwise win find_card_quad's normal largest-survivor selection -- the
    scenario that silently measured a whole display case as "one card".
    Returns (image, cell_centers).
    """
    img = np.full((1300, 1000, 3), 180, dtype=np.uint8)
    tray_x0, tray_y0, tray_x1, tray_y1 = 40, 40, 960, 1260
    cv2.rectangle(img, (tray_x0, tray_y0), (tray_x1, tray_y1), (30, 30, 30), -1)
    # Generous margin so no cell's own contour clips against the tray rim --
    # that would make refine_quad legitimately fail on it (too few contour
    # points on the clipped side) for reasons unrelated to what this test is
    # checking, the way a real photographed edge-of-tray cell might too.
    inner_x0, inner_y0 = tray_x0 + 40, tray_y0 + 40
    inner_x1, inner_y1 = tray_x1 - 40, tray_y1 - 40
    cv2.rectangle(img, (inner_x0, inner_y0), (inner_x1, inner_y1), (150, 150, 150), -1)

    cell_w, cell_h, gap = 258, 360, 20
    colors = [
        (60, 60, 220), (60, 180, 60), (220, 60, 60),
        (60, 200, 200), (180, 60, 180), (200, 200, 60),
        (110, 110, 240), (240, 110, 110), (110, 240, 110),
    ]
    centers = []
    for r in range(3):
        for c in range(3):
            cx = inner_x0 + c * (cell_w + gap) + cell_w // 2 + 5
            cy = inner_y0 + r * (cell_h + gap) + cell_h // 2 + 5
            centers.append((cx, cy))
            x_a, y_a = cx - cell_w // 2, cy - cell_h // 2
            x_b, y_b = cx + cell_w // 2, cy + cell_h // 2
            color = colors[r * 3 + c]
            cv2.rectangle(img, (x_a, y_a), (x_b, y_b), color, -1)
            cv2.rectangle(
                img, (x_a + 12, y_a + 12), (x_b - 12, y_b - 12),
                (color[0] - 25, color[1] - 25, color[2] - 25), 4,
            )
    return img, centers


def test_find_card_quad_refuses_a_packed_tray_without_a_reticle() -> None:
    """A display case of 9 cards must not be silently measured as one card."""
    img, _ = _draw_tray()
    with pytest.raises(DetectionError, match="card-shaped regions nested"):
        find_card_quad(img)


def test_find_card_quad_drills_into_the_cell_under_the_reticle() -> None:
    """With prefer_point, the tool measures the cell aimed at, not the tray."""
    img, centers = _draw_tray()
    for target in (centers[4], centers[0], centers[8]):
        quad, _, _ = find_card_quad(img, prefer_point=target)
        centroid = quad.mean(axis=0)
        assert np.linalg.norm(centroid - np.asarray(target, dtype=np.float64)) < 20.0
        # Must be one cell, not the whole tray.
        e01 = np.linalg.norm(quad[1] - quad[0])
        e12 = np.linalg.norm(quad[2] - quad[1])
        assert max(e01, e12) < 500.0
