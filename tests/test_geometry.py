"""Geometry tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cardcenter.geometry import (
    apply_h,
    card_plane_corners_mm,
    enforce_portrait,
    fit_line_tls,
    intersect_lines,
    order_quad,
    refine_quad,
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
