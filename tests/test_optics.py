"""Optics tests.

The important one is test_matches_fermat_raytrace. Everything else in this
package is checked against a synthetic renderer that shares code with the
solver, so it can only catch implementation bugs, not physics errors. This test
derives the displacement a completely different way -- by numerically minimising
optical path length, i.e. Fermat's principle -- and compares. If equation (1) in
optics.py were wrong, this is the test that would catch it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from cardcenter.optics import (
    inplane_shift_measured,
    inplane_shift_mm,
    pose_from_homography,
    refraction_angle,
)
from cardcenter.types import SLAB_PRESETS, SlabSpec


def _fermat_apparent_x(
    card_x: float, t: float, n: float, cam_x: float, cam_h: float
) -> tuple[float, float]:
    """Ray trace by Fermat's principle. Returns (apparent_x, exit_angle_rad).

    Geometry in a 2D slice: the slab's outer surface is z=0, the card point is
    at (card_x, -t) inside a medium of index n, and the camera is at
    (cam_x, cam_h) in air. The physical light path minimises n*|E-P| + |C-E|
    over the exit point E on the surface. The camera then back-projects the
    straight line C->E down to z=-t, which is where the point appears to be.
    """

    def optical_path(x_e: float) -> float:
        in_medium = math.hypot(x_e - card_x, t)
        in_air = math.hypot(cam_x - x_e, cam_h)
        return n * in_medium + in_air

    lo = min(card_x, cam_x) - 10.0
    hi = max(card_x, cam_x) + 10.0
    res = minimize_scalar(optical_path, bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-12})
    x_e = float(res.x)

    # Back-project the straight line from the camera through the exit point.
    dx = x_e - cam_x
    dz = 0.0 - cam_h
    s = (-t - cam_h) / dz
    apparent_x = cam_x + s * dx
    exit_angle = math.atan2(abs(dx), abs(dz))
    return apparent_x, exit_angle


@pytest.mark.parametrize("tilt_deg", [5.0, 15.0, 25.0, 35.0, 45.0])
@pytest.mark.parametrize("slab_name", ["psa", "bgs", "toploader"])
def test_matches_fermat_raytrace(tilt_deg: float, slab_name: str) -> None:
    slab = SLAB_PRESETS[slab_name]
    t = slab.acrylic_thickness_mm
    n = slab.refractive_index

    cam_h = 220.0
    card_x = 0.0
    cam_x = cam_h * math.tan(math.radians(tilt_deg))

    apparent_x, exit_angle = _fermat_apparent_x(card_x, t, n, cam_x, cam_h)
    traced_shift = card_x - apparent_x  # positive: appears displaced away from camera

    # Equation (1) evaluated at the true exit ray angle.
    formula_exact = float(inplane_shift_mm(exit_angle, slab))
    assert formula_exact == pytest.approx(traced_shift, abs=1e-6), (
        "equation (1) disagrees with the Fermat ray trace"
    )

    # Equation (1) as the pipeline actually uses it: with theta taken from the
    # straight line between the card point and the camera, which is what pose
    # recovery gives us. This is an approximation because the true exit ray
    # leaves from a point offset by t*tan(theta2). Quantify the resulting error.
    straight_angle = math.atan2(abs(cam_x - card_x), cam_h)
    formula_used = float(inplane_shift_mm(straight_angle, slab))
    approximation_error = abs(formula_used - traced_shift)
    assert approximation_error < 0.006, (
        f"straight-line-angle approximation costs {approximation_error * 1000:.1f} um, "
        "which is larger than budgeted"
    )


def test_snell_holds() -> None:
    for deg in (0.0, 10.0, 30.0, 60.0, 89.0):
        th1 = math.radians(deg)
        th2 = float(refraction_angle(th1, 1.49))
        assert math.sin(th1) == pytest.approx(1.49 * math.sin(th2), abs=1e-12)
        assert th2 <= th1 + 1e-12  # bends toward the normal entering a denser medium


def test_no_shift_without_optical_medium() -> None:
    assert float(inplane_shift_mm(math.radians(30), SLAB_PRESETS["raw"])) == 0.0
    flat = SlabSpec(name="x", acrylic_thickness_mm=2.0, refractive_index=1.0)
    assert float(inplane_shift_mm(math.radians(30), flat)) == 0.0
    zero_t = SlabSpec(name="y", acrylic_thickness_mm=0.0, refractive_index=1.6)
    assert float(inplane_shift_mm(math.radians(30), zero_t)) == 0.0


def test_shift_is_zero_at_normal_incidence_and_monotonic() -> None:
    slab = SLAB_PRESETS["bgs"]
    assert float(inplane_shift_mm(0.0, slab)) == pytest.approx(0.0, abs=1e-12)
    angles = np.radians(np.arange(0, 70, 2.5))
    shifts = np.asarray(inplane_shift_mm(angles, slab))
    assert np.all(np.diff(shifts) > 0), "shift must increase with incidence angle"


def test_perpendicular_formula_relationship() -> None:
    """The textbook parallel-plate offset relates to ours by 1/cos(theta1).

    Guards against anyone 'fixing' optics.py by swapping in the more commonly
    cited formula, which measures a different quantity.
    """
    slab = SLAB_PRESETS["bgs"]
    for deg in (10.0, 25.0, 40.0):
        th1 = math.radians(deg)
        th2 = float(refraction_angle(th1, slab.refractive_index))
        perpendicular = slab.acrylic_thickness_mm * math.sin(th1 - th2) / math.cos(th2)
        inplane = float(inplane_shift_mm(th1, slab))
        assert inplane == pytest.approx(perpendicular / math.cos(th1), rel=1e-10)
        assert inplane > perpendicular  # the two are not interchangeable


def test_uncertainty_matches_numeric_jacobian() -> None:
    """Analytic partials in inplane_shift_measured must match finite differences."""
    slab = SlabSpec(
        name="test",
        acrylic_thickness_mm=1.6,
        acrylic_thickness_sigma_mm=0.4,
        refractive_index=1.49,
        refractive_index_sigma=0.01,
    )
    th1 = math.radians(25.0)
    th_sigma = math.radians(2.0)
    got = inplane_shift_measured(th1, slab, th_sigma)

    eps = 1e-7

    def f(t: float, n: float, th: float) -> float:
        s = SlabSpec(name="t", acrylic_thickness_mm=t, refractive_index=n)
        return float(inplane_shift_mm(th, s))

    base = f(slab.acrylic_thickness_mm, slab.refractive_index, th1)
    d_dt = (f(slab.acrylic_thickness_mm + eps, slab.refractive_index, th1) - base) / eps
    d_dn = (f(slab.acrylic_thickness_mm, slab.refractive_index + eps, th1) - base) / eps
    d_dth = (f(slab.acrylic_thickness_mm, slab.refractive_index, th1 + eps) - base) / eps

    numeric_sigma = math.sqrt(
        (d_dt * slab.acrylic_thickness_sigma_mm) ** 2
        + (d_dn * slab.refractive_index_sigma) ** 2
        + (d_dth * th_sigma) ** 2
    )
    assert got.value == pytest.approx(base, rel=1e-9)
    assert got.sigma == pytest.approx(numeric_sigma, rel=1e-4)


def test_pose_recovery_round_trip() -> None:
    """A synthesised pose must be recoverable from its own homography."""
    from cardcenter.synth import _camera_matrices

    for tilt in (0.0, 12.0, 28.0, 40.0):
        K, R, C = _camera_matrices(tilt, 37.0, 220.0, 2400.0, (1400, 1800))
        tvec = -R @ C
        H = K @ np.column_stack([R[:, 0], R[:, 1], tvec])
        pose = pose_from_homography(H, K)
        assert pose is not None, f"decomposition failed at tilt {tilt}"
        assert np.allclose(pose.center_mm, C, atol=1e-6), "camera centre wrong"
        assert pose.tilt_deg == pytest.approx(tilt, abs=1e-4)


def test_incidence_angle_sign_agnostic() -> None:
    """Cameras on either side of the plane must give the same acute angle."""
    from cardcenter.optics import CameraPose

    pts = np.array([[10.0, 20.0], [50.0, 70.0]])
    above = CameraPose(np.array([30.0, 45.0, 200.0]), np.eye(3), 0.0)
    below = CameraPose(np.array([30.0, 45.0, -200.0]), np.eye(3), 0.0)
    assert np.allclose(above.incidence_angles(pts), below.incidence_angles(pts))
