"""Refraction through the slab face.

DERIVATION
----------
Set up card-plane coordinates: the slab's outer surface is the plane z=0 with
normal +z, the camera is somewhere at z>0, and the card sits a distance t of
solid plastic below the surface. (Air gaps and inner sleeves between the card
and the acrylic contribute *zero* lateral displacement, because n=1 makes the
two tangent terms below cancel. Only the solid plastic path counts.)

A point P on the card at (0, -t) emits a ray that travels through the plastic
at angle theta2 from the normal, exits at the surface point E = (t*tan(theta2), 0),
and refracts to angle theta1 in air, with

    sin(theta1) = n * sin(theta2)          [Snell]

The camera has no way to know refraction happened; it back-projects the exit
ray in a straight line. At depth z = -t that back-projection lands at

    x_apparent = t*tan(theta2) - t*tan(theta1)

so the apparent position is displaced from the true position by

    delta = t * (tan(theta1) - tan(theta2))                            (1)

directed *away* from the camera's in-plane direction. Equivalently the true
position is recovered by moving the apparent position *toward* the camera:

    P_true = P_apparent + delta * u_hat,   u_hat = in-plane unit vector to camera

NOTE ON A COMMON WRONG FORMULA
------------------------------
The textbook "lateral displacement through a parallel plate" is

    d = t * sin(theta1 - theta2) / cos(theta2)                         (2)

That is a real formula but it measures a *different thing*: the perpendicular
offset between the incoming and outgoing ray lines, measured perpendicular to
the ray. What we need for card measurement is the in-plane displacement on the
card surface. They are related by delta = d / cos(theta1), so using (2)
directly under-reports the correction by a factor of cos(theta1) -- 9% at 25
degrees, 23% at 40 degrees. This module uses (1).

MAGNITUDE, HONESTLY
-------------------
BGS-style acrylic, t=1.6mm of PMMA (n=1.49), viewed at 25 degrees off-normal:
delta = 0.27mm. Against a ~3mm border that is a ~9% error, which is roughly
one grading step at the tight end of the scale. Real but not enormous. If you
plug in a 3.25mm *total slab depth* instead of the ~1.6mm acrylic face you get
0.55mm and overcorrect by 2x. Measure your slabs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import Measured, SlabSpec


def refraction_angle(theta1_rad: float | np.ndarray, n: float):
    """Snell's law, air -> medium. Returns theta2 (inside the plastic)."""
    s = np.sin(theta1_rad) / n
    s = np.clip(s, -1.0, 1.0)  # cannot TIR going into a denser medium
    return np.arcsin(s)


def inplane_shift_mm(theta1_rad: float | np.ndarray, slab):
    """Equation (1): apparent in-plane displacement magnitude, in mm.

    Accepts a SlabSpec or a SlabStack. For a stack the layer contributions add:
    the ray re-emerges parallel after each plane-parallel layer, so each layer
    sees the same external angle theta1 and displaces independently. Air gaps
    between layers contribute nothing.

    Always >= 0. Direction is handled by the caller.
    """
    if not slab.is_optically_active:
        return np.zeros_like(np.asarray(theta1_rad, dtype=float))
    total = np.zeros_like(np.asarray(theta1_rad, dtype=float))
    for layer in slab.layers:
        if not layer.is_optically_active:
            continue
        th2 = refraction_angle(theta1_rad, layer.refractive_index)
        total = total + layer.acrylic_thickness_mm * (np.tan(theta1_rad) - np.tan(th2))
    return total


def inplane_shift_measured(
    theta1_rad: float, slab, theta1_sigma_rad: float = 0.0
) -> Measured:
    """Equation (1) with first-order propagation of t, n and theta1 uncertainty.

    Partial derivatives (s = sin(theta1)):
        d(delta)/dt      = tan(th1) - tan(th2)
        d(delta)/dn      =  t * s / (n^2 * cos^3(th2))
        d(delta)/dth1    =  t * (sec^2(th1) - cos(th1) / (n * cos^3(th2)))
    """
    if not slab.is_optically_active:
        return Measured(0.0, 0.0)

    th1 = float(theta1_rad)
    value = 0.0
    # Thickness and index errors are independent per layer, so their variances
    # add. The shared angle error is correlated across layers, so its
    # derivatives must be summed BEFORE squaring, not squared and summed.
    var_independent = 0.0
    d_dth1_total = 0.0

    for layer in slab.layers:
        if not layer.is_optically_active:
            continue
        t = layer.acrylic_thickness_mm
        n = layer.refractive_index
        th2 = float(refraction_angle(th1, n))

        value += t * (math.tan(th1) - math.tan(th2))

        d_dt = math.tan(th1) - math.tan(th2)
        d_dn = t * math.sin(th1) / (n**2 * math.cos(th2) ** 3)
        d_dth1_total += t * (
            1.0 / math.cos(th1) ** 2 - math.cos(th1) / (n * math.cos(th2) ** 3)
        )

        var_independent += (
            (d_dt * layer.acrylic_thickness_sigma_mm) ** 2
            + (d_dn * layer.refractive_index_sigma) ** 2
        )

    var = var_independent + (d_dth1_total * theta1_sigma_rad) ** 2
    return Measured(value, math.sqrt(var))


@dataclass(frozen=True)
class CameraPose:
    """Camera position expressed in card-plane coordinates (mm), z up."""

    center_mm: np.ndarray  # (3,)
    R: np.ndarray  # (3,3) card -> camera rotation
    tilt_deg: float  # angle between optical axis and plane normal

    def incidence_angles(self, pts_mm: np.ndarray) -> np.ndarray:
        """Per-point angle from the plane normal, radians. pts_mm is (N,2).

        The card plane's normal is +/-z; which sign faces the camera depends on
        the handedness of the card coordinate frame (ours is x-right, y-down to
        match image convention, which puts the camera at negative z). The
        physically meaningful quantity is the acute angle between the viewing
        ray and the plane normal, so take the absolute z component.
        """
        pts3 = np.column_stack([pts_mm, np.zeros(len(pts_mm))])
        v = self.center_mm[None, :] - pts3
        norm = np.linalg.norm(v, axis=1)
        norm = np.where(norm < 1e-9, 1e-9, norm)
        cos_t = np.clip(np.abs(v[:, 2]) / norm, 0.0, 1.0)
        return np.arccos(cos_t)

    def inplane_directions(self, pts_mm: np.ndarray) -> np.ndarray:
        """Unit vectors in the card plane pointing toward the camera. (N,2)."""
        v = self.center_mm[None, :2] - pts_mm
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        norm = np.where(norm < 1e-9, 1e-9, norm)
        return v / norm


def pose_from_homography(H: np.ndarray, K: np.ndarray) -> Optional[CameraPose]:
    """Recover camera pose from a card-plane -> image homography.

    H maps homogeneous card-plane mm coords to image pixels. With intrinsics K,
    K^-1 H is proportional to [r1 | r2 | tvec]. We rescale, re-orthonormalize R
    via SVD (the raw columns are only approximately orthonormal because H was
    fit from noisy points), and invert to get the camera center in card coords.

    Returns None if the decomposition is degenerate.
    """
    try:
        M = np.linalg.inv(K) @ H
    except np.linalg.LinAlgError:
        return None

    n1 = np.linalg.norm(M[:, 0])
    n2 = np.linalg.norm(M[:, 1])
    if n1 < 1e-12 or n2 < 1e-12:
        return None

    lam = 1.0 / ((n1 + n2) / 2.0)
    M = M * lam

    r1, r2, tvec = M[:, 0], M[:, 1], M[:, 2]
    if tvec[2] < 0:  # card must be in front of the camera
        r1, r2, tvec = -r1, -r2, -tvec

    r3 = np.cross(r1, r2)
    R_approx = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    center = -R.T @ tvec
    # The camera may sit at +z or -z depending on the handedness of the card
    # frame. Only an in-plane camera is degenerate.
    if abs(center[2]) < 1e-6:
        return None

    # Optical axis is the camera's +z expressed in card coords: third row of R.
    axis = R[2, :]
    tilt = math.degrees(math.acos(min(1.0, abs(float(axis[2])))))
    return CameraPose(center_mm=center, R=R, tilt_deg=tilt)


def correct_points_for_refraction(
    pts_mm: np.ndarray, pose: CameraPose, slab
) -> tuple[np.ndarray, np.ndarray]:
    """Move apparent card-plane points to their true positions.

    Returns (corrected_pts, shift_magnitudes_mm).
    """
    if not slab.is_optically_active or len(pts_mm) == 0:
        return pts_mm.copy(), np.zeros(len(pts_mm))

    theta = pose.incidence_angles(pts_mm)
    mag = np.asarray(inplane_shift_mm(theta, slab), dtype=float)
    u = pose.inplane_directions(pts_mm)
    return pts_mm + mag[:, None] * u, mag


def refraction_report(slab, tilt_deg: float) -> str:
    """Human-readable sanity line so the user can see what the physics did."""
    if not slab.is_optically_active:
        return f"slab '{slab.name}': optically inactive, no refraction correction"
    m = inplane_shift_measured(math.radians(tilt_deg), slab, math.radians(2.0))
    return (
        f"slab '{slab.name}': t={slab.acrylic_thickness_mm:.2f}mm n={slab.refractive_index:.3f} "
        f"at {tilt_deg:.1f}deg -> shift {m.value:.3f}±{m.sigma:.3f} mm"
    )
