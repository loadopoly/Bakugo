"""Depth from defocus: checking that the caliper is actually coplanar with the card.

WHY THIS EXISTS
---------------
`ar.calibrate_from_points` has one dominant failure mode and it is not the
caliper's tolerance. Scale is apparent size over true size, and apparent size
goes as 1/distance, so a caliper held 10% nearer than the card makes every
dimension 10% wrong -- 6.3 mm on a card width, a hundred times the quantity
being measured. Until now the module could only warn about this; it had no way
to detect it. `depth_mismatch_frac` was a parameter the caller had to supply
from information it did not have.

Defocus supplies it. A lens focused at distance s images an object at distance
z with a blur circle of diameter

    c = A * f * |z - s| / (z * (s - f))

so measuring blur measures depth. This is standard depth-from-defocus, and it
works here for one reason: the working distance is short. At kilometre ranges
everything sits beyond hyperfocal and the blur derivative vanishes -- which is
why the same idea was useless for the filament work. At 200 mm it is not:

    focus 200mm, object +5mm  ->  2.2 px blur
    focus 200mm, object +10mm ->  4.3 px blur
    focus 300mm, object +10mm ->  1.9 px blur
    focus 500mm, object +10mm ->  0.7 px blur

Hyperfocal for a typical phone (f=5.6mm, f/1.8, 2px circle of confusion) is
about 8.7 m. Inside a metre the technique has real discrimination; past a few
metres it has none. So this check is meaningful only for close work, and it
reports its own resolution rather than pretending otherwise.

WHAT IT DOES NOT DO
-------------------
It does not give absolute depth. Blur is symmetric about the focal plane --
c depends on |z - s| -- so an object 10 mm nearer and one 10 mm further produce
identical blur. The SIGN is unmeasured, exactly as the tangent sign was in the
filament case, and for the same structural reason: the forward map is
two-to-one. What this measures is depth SEPARATION between two objects in one
frame, which is all the coplanarity check needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .types import STANDARD_CARD_W_MM, DetectionError

# Typical phone main camera. Override per device if known.
DEFAULT_FOCAL_MM = 5.6
DEFAULT_APERTURE_F = 1.8
DEFAULT_PIXEL_PITCH_UM = 1.0


@dataclass(frozen=True)
class LensModel:
    focal_mm: float = DEFAULT_FOCAL_MM
    f_number: float = DEFAULT_APERTURE_F
    pixel_pitch_um: float = DEFAULT_PIXEL_PITCH_UM

    @property
    def aperture_mm(self) -> float:
        return self.focal_mm / max(self.f_number, 1e-6)

    def blur_px(self, focus_mm: float, object_mm: float) -> float:
        """Blur circle diameter in image pixels."""
        f, A = self.focal_mm, self.aperture_mm
        if object_mm <= f or focus_mm <= f:
            return float("inf")
        c_mm = A * f * abs(object_mm - focus_mm) / (object_mm * (focus_mm - f))
        return c_mm * 1000.0 / max(self.pixel_pitch_um, 1e-9)

    def depth_offset_mm(self, focus_mm: float, blur_px: float) -> float:
        """Invert the blur relation. Returns |z - s|, magnitude only.

        The sign is not recoverable from blur alone: an object nearer than the
        focal plane and one further produce the same circle.
        """
        f, A = self.focal_mm, self.aperture_mm
        c_mm = blur_px * self.pixel_pitch_um / 1000.0
        denom = A * f - c_mm * (focus_mm - f)
        if abs(denom) < 1e-12:
            return float("inf")
        # Solving c = A f (z-s) / (z (s-f)) for z, near branch:
        z = A * f * focus_mm / denom
        return abs(z - focus_mm)

    @property
    def hyperfocal_mm(self) -> float:
        c_conf = 2.0 * self.pixel_pitch_um / 1000.0
        return self.focal_mm**2 / (self.f_number * c_conf) + self.focal_mm


def edge_blur_px(image: np.ndarray, quad: np.ndarray, samples: int = 60) -> float:
    """Measure blur from the 10-90% rise width across an object's edges.

    A step edge convolved with a blur circle has a rise width proportional to
    the circle diameter. Measuring it along the quad's own boundary means the
    object supplies its own test target -- no separate calibration chart.
    """
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ).astype(np.float64)
    h, w = gray.shape
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    centre = q.mean(axis=0)

    widths: list[float] = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        edge = b - a
        n = np.array([-edge[1], edge[0]], dtype=np.float64)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if float(n @ (0.5 * (a + b) - centre)) < 0:
            n = -n

        for t in np.linspace(0.2, 0.8, samples // 4):
            base = a + t * edge
            offs = np.arange(-8.0, 8.01, 0.5)
            pts = base[None, :] + offs[:, None] * n[None, :]
            xi = np.round(pts[:, 0]).astype(int)
            yi = np.round(pts[:, 1]).astype(int)
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            if ok.sum() < 20:
                continue
            prof = gray[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)].astype(float)
            prof = np.where(ok, prof, np.nan)
            # The profile runs bright-to-dark going outward (object is lighter
            # than background) or dark-to-bright (darker object). Searching for
            # the FIRST sample above a threshold gives index 0 in the
            # bright-to-dark case for both thresholds, yielding rise=0 -- which
            # silently discarded every edge. Normalise the direction first.
            valid = np.isfinite(prof)
            if valid.sum() < 20:
                continue
            p_ok = prof[valid]
            o_ok = offs[valid]
            if p_ok[0] > p_ok[-1]:
                p_ok = p_ok[::-1]
                o_ok = o_ok[::-1]
            lo, hi = float(np.min(p_ok)), float(np.max(p_ok))
            if hi - lo < 12.0:
                continue
            t10, t90 = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
            i10 = np.argmax(p_ok >= t10)
            i90 = np.argmax(p_ok >= t90)
            if i90 > i10:
                rise = abs(float(o_ok[i90]) - float(o_ok[i10]))
                if 0.0 < rise < 16.0:
                    widths.append(rise)

    if len(widths) < 8:
        raise DetectionError(
            "not enough usable edges to measure blur; the object needs a "
            "high-contrast boundary against its background"
        )
    return float(np.median(widths))


@dataclass(frozen=True)
class CoplanarityCheck:
    card_blur_px: float
    reference_blur_px: float
    blur_difference_px: float
    depth_separation_mm: Optional[float]
    scale_error_frac: Optional[float]
    coplanar: bool
    resolution_mm: float
    warnings: tuple[str, ...]

    def describe(self) -> str:
        lines = [
            f"card blur {self.card_blur_px:.2f} px, reference blur "
            f"{self.reference_blur_px:.2f} px "
            f"(difference {self.blur_difference_px:+.2f} px)"
        ]
        if self.depth_separation_mm is not None:
            lines.append(
                f"  implied depth separation ~{self.depth_separation_mm:.1f} mm"
            )
        if self.scale_error_frac is not None:
            lines.append(
                f"  which would put scale off by {self.scale_error_frac * 100:.1f}% "
                f"= {self.scale_error_frac * STANDARD_CARD_W_MM:.2f} mm on a card width"
            )
        lines.append(
            f"  this check resolves depth to about {self.resolution_mm:.1f} mm here"
        )
        lines.append("  COPLANAR" if self.coplanar else "  NOT COPLANAR -- re-shoot")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def check_coplanarity(
    image: np.ndarray,
    card_quad: np.ndarray,
    reference_quad: np.ndarray,
    working_distance_mm: float = 250.0,
    lens: Optional[LensModel] = None,
    tolerance_px: float = 1.0,
) -> CoplanarityCheck:
    """Are the card and the scale reference at the same depth?

    Compares blur on each object's own edges. Equal blur means equal distance
    from the focal plane; unequal blur means one is nearer, which corrupts the
    scale by the same fraction.
    """
    lens = lens or LensModel()
    card_blur = edge_blur_px(image, card_quad)
    ref_blur = edge_blur_px(image, reference_quad)
    diff = ref_blur - card_blur

    warnings: list[str] = []
    if working_distance_mm > 0.35 * lens.hyperfocal_mm:
        warnings.append(
            f"working distance {working_distance_mm:.0f} mm is a large fraction "
            f"of hyperfocal ({lens.hyperfocal_mm:.0f} mm); defocus carries little "
            "depth information here and this check is weak"
        )

    # Smallest depth difference this setup could detect at all.
    probe = 1.0
    while probe < 200.0:
        if lens.blur_px(working_distance_mm, working_distance_mm + probe) >= tolerance_px:
            break
        probe += 0.5
    resolution = probe

    depth_sep: Optional[float] = None
    scale_err: Optional[float] = None
    if abs(diff) > tolerance_px:
        depth_sep = lens.depth_offset_mm(working_distance_mm, abs(diff))
        if math.isfinite(depth_sep):
            scale_err = depth_sep / max(working_distance_mm, 1e-9)

    coplanar = abs(diff) <= tolerance_px
    if not coplanar and scale_err is not None and scale_err > 0.01:
        warnings.append(
            "rest the caliper on the same surface as the card; this depth "
            "mismatch alone exceeds every other error in the measurement"
        )
    if coplanar and resolution > 15.0:
        warnings.append(
            f"reported coplanar, but this setup cannot detect a mismatch smaller "
            f"than {resolution:.0f} mm -- treat as weak evidence, not a clean bill"
        )

    return CoplanarityCheck(
        card_blur_px=card_blur,
        reference_blur_px=ref_blur,
        blur_difference_px=diff,
        depth_separation_mm=depth_sep,
        scale_error_frac=scale_err,
        coplanar=coplanar,
        resolution_mm=resolution,
        warnings=tuple(warnings),
    )
