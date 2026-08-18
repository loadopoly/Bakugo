"""Illumination geometry: what the light does to the measurement.

ON THE ENTANGLEMENT QUESTION
----------------------------
The proposition was that a shadow carries the occluder's structure into the
space beyond it, and that this persistence is a form of entanglement across
"layered space". The first half is right and the second half is not, and the
distinction is worth stating precisely because it is the one that decides
whether any of this is exploitable.

  1. CORRELATION IS NOT ENTANGLEMENT. A shadow is a correlation between the
     occluder's geometry and the intensity pattern downstream. Correlations that
     strong are trivially classical: any shared random variable produces them.
     Formally, a separable state rho = SUM_i p_i (rho_A^i (x) rho_B^i) can carry
     arbitrarily strong correlations, and entanglement is precisely what is left
     over when you subtract everything a shared random variable can do. The
     operational test is a Bell inequality; shadows never violate one.

  2. ABSORPTION IS A LOCAL OPERATION, AND LOCAL OPERATIONS CANNOT CREATE
     ENTANGLEMENT. This is a theorem, not a modelling choice: LOCC cannot
     increase entanglement. An occluder acts locally on the field passing
     through it. So the light past a card has exactly the entanglement the
     light had before it, and ambient thermal light has none.

  3. THERE IS NO "LAYER DISPLACED PHOTON." Photons striking the card are
     absorbed. A shadow is the statistical absence of arrivals, and an absence
     is not a carrier. Nothing propagates the shape; the shape is what is
     missing.

  4. THE HOLOGRAM INTUITION IS REAL BUT CLASSICAL, AND UNAVAILABLE HERE. The
     complex field past an occluder genuinely does encode its shape, and you can
     numerically back-propagate it -- that is holography and phase retrieval,
     and it is Maxwell, not quantum mechanics. It needs the phase, which an
     intensity sensor discards, AND it needs mutual coherence across the feature
     you want to reconstruct. Measured below: ordinary light is coherent over
     about 1-40 um, while a card border is 3000 um. The field across a single
     border is mutually incoherent by three orders of magnitude, so there is no
     extended phase relationship to retrieve even with a perfect phase-sensitive
     detector.

WHAT THE INTUITION WAS ACTUALLY POINTING AT, AND IT MATTERS A LOT
------------------------------------------------------------------
Shadows do carry information about the object, and one of them wrecks this
measurement. A card is not a plane: it is about 0.30 mm thick. Lit obliquely it
casts a shadow of its own edge onto whatever it is sitting on -- and that shadow
falls ONLY on the side away from the light.

Almost every other error in this package is common-mode and cancels in a ratio.
This one does not. It adds to one border and not its opposite:

    light elevation   shadow cast    measured ratio (truth 50/50)   error
        60 deg          0.17 mm              50.8 / 49.2           +0.8 pp
        45 deg          0.30 mm              50.8 / 49.2           +0.8 pp
        30 deg          0.52 mm              82.8 / 17.2          +32.8 pp
        20 deg          0.82 mm              76.7 / 23.3          +26.7 pp

At 30 degrees a perfectly centred card measures as 83/17, and the reported error
bar misses the truth by seven sigma. That is the worst failure mode found in this
project, it is caused entirely by where the lamp is, and it was invisible until
someone asked what shadows do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# Typical modern card stock. Thicker for slabbed cards, but a slab's own edge is
# what casts then, and slabs are several millimetres.
TYPICAL_CARD_THICKNESS_MM = 0.30

# Below this the shadow starts to matter more than everything else combined.
SAFE_ELEVATION_DEG = 55.0


@dataclass(frozen=True)
class CoherenceCheck:
    """Is there recoverable phase structure across a card border?"""

    temporal_um: float
    spatial_um: float
    feature_um: float

    @property
    def ratio(self) -> float:
        return self.feature_um / max(min(self.temporal_um, self.spatial_um), 1e-9)

    @property
    def phase_retrieval_possible(self) -> bool:
        return self.ratio < 1.0

    def describe(self) -> str:
        return (
            f"coherence: temporal {self.temporal_um:.2f} um, spatial "
            f"{self.spatial_um:.1f} um, against a {self.feature_um:.0f} um "
            f"feature -- incoherent by {self.ratio:.0f}x. "
            + (
                "phase retrieval is possible"
                if self.phase_retrieval_possible
                else "no extended phase relationship exists to retrieve"
            )
        )


def coherence_check(
    bandwidth_nm: float = 300.0,
    centre_nm: float = 550.0,
    source_angular_radius_rad: float = 0.0093,
    feature_mm: float = 3.0,
) -> CoherenceCheck:
    """Temporal and spatial coherence lengths of the illumination.

    Temporal from the bandwidth (l = lambda^2 / dlambda); spatial from the
    van Cittert-Zernike theorem (l = 0.61 lambda / source angular radius). The
    default source radius is the sun's, 0.0093 rad.
    """
    lam = centre_nm * 1e-9
    dlam = max(bandwidth_nm * 1e-9, 1e-12)
    temporal = lam**2 / dlam
    spatial = 0.61 * lam / max(source_angular_radius_rad, 1e-9)
    return CoherenceCheck(
        temporal_um=temporal * 1e6,
        spatial_um=spatial * 1e6,
        feature_um=feature_mm * 1000.0,
    )


def edge_shadow_mm(
    thickness_mm: float = TYPICAL_CARD_THICKNESS_MM,
    elevation_deg: float = 45.0,
) -> float:
    """Width of the shadow a card's own edge casts, on the far side from the light."""
    if elevation_deg >= 89.5:
        return 0.0
    if elevation_deg <= 0.5:
        return float("inf")
    return thickness_mm / math.tan(math.radians(elevation_deg))


def ratio_bias_pp(
    shadow_mm: float, border_a_mm: float, border_b_mm: float
) -> float:
    """How much a one-sided shadow shifts the centering ratio, in points."""
    if not math.isfinite(shadow_mm):
        return float("inf")
    a, b = border_a_mm + shadow_mm, border_b_mm
    true_ratio = 100.0 * max(border_a_mm, border_b_mm) / (border_a_mm + border_b_mm)
    seen = 100.0 * max(a, b) / (a + b)
    return seen - true_ratio


def correct_quad_for_shadow(
    quad: np.ndarray, side_index: int, shadow_mm: float, px_per_mm: float
) -> np.ndarray:
    """Pull one edge of the quad inward by the measured shadow width.

    Refusing was the wrong response. In a shop the lighting is whatever the case
    has -- usually LED strips at a low angle behind glass -- and a tool that
    declines below 55 degrees of elevation declines nearly everything, which
    makes it useless exactly where it is meant to be used.

    The shadow is detectable, its width is measurable, and the true card edge is
    the INNER boundary of the dark band rather than its outer one. So the honest
    move is to correct the boundary and carry the correction's own uncertainty,
    not to walk away from the measurement.
    """
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2).copy()
    a_i, b_i = side_index, (side_index + 1) % 4
    a, b = q[a_i], q[b_i]
    centre = q.mean(axis=0)
    edge = b - a
    n = np.array([-edge[1], edge[0]], dtype=np.float64)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return q
    n /= nn
    if float(n @ (0.5 * (a + b) - centre)) < 0:
        n = -n  # outward
    shift = shadow_mm * px_per_mm * n
    q[a_i] = a - shift
    q[b_i] = b - shift
    return q


@dataclass(frozen=True)
class ShadowVerdict:
    asymmetry: float
    darker_side: Optional[str]
    estimated_shadow_mm: float
    estimated_elevation_deg: Optional[float]
    directional: bool
    severe: bool
    warnings: tuple[str, ...]
    side_index: int = -1
    correctable: bool = False

    def describe(self) -> str:
        if not self.directional:
            return "lighting looks even; no edge shadow detected"
        lines = [
            f"DIRECTIONAL LIGHT detected: background on the {self.darker_side} "
            f"is {self.asymmetry:.0%} darker than the opposite side"
        ]
        if self.estimated_elevation_deg is not None:
            lines.append(
                f"  implies a light elevation near {self.estimated_elevation_deg:.0f} "
                f"deg and an edge shadow of about "
                f"{self.estimated_shadow_mm:.2f} mm"
            )
        for w in self.warnings:
            lines.append(f"  {w}")
        return "\n".join(lines)


def detect_edge_shadow(
    image: np.ndarray,
    card_quad: np.ndarray,
    px_per_mm: float,
    thickness_mm: float = TYPICAL_CARD_THICKNESS_MM,
    darkness_threshold: float = 0.6,
) -> ShadowVerdict:
    """Find an edge shadow by looking JUST INSIDE the detected card boundary.

    The obvious test -- compare the background outside opposite edges -- does not
    work, and the reason is instructive. When an edge shadow is present the
    detector has usually already swallowed it: the boundary it found is the
    OUTER edge of the shadow, not the card. Sampling outside that boundary then
    finds ordinary background on every side and the asymmetry vanishes exactly
    when it matters most.

    The shadow is inside the detected quad, and it is darker than BOTH the card
    and the background, because it is background with the light taken away.
    Measured on a synthetic capture at 30 degrees elevation: card border 188,
    backdrop 34, shadow 14. That third level, present on one side only, is an
    unambiguous signature.
    """
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ).astype(np.float64)
    h, w = gray.shape
    q = np.asarray(card_quad, dtype=np.float64).reshape(4, 2)
    centre = q.mean(axis=0)
    names = ["top", "right", "bottom", "left"]
    warnings: list[str] = []

    def inner_profile(i: int) -> list[float]:
        a, b = q[i], q[(i + 1) % 4]
        mid = 0.5 * (a + b)
        edge = b - a
        n = np.array([-edge[1], edge[0]], dtype=np.float64)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            return []
        n /= nn
        if float(n @ (mid - centre)) < 0:
            n = -n  # outward
        ts = np.linspace(-0.3, 0.3, 40)
        out = []
        for d_mm in np.linspace(0.12, 1.6, 9):
            pts = mid[None, :] + ts[:, None] * edge[None, :] - (
                d_mm * px_per_mm
            ) * n[None, :]
            xi = np.round(pts[:, 0]).astype(int)
            yi = np.round(pts[:, 1]).astype(int)
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            if ok.sum() >= 8:
                out.append(float(np.median(gray[yi[ok], xi[ok]])))
        return out

    profiles = [inner_profile(i) for i in range(4)]
    minima = [min(p) if p else float("nan") for p in profiles]
    valid = [m for m in minima if not math.isnan(m)]
    if len(valid) < 3:
        return ShadowVerdict(0.0, None, 0.0, None, False, False, (), -1, False)

    reference = float(np.median(valid))
    darkest_idx = int(np.nanargmin(np.array(minima, dtype=float)))
    darkest = minima[darkest_idx]
    ratio = darkest / max(reference, 1e-9)
    directional = ratio < darkness_threshold
    best_asym = 1.0 - ratio

    darker = names[darkest_idx] if directional else None
    elevation: Optional[float] = None
    shadow = 0.0
    severe = False

    if directional:
        # Width of the contaminated band, read off the profile directly, which
        # is a far better estimate of the shadow than inferring it from
        # brightness ratios.
        prof = profiles[darkest_idx]
        depths = np.linspace(0.12, 1.6, len(prof))
        dark_mask = np.array(prof) < 0.7 * reference
        shadow = float(depths[dark_mask].max()) if dark_mask.any() else 0.2
        elevation = float(
            math.degrees(math.atan2(thickness_mm, max(shadow, 1e-6)))
        )
        severe = elevation < SAFE_ELEVATION_DEG
        warnings.append(
            f"a {shadow:.2f} mm edge shadow biases ONE border and not its "
            "opposite, so unlike most errors here it does not cancel in the ratio"
        )
        if shadow > 1.2:
            warnings.append(
                "the shadow is wide enough to overlap the printed border; the "
                "correction below cannot be trusted at this width"
            )

    # A shadow narrower than about a third of a typical border can be measured
    # off the profile well enough to subtract. Beyond that the dark band starts
    # to swallow the border itself and the correction stops being trustworthy.
    correctable = directional and shadow <= 1.2

    return ShadowVerdict(
        side_index=darkest_idx if directional else -1,
        correctable=correctable,
        asymmetry=best_asym,
        darker_side=darker,
        estimated_shadow_mm=shadow,
        estimated_elevation_deg=elevation,
        directional=directional,
        severe=severe,
        warnings=tuple(warnings),
    )
