"""Absolute dimensioning: getting real millimetres out of a photograph.

WHAT QUIPU'S PHOTOGRAMMETRY ACTUALLY DOES
------------------------------------------
`geospatial_relation.unproject_pixel` is a standard pinhole ray-cast: it takes a
pixel, builds a normalised ray from `CameraIntrinsics`, rotates it by the sensor
bearing/pitch, and intersects it with a ground plane. The scale comes from
`pose.alt` -- a KNOWN sensor altitude above that plane. Its uncertainty model is
`sigma_m = max(0.3, 0.05 * range)`: a 30 cm floor and 5% of range.

That is sound for a drone identifying objects at tens of metres. It does not
transfer here, for one reason that is worth stating plainly rather than
engineering around:

    A single uncalibrated photograph is scale-ambiguous. You cannot recover
    metric size from it without either a known camera-to-subject distance or a
    known length in the scene. QUIPU has the first (barometric altitude). A
    phone held over a display case has neither.

And its error floor is 300 mm. A trading card is 63.5 mm wide.

So the mechanism transfers; the scale source does not. This module supplies the
missing piece the only way it can be supplied: a REFERENCE OBJECT of known
length, coplanar with the card.

WHY BOTHER, GIVEN CENTERING IS SCALE-INVARIANT
-----------------------------------------------
Centering is a ratio, so scale cancels and none of this affects it. Absolute
dimensioning buys something different and arguably more valuable: TRIM
DETECTION. A trimmed card -- one cut down to improve its centering -- is
undersized, and every grading company rejects it outright. It is one of the
hobby's real fraud vectors, and it is invisible to a ratio because trimming
*improves* the ratio. Measuring the card's actual outline in millimetres is the
only way to see it.

THE ERROR BUDGET, QUANTIFIED
-----------------------------
Scale uncertainty propagates directly into every dimension:

    reference                     tolerance     scale err   on 63.5 mm
    ISO/IEC 7810 ID-1 bank card   +/-0.13 mm      0.152%     +/-0.097 mm
    printed ArUco, unverified     +/-0.30 mm      0.35%      +/-0.22  mm
    same, measured with calipers  +/-0.02 mm      0.023%     +/-0.015 mm

A meaningful trim removes 0.5-1.5 mm. Standard cutting tolerance on a legitimate
card is roughly +/-0.2 mm. So a bank card as reference detects gross trims and
nothing marginal; only a self-verified target reaches caliper grade. The module
reports which regime it is in rather than implying the tighter one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from .geometry import apply_h, order_quad, refine_quad
from .types import (
    STANDARD_CARD_H_MM,
    STANDARD_CARD_W_MM,
    DetectionError,
    Measured,
)

# ISO/IEC 7810 ID-1: the bank-card format. Width tolerance is the binding one.
ID1_WIDTH_MM = 85.60
ID1_HEIGHT_MM = 53.98
ID1_TOLERANCE_MM = 0.13

# Published cutting tolerance is not standardised across manufacturers; this is
# a working figure for modern machine-cut stock and is used only to decide when
# an undersize measurement is worth flagging.
TYPICAL_CUT_TOLERANCE_MM = 0.20


@dataclass(frozen=True)
class ScaleReference:
    """A known length in the scene, coplanar with the card."""

    name: str
    length_mm: float
    tolerance_mm: float
    verified_with_calipers: bool = False

    @property
    def relative_uncertainty(self) -> float:
        return self.tolerance_mm / self.length_mm

    @property
    def grade(self) -> str:
        r = self.relative_uncertainty
        if r <= 0.0005:
            return "caliper"
        if r <= 0.002:
            return "usable"
        return "coarse"


BANK_CARD = ScaleReference("ISO ID-1 bank card", ID1_WIDTH_MM, ID1_TOLERANCE_MM)


def caliper_verified(name: str, length_mm: float, tolerance_mm: float = 0.02) -> ScaleReference:
    """A target you measured yourself. The only route to caliper grade."""
    return ScaleReference(name, length_mm, tolerance_mm, verified_with_calipers=True)


@dataclass(frozen=True)
class Dimensions:
    width: Measured
    height: Measured
    scale_px_per_mm: Measured
    reference: ScaleReference
    coplanar_warning: bool
    warnings: tuple[str, ...]
    scale_term_mm: float = 0.0
    edge_term_mm: float = 0.0

    @property
    def limited_by(self) -> str:
        """Which term dominates the dimensional uncertainty.

        Worth reporting because the two have completely different fixes and
        neither helps the other. Below roughly 0.05% scale error the reference
        stops mattering and image resolution takes over: at 10 px/mm a half-pixel
        edge uncertainty is already 0.07mm on a 63.5mm card, which is more than a
        caliper-verified target contributes. Buying a better ruler at that point
        buys nothing.
        """
        if self.edge_term_mm > 1.5 * self.scale_term_mm:
            return "image resolution"
        if self.scale_term_mm > 1.5 * self.edge_term_mm:
            return "scale reference"
        return "both roughly equally"

    @property
    def width_delta(self) -> Measured:
        return Measured(
            self.width.value - STANDARD_CARD_W_MM, self.width.sigma
        )

    @property
    def height_delta(self) -> Measured:
        return Measured(
            self.height.value - STANDARD_CARD_H_MM, self.height.sigma
        )

    def describe(self) -> str:
        lines = [
            f"measured {self.width.value:.2f} x {self.height.value:.2f} mm "
            f"(nominal {STANDARD_CARD_W_MM} x {STANDARD_CARD_H_MM})",
            f"  width  {self.width_delta.value:+.2f} +/- {self.width.sigma:.2f} mm",
            f"  height {self.height_delta.value:+.2f} +/- {self.height.sigma:.2f} mm",
            f"  scale reference: {self.reference.name} "
            f"({self.reference.grade} grade, "
            f"{self.reference.relative_uncertainty * 100:.3f}%)",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def scale_from_reference(
    reference_quad: np.ndarray,
    reference: ScaleReference,
    long_edge: bool = True,
) -> Measured:
    """Pixels per millimetre, from a reference object of known length.

    The quad must be the reference object's outline in the SAME rectified frame
    as the card, so that both share one perspective correction. Measuring the
    reference in the raw image and the card in a rectified one silently mixes
    two different scales.
    """
    q = order_quad(np.asarray(reference_quad, dtype=np.float64).reshape(4, 2))
    e01 = float(np.linalg.norm(q[1] - q[0]))
    e12 = float(np.linalg.norm(q[2] - q[1]))
    e23 = float(np.linalg.norm(q[3] - q[2]))
    e30 = float(np.linalg.norm(q[0] - q[3]))

    horizontal = 0.5 * (e01 + e23)
    vertical = 0.5 * (e12 + e30)
    px = max(horizontal, vertical) if long_edge else min(horizontal, vertical)

    # Opposite sides of a rectangle viewed fronto-parallel are equal. Residual
    # inequality is uncorrected perspective, and it biases scale.
    pair = (e01, e23) if (horizontal >= vertical) == long_edge else (e12, e30)
    asymmetry = abs(pair[0] - pair[1]) / max(1e-9, px)

    value = px / reference.length_mm
    rel = math.hypot(reference.relative_uncertainty, asymmetry)
    return Measured(value, value * rel)


def measure_absolute(
    card_quad: np.ndarray,
    reference_quad: np.ndarray,
    reference: ScaleReference = BANK_CARD,
    rectified: bool = True,
    reference_long_edge: bool = True,
    edge_sigma_px: float = 0.5,
) -> Dimensions:
    """Card outline in millimetres, using a coplanar reference for scale."""
    scale = scale_from_reference(reference_quad, reference, reference_long_edge)
    if scale.value <= 0:
        raise DetectionError("reference object has zero extent; scale undefined")

    q = order_quad(np.asarray(card_quad, dtype=np.float64).reshape(4, 2))
    w_px = 0.5 * (
        np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])
    )
    h_px = 0.5 * (
        np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])
    )
    if w_px > h_px:
        w_px, h_px = h_px, w_px

    rel_scale = scale.sigma / scale.value
    # Edge localisation contributes on top of scale error. Half a pixel per edge
    # is conservative for a line-fit corner; two edges per dimension.
    edge_px = math.sqrt(2) * edge_sigma_px
    width = w_px / scale.value
    height = h_px / scale.value
    w_scale_term = width * rel_scale
    w_edge_term = width * edge_px / max(w_px, 1e-9)
    w_sigma = math.hypot(w_scale_term, w_edge_term)
    h_sigma = height * math.hypot(rel_scale, edge_px / max(h_px, 1e-9))

    warnings: list[str] = []
    if not rectified:
        warnings.append(
            "card and reference were not measured in a common rectified frame; "
            "perspective makes the two scales differ and this result is "
            "indicative only"
        )
    if not reference.verified_with_calipers:
        warnings.append(
            f"reference tolerance is the published spec ({reference.tolerance_mm:.2f}mm), "
            "not a measurement of your actual object. Measuring it once with "
            f"calipers cuts the scale error by roughly "
            f"{reference.tolerance_mm / 0.02:.0f}x."
        )
    if reference.grade == "coarse":
        warnings.append(
            "at this reference grade only gross trims (>1mm) are detectable"
        )
    if w_edge_term > 1.5 * w_scale_term:
        warnings.append(
            f"the scale reference is no longer the limit: edge localisation "
            f"contributes {w_edge_term:.3f}mm against the reference's "
            f"{w_scale_term:.3f}mm. A better ruler buys nothing here; more "
            "pixels across the card does."
        )

    return Dimensions(
        width=Measured(width, w_sigma),
        height=Measured(height, h_sigma),
        scale_px_per_mm=scale,
        reference=reference,
        coplanar_warning=not rectified,
        warnings=tuple(warnings),
        scale_term_mm=w_scale_term,
        edge_term_mm=w_edge_term,
    )


@dataclass(frozen=True)
class TrimVerdict:
    undersize_w: Measured
    undersize_h: Measured
    z_worst: float
    likely_trimmed: bool
    detectable_floor_mm: float
    verdict: str
    warnings: tuple[str, ...]

    def describe(self) -> str:
        lines = [
            self.verdict,
            f"  width  {self.undersize_w.value:+.2f} +/- {self.undersize_w.sigma:.2f} mm vs nominal",
            f"  height {self.undersize_h.value:+.2f} +/- {self.undersize_h.sigma:.2f} mm vs nominal",
            f"  smallest trim this setup could detect: {self.detectable_floor_mm:.2f} mm",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def assess_trim(dims: Dimensions, z_threshold: float = 3.0) -> TrimVerdict:
    """Flag a card as possibly trimmed -- carefully, and never as an accusation.

    Two things keep this honest. First, the comparison is against nominal PLUS
    the legitimate cutting tolerance, because a card 0.15 mm under nominal is a
    normal card. Second, the detectable floor is reported: if the scale
    reference cannot resolve a 0.5 mm trim, saying "no trim detected" would be
    meaningless, so the verdict says what it could and could not have seen.
    """
    dw = dims.width_delta
    dh = dims.height_delta

    # Only undersize matters; a card cannot be trimmed larger.
    def z_under(d: Measured) -> float:
        excess = -(d.value) - TYPICAL_CUT_TOLERANCE_MM
        if excess <= 0:
            return 0.0
        return excess / max(d.sigma, 1e-9)

    zw, zh = z_under(dw), z_under(dh)
    z = max(zw, zh)
    floor = z_threshold * max(dw.sigma, dh.sigma) + TYPICAL_CUT_TOLERANCE_MM

    warnings: list[str] = list(dims.warnings)
    likely = z >= z_threshold

    if likely:
        verdict = (
            f"UNDERSIZE at {z:.1f} sigma beyond normal cutting tolerance. "
            "Consistent with trimming, but consistent is not proof -- verify "
            "with calipers before saying so to anyone."
        )
    elif floor > 1.5:
        verdict = (
            "no trim detected, but this setup could not have detected one "
            f"smaller than {floor:.2f} mm. Treat as uninformative rather than "
            "as a clean bill."
        )
        warnings.append(
            "scale reference is too coarse for a meaningful trim check"
        )
    else:
        verdict = (
            f"within normal cutting tolerance; a trim larger than {floor:.2f} mm "
            "would have shown up"
        )

    return TrimVerdict(
        undersize_w=dw,
        undersize_h=dh,
        z_worst=z,
        likely_trimmed=likely,
        detectable_floor_mm=floor,
        verdict=verdict,
        warnings=tuple(warnings),
    )


def find_reference_quad(
    image: np.ndarray,
    aspect: float = ID1_WIDTH_MM / ID1_HEIGHT_MM,
    tolerance: float = 0.12,
    exclude: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Locate a rectangular reference object by its aspect ratio.

    A bank card is 1.586:1 against a trading card's 1.4:1, which separates them
    reliably. ``exclude`` suppresses a quad already claimed as the card.
    """
    from .geometry import quad_candidates

    found = quad_candidates(image, min_area_frac=0.01)
    if not found:
        raise DetectionError("no rectangular reference object found in the frame")

    best = None
    for area, quad, contour in found:
        e01 = float(np.linalg.norm(quad[1] - quad[0]))
        e12 = float(np.linalg.norm(quad[2] - quad[1]))
        if min(e01, e12) < 1e-6:
            continue
        a = max(e01, e12) / min(e01, e12)
        if abs(a - aspect) / aspect > tolerance:
            continue
        if exclude is not None:
            if np.linalg.norm(quad.mean(axis=0) - exclude.mean(axis=0)) < 20:
                continue
        if best is None or area > best[0]:
            try:
                refined, _ = refine_quad(contour, quad)
            except DetectionError:
                continue
            best = (area, order_quad(refined))

    if best is None:
        raise DetectionError(
            f"no quad matching the reference aspect {aspect:.3f} was found. "
            "Place the reference flat, fully visible, beside the card."
        )
    return best[1]
