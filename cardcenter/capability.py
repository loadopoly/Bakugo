"""What can be graded, for every card — not just bordered ones.

THE PROBLEM THIS FIXES
----------------------
`measure_centering` required all four printed borders to be detectable, and
refused otherwise. Measured against 75 real cards:

    all four borders detectable       0%
    three                             3%
    two                               9%
    one                              17%
    none                            71%

Zero percent. The requirement was not strict, it was unsatisfiable, and the tool
reported a refusal for every real card while its synthetic tests passed at 0.4pp
RMS. That is the whole synthetic-to-real gap in one number.

Two separate causes, and only one of them is about card design:

1. A THRESHOLD BUG, now fixed in `detect.py`. Otsu splits the deltaE field on its
   own histogram, which assumes the two classes are comparably populated. In a
   border strip they are not — most of the scan depth is artwork — so Otsu landed
   near the artwork mode instead of between border and artwork. On a real
   bordered card at 4.7 px/mm the signal was 14–27 while Otsu set the threshold
   at 31–38, so no column ever crossed it and all four sides failed silently.
   Anchoring the threshold to the measured step recovered 3 of 4 sides on that
   card and moved the fully-undetectable population from 85% to 71%.

2. MOST CARDS GENUINELY HAVE NO FULL BORDER. Full-arts, Hyper Rares, Energy,
   textured holos and modern alt-arts are full-bleed by design. Verified against
   a certified PSA GEM MT 10 (cert 143341329, 2024 TWM #225 Rescue Board): the
   detector found the card correctly at 1.446 aspect with 1.44 px residual, and
   the card simply has no printed border. That refusal was right.

THE FIX: GRADE WHAT IS THERE
-----------------------------
Centering on one axis needs one opposing PAIR, not four sides. A card with left
and right detectable yields a complete, honest horizontal measurement even
though the vertical is unknown — and that is exactly what graders do when they
call a card "off-centre left-right". Reporting nothing in that case throws away a
real measurement.

So capability is graded rather than binary:

    FULL          both axes measurable            centering ceiling, both axes
    SINGLE_AXIS   one opposing pair               ceiling on that axis only
    PARTIAL       sides but no opposing pair      no ratio; per-side widths only
    GEOMETRY_ONLY no borders at all               cut squareness, size, trim check
    NONE          card not located

The lower tiers are not consolation prizes. GEOMETRY_ONLY still detects trimming,
which is a fraud check centering can never perform because centering is a ratio
and scale cancels out of it. For a full-bleed Hyper Rare, geometry is the *only*
objective measurement available, and it is the one that matters most.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import cv2
import numpy as np

from .detect import SIDES, SideProfile, detect_side_border
from .geometry import find_card_quad, rectify
from .types import (
    STANDARD_CARD_H_MM,
    STANDARD_CARD_W_MM,
    BorderPair,
    DetectionError,
    Measured,
)

# Below this a side profile is too weak to use in a ratio.
MIN_SIDE_CONFIDENCE = 0.20

OPPOSING = {"horizontal": ("left", "right"), "vertical": ("top", "bottom")}

# Above this much corner deviation, perspective dominates and squareness cannot
# be separated from viewing angle. Measured on real photographs: 10-15 degrees is
# routine for a handheld oblique shot of a perfectly square card.
PERSPECTIVE_CONFOUND_DEG = 4.0

# A printed border runs parallel to the cut, so its width is near-constant along
# the side. Beyond this drift the detected boundary is probably not a border --
# on a slabbed card it is typically the label edge or the slab well.
MAX_BORDER_DRIFT = 0.02  # mm of width change per mm along the side


class Capability(str, Enum):
    FULL = "full"
    SINGLE_AXIS = "single_axis"
    PARTIAL = "partial"
    GEOMETRY_ONLY = "geometry_only"
    NONE = "none"

    @property
    def has_ratio(self) -> bool:
        return self in (Capability.FULL, Capability.SINGLE_AXIS)


@dataclass(frozen=True)
class CutGeometry:
    """What the card's outline alone reveals — available on every card.

    Squareness and size need no printed border, so they work on full-bleed cards
    where centering cannot. Corner angles departing from 90 degrees indicate a
    skewed cut; dimensions under nominal indicate trimming.
    """

    width_mm: Optional[float]
    height_mm: Optional[float]
    corner_angles_deg: tuple[float, float, float, float]
    max_angle_error_deg: float
    aspect: float
    notes: tuple[str, ...]

    @property
    def square(self) -> bool:
        return self.max_angle_error_deg < 1.5

    @property
    def squareness_assessable(self) -> bool:
        """False when perspective dominates and the number would be meaningless."""
        return self.max_angle_error_deg < PERSPECTIVE_CONFOUND_DEG

    def describe(self) -> str:
        if not self.squareness_assessable:
            lines = ["cut squareness: not assessable at this viewing angle"]
        else:
            lines = [
                f"cut squareness: worst corner {self.max_angle_error_deg:.2f}deg from 90"
                + ("  (square)" if self.square else "  (SKEWED CUT)")
            ]
        if self.width_mm and self.squareness_assessable:
            lines.append(
                f"outline {self.width_mm:.2f} x {self.height_mm:.2f} mm "
                f"(nominal {STANDARD_CARD_W_MM} x {STANDARD_CARD_H_MM})"
            )
        elif self.width_mm:
            lines.append(
                "outline dimensions withheld: perspective at this angle biases "
                "them more than a trim would"
            )
        for n in self.notes:
            lines.append(f"  {n}")
        return "\n".join(lines)


@dataclass(frozen=True)
class GradeCapability:
    """Everything measurable on this card, and what is not."""

    capability: Capability
    horizontal: Optional[BorderPair]
    vertical: Optional[BorderPair]
    sides: dict[str, SideProfile]
    geometry: Optional[CutGeometry]
    px_per_mm: float
    reason: str
    available: tuple[str, ...]
    unavailable: tuple[str, ...]

    @property
    def worst_ratio(self) -> Optional[Measured]:
        pairs = [p for p in (self.horizontal, self.vertical) if p is not None]
        if not pairs:
            return None
        return max((p.ratio_pct for p in pairs), key=lambda m: m.value)

    @property
    def worst_axis_name(self) -> Optional[str]:
        pairs = [(p.axis, p.ratio_pct.value) for p in (self.horizontal, self.vertical) if p]
        return max(pairs, key=lambda kv: kv[1])[0] if pairs else None

    def describe(self) -> str:
        lines = [f"capability: {self.capability.value} — {self.reason}"]
        w = self.worst_ratio
        if w is not None:
            lo, hi = w.interval()
            lines.append(
                f"  centering ({self.worst_axis_name}): {w.value:.1f}/{100 - w.value:.1f}"
                f"  95% CI {lo:.1f}-{hi:.1f}"
            )
        if self.capability is Capability.SINGLE_AXIS:
            missing = "vertical" if self.horizontal else "horizontal"
            lines.append(
                f"  {missing} axis not measurable — this is a ceiling on ONE axis, "
                "and the other could be worse"
            )
        for name, p in sorted(self.sides.items()):
            lines.append(f"  {name:<7} {p.depth_mm:.2f} mm  (confidence {p.confidence:.2f})")
        if self.geometry:
            for ln in self.geometry.describe().splitlines():
                lines.append("  " + ln)
        if self.available:
            lines.append("  available: " + ", ".join(self.available))
        if self.unavailable:
            lines.append("  NOT available: " + ", ".join(self.unavailable))
        return "\n".join(lines)


def cut_geometry(quad: np.ndarray, px_per_mm: float) -> CutGeometry:
    """Squareness and size from the card outline alone.

    Works on every card including full-bleed ones, because it uses the die cut
    rather than the printing. A skewed cut is a real defect graders penalise and
    it is invisible to centering, which only compares opposing border widths.
    """
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    angles = []
    for i in range(4):
        prev, cur, nxt = q[(i - 1) % 4], q[i], q[(i + 1) % 4]
        a, b = prev - cur, nxt - cur
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            angles.append(90.0)
            continue
        cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        angles.append(math.degrees(math.acos(cos)))
    worst = max(abs(a - 90.0) for a in angles)

    e0 = float(np.linalg.norm(q[1] - q[0]))
    e1 = float(np.linalg.norm(q[3] - q[0]))
    w_px, h_px = min(e0, e1), max(e0, e1)
    width = w_px / px_per_mm if px_per_mm > 0 else None
    height = h_px / px_per_mm if px_per_mm > 0 else None

    notes: list[str] = []
    # A quad measured in the IMAGE plane is perspective-distorted: an oblique
    # shot turns a perfectly square card into a trapezoid. Measured on real
    # photographs, corner deviations of 10-15 degrees are routine and are camera
    # angle, not cut quality. Reporting them as skewed cuts would flag almost
    # every real card as defective.
    #
    # Squareness is therefore only assertable when the card was shot close to
    # square-on. Above the threshold the measurement is withheld rather than
    # reported, because a number that mostly measures the photographer is worse
    # than no number.
    if worst >= PERSPECTIVE_CONFOUND_DEG:
        notes.append(
            f"corner angles deviate up to {worst:.1f}deg, which at this viewing "
            "angle is perspective rather than cut quality — squareness not "
            "assessed. Re-shoot square-on to measure it."
        )
    elif worst >= 1.5:
        notes.append(
            f"corner angles deviate up to {worst:.1f}deg from square — skewed cut, "
            "which centering cannot see"
        )
    if width and worst < PERSPECTIVE_CONFOUND_DEG and abs(width - STANDARD_CARD_W_MM) > 0.6:
        notes.append(
            f"outline width {width - STANDARD_CARD_W_MM:+.2f} mm from nominal; "
            "verify scale before treating this as a trim indication"
        )
    return CutGeometry(width, height, tuple(angles), worst, h_px / max(w_px, 1e-9), tuple(notes))


def assess(
    image: np.ndarray,
    px_per_mm: Optional[float] = None,
    min_confidence: float = MIN_SIDE_CONFIDENCE,
) -> GradeCapability:
    """Measure whatever this card supports, at the highest tier available."""
    try:
        quad, _, _ = find_card_quad(image)
    except DetectionError as exc:
        return GradeCapability(
            Capability.NONE, None, None, {}, None, 0.0, str(exc).split("\n")[0],
            (), ("centering", "cut geometry", "trim check"),
        )

    ppm = px_per_mm or float(np.linalg.norm(quad[1] - quad[0]) / STANDARD_CARD_W_MM)
    geom = cut_geometry(quad, ppm)
    rect, _ = rectify(image, quad, ppm)

    sides: dict[str, SideProfile] = {}
    for side in SIDES:
        try:
            p = detect_side_border(rect, side, ppm)
            if p.confidence >= min_confidence:
                sides[side] = p
        except DetectionError:
            continue

    def make_pair(axis: str) -> Optional[BorderPair]:
        lo_name, hi_name = OPPOSING[axis]
        if lo_name not in sides or hi_name not in sides:
            return None
        lo, hi = sides[lo_name], sides[hi_name]
        return BorderPair(
            axis=axis, low_name=lo_name, high_name=hi_name,
            low_mm=Measured(lo.depth_mm, lo.sigma_mm),
            high_mm=Measured(hi.depth_mm, hi.sigma_mm),
        )

    horiz, vert = make_pair("horizontal"), make_pair("vertical")

    # A real printed border is roughly constant width along its side. A boundary
    # that DRIFTS along the side is usually not a border at all -- it is slab
    # furniture, a label edge, or artwork that happens to change colour.
    #
    # Caught by the certified PSA GEM MT 10 (cert 143341329): the tool reported
    # 68.7/31.3 vertical on a card that PSA graded 10, which requires roughly
    # 55/45 or better. It was measuring the slab label boundary against the slab
    # well edge. A one-sided ground-truth label cannot confirm precision, but it
    # falsifies a result this far out, which is exactly what it did.
    def drifts(pair: Optional[BorderPair]) -> bool:
        if pair is None:
            return False
        for name in (pair.low_name, pair.high_name):
            p = sides.get(name)
            if p is not None and abs(p.slope_mm_per_mm) > MAX_BORDER_DRIFT:
                return True
        return False

    drift_notes: list[str] = []
    if drifts(horiz):
        drift_notes.append("horizontal border width drifts along the side")
        horiz = None
    if drifts(vert):
        drift_notes.append("vertical border width drifts along the side")
        vert = None
    always = ("cut squareness", "outline dimensions", "trim check")

    if horiz and vert:
        return GradeCapability(
            Capability.FULL, horiz, vert, sides, geom, ppm,
            "both axes measurable",
            ("centering (both axes)",) + always, (),
        )
    if horiz or vert:
        axis = "horizontal" if horiz else "vertical"
        other = "vertical" if horiz else "horizontal"
        return GradeCapability(
            Capability.SINGLE_AXIS, horiz, vert, sides, geom, ppm,
            f"only the {axis} border pair is detectable"
            + ("; " + "; ".join(drift_notes) if drift_notes else ""),
            (f"centering ({axis} only)",) + always,
            (f"centering ({other})",),
        )
    if sides:
        why = f"{len(sides)} border(s) found but no opposing pair, so no ratio"
        if drift_notes:
            why = "; ".join(drift_notes) + " -- rejected as not a printed border"
        return GradeCapability(
            Capability.PARTIAL, None, None, sides, geom, ppm, why,
            ("per-side border widths",) + always,
            ("centering (needs an opposing pair)",),
        )
    return GradeCapability(
        Capability.GEOMETRY_ONLY, None, None, {}, geom, ppm,
        "no printed border — full-art, Hyper Rare, Energy or textured holo",
        always,
        ("centering (no border exists to measure)",),
    )
