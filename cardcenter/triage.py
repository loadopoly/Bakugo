"""Routing a real capture session: what is each photograph FOR?

WHY THIS EXISTS
---------------
A real session is not 200 measurable card photos. Running the first 60 images of
an actual shoot through the measurement pipeline produced:

    0 measured, 60 refused

and the refusal reasons were not one problem but five, most of which are not
measurement failures at all. The images are a mixture:

    * cards with a caliper in frame, whole card visible   -> measurable
    * cards with the frame cropping an edge               -> unmeasurable, reshoot
    * the phone, its case, the lens cover                 -> equipment reference
    * the desk and lamp                                   -> lighting reference
    * a settings screenshot                               -> metadata, not an image

Feeding all of those to `measure_centering` and reporting refusals is a category
error: most were never candidates. The tool needs to sort them first and route
each kind to what it is actually good for.

WHAT THE REAL DATA SHOWED, MEASURED
------------------------------------
    complete card quad found          89 / 200   (44%)
    no closed quad (cropped/absent)  111 / 200   (56%)
    median card fill of frame        3.6% of pixels
    best case                        55.6%

At 3.6% of a 1600px frame the card spans roughly 190x270 px, which is about
3 px/mm -- below the usable floor. The single largest quality problem in this
dataset is not focus or lighting, it is **standoff distance**: the card is too
small in frame. That is a capture instruction, not an algorithm fix, and it is
the most valuable thing this module can tell a user.

THE CALIPER IS THE HEADLINE FINDING
------------------------------------
Nearly every card frame has digital calipers physically in shot, resting on the
same surface as the card. That is the coplanar reference `photogrammetry.py` was
built around and never had. It means absolute dimensioning -- and therefore trim
detection -- is genuinely available on this data, which centering alone can never
provide because centering is a ratio and scale cancels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np


class Route(str, Enum):
    """What a given photograph should be used for."""

    MEASURE = "measure"                # complete card, enough resolution
    MEASURE_LOW_RES = "measure_low_res"  # complete card, too small to trust
    RESHOOT_CROPPED = "reshoot_cropped"  # card present but running off-frame
    CALIBRATION = "calibration"          # caliper dominant / card absent
    EQUIPMENT = "equipment"              # phone, case, lens cover
    ENVIRONMENT = "environment"          # desk, lamp, empty surface
    SCREENSHOT = "screenshot"            # UI capture, not a photograph
    UNUSABLE = "unusable"

    @property
    def is_measurable(self) -> bool:
        return self in (Route.MEASURE, Route.MEASURE_LOW_RES)


# A card must span at least this to be worth measuring. Derived from the
# resolution work: below ~6 px/mm border localisation stops being reliable, and
# a 63.5mm card at 6 px/mm needs ~380px across.
MIN_CARD_WIDTH_PX = 380
GOOD_CARD_WIDTH_PX = 700

# Screenshots have exact device aspect ratios and no lens vignetting.
SCREENSHOT_ASPECTS = ((1080, 2400), (1080, 2410), (1440, 3120), (1080, 2340))


@dataclass(frozen=True)
class Triage:
    path: str
    route: Route
    card_width_px: float
    px_per_mm: Optional[float]
    caliper_present: bool
    edge_contact: tuple[str, ...]
    sharpness: float
    reason: str
    advice: str

    def describe(self) -> str:
        head = f"{self.route.value:<18} {self.path}"
        if self.px_per_mm:
            head += f"  ({self.px_per_mm:.1f} px/mm)"
        lines = [head, f"    {self.reason}"]
        if self.advice:
            lines.append(f"    -> {self.advice}")
        return "\n".join(lines)


def _is_screenshot(image: np.ndarray) -> bool:
    """Screenshots have exact device aspect ratios and no lens vignetting.

    The vignette test alone is not sufficient and was producing false positives.
    A card held over a BULK BIN of other cards fills the frame edge-to-edge with
    similarly-lit material, so corner and centre luminance match and the frame
    reads as unvignetted. Measured on a real bulk-bin session: 10 of 160 card
    photographs were routed to SCREENSHOT and dropped from measurement.

    A screenshot is now required to BOTH lack vignetting and carry an exact
    device aspect ratio. Photographs from a phone camera do not have UI aspect
    ratios, so the conjunction is safe in a way either test alone is not.
    """
    h, w = image.shape[:2]
    exact_device_ratio = any(
        (w, h) in ((sw, sh), (sh, sw)) for sw, sh in SCREENSHOT_ASPECTS
    )
    if not exact_device_ratio:
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    k = max(8, min(h, w) // 12)
    corners = [
        gray[:k, :k].mean(), gray[:k, -k:].mean(),
        gray[-k:, :k].mean(), gray[-k:, -k:].mean(),
    ]
    centre = gray[h // 2 - k : h // 2 + k, w // 2 - k : w // 2 + k].mean()
    if centre < 1e-6:
        return False
    return abs(np.mean(corners) - centre) / centre < 0.02


def _detect_caliper(image: np.ndarray) -> bool:
    """Look for the caliper's steel scale: a long, straight, low-saturation bar.

    Deliberately crude. Its job is to flag that a metric reference is in frame,
    not to locate it precisely -- that is `photogrammetry.find_reference_quad`'s
    work once a frame has been routed here.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    # Metal: low saturation, mid-to-high value.
    # Holographic foil is also low-saturation and bright, so the naive metal
    # test fires on any holo card. Measured on a bulk-bin session: 18 of 160
    # frames were routed to CALIBRATION with no caliper anywhere in shot.
    #
    # A caliper's steel scale is additionally FLAT -- near-uniform luminance
    # across its width. Holo foil is not: it is full of high-frequency specular
    # structure. Requiring low local variance as well as low saturation
    # separates them.
    metal = ((sat < 0.22) & (val > 0.35)).astype(np.uint8) * 255
    blur = cv2.blur(val, (9, 9))
    local_var = cv2.blur(val * val, (9, 9)) - blur * blur
    flat = (local_var < 0.0010).astype(np.uint8) * 255
    metal = cv2.bitwise_and(metal, flat)
    metal = cv2.morphologyEx(metal, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(metal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]
    for c in cnts:
        if cv2.contourArea(c) < 0.004 * h * w:
            continue
        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        if min(rw, rh) < 1e-6:
            continue
        elong = max(rw, rh) / min(rw, rh)
        # A caliper beam is long, thin AND straight. Requiring high elongation
        # plus a high fill of its own minimum-area rectangle rejects the
        # irregular bright patches holo foil produces.
        fill = cv2.contourArea(c) / max(rw * rh, 1e-9)
        if elong > 5.5 and max(rw, rh) > 0.27 * max(h, w) and fill > 0.55:
            return True
    return False


def _edge_contact(quad: np.ndarray, shape: tuple[int, int], margin: int = 6) -> tuple[str, ...]:
    """Which frame edges the detected card touches -- i.e. where it is cropped."""
    h, w = shape[:2]
    out = []
    if quad[:, 0].min() <= margin:
        out.append("left")
    if quad[:, 0].max() >= w - margin:
        out.append("right")
    if quad[:, 1].min() <= margin:
        out.append("top")
    if quad[:, 1].max() >= h - margin:
        out.append("bottom")
    return tuple(out)


def _cropped_card_present(image: np.ndarray) -> bool:
    """Is there a card-like region running off the frame edge?

    When a card is cropped there is no closed quadrilateral, so the ordinary
    detector finds nothing and the frame looks empty. But a cropped card still
    presents long straight high-contrast edges touching the border, which
    distinguishes 'you cut the card off' from 'there is no card here'. The
    difference matters: the first is a reshoot, the second is not a card photo
    at all.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    edges = cv2.Canny(cv2.bilateralFilter(gray, 7, 50, 50), 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=90,
        minLineLength=int(0.28 * min(h, w)), maxLineGap=12,
    )
    if lines is None:
        return False
    margin = max(8, min(h, w) // 60)
    touching = 0
    for x1, y1, x2, y2 in lines[:, 0]:
        near = (
            min(x1, x2) <= margin or max(x1, x2) >= w - margin
            or min(y1, y2) <= margin or max(y1, y2) >= h - margin
        )
        if near:
            touching += 1
    return touching >= 2


def triage(path: str, image: Optional[np.ndarray] = None, long_side: int = 1600) -> Triage:
    """Decide what a single photograph is for."""
    from .geometry import quad_candidates

    img = image if image is not None else cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return Triage(path, Route.UNUSABLE, 0.0, None, False, (), 0.0,
                      "file could not be decoded as an image", "")

    if _is_screenshot(img):
        return Triage(path, Route.SCREENSHOT, 0.0, None, False, (), 0.0,
                      "device aspect ratio and no lens vignetting",
                      "metadata only; not a measurement input")

    scale = long_side / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    caliper = _detect_caliper(small)

    cands = quad_candidates(small, min_area_frac=0.008)
    if cands:
        area, quad, _ = max(cands, key=lambda x: x[0])
        e0 = float(np.linalg.norm(quad[1] - quad[0]))
        e1 = float(np.linalg.norm(quad[3] - quad[0]))
        card_w = min(e0, e1)
        # Convert to the FULL-resolution equivalent, since measurement can run
        # on the original file rather than this downscaled copy.
        full_w = card_w / max(scale, 1e-9)
        ppm = full_w / 63.5
        contact = _edge_contact(quad, small.shape)

        if contact:
            return Triage(path, Route.RESHOOT_CROPPED, full_w, ppm, caliper, contact, sharp,
                          f"card touches frame edge ({', '.join(contact)})",
                          "step back or recompose; all four card edges must be inside the frame")
        if card_w < MIN_CARD_WIDTH_PX * scale:
            return Triage(path, Route.MEASURE_LOW_RES, full_w, ppm, caliper, (), sharp,
                          f"card spans only {ppm:.1f} px/mm",
                          "move closer or zoom; below ~6 px/mm border location is unreliable")
        return Triage(path, Route.MEASURE, full_w, ppm, caliper, (), sharp,
                      f"complete card at {ppm:.1f} px/mm"
                      + (" with caliper reference" if caliper else ""),
                      "" if caliper else "no metric reference in frame; centering only, no absolute size")

    # No closed quad. Distinguish a cropped card from a frame with no card.
    if _cropped_card_present(small):
        return Triage(path, Route.RESHOOT_CROPPED, 0.0, None, caliper, (), sharp,
                      "straight card-like edges run off the frame; no closed outline",
                      "the card is cut off. Step back until all four edges are visible")
    if caliper:
        return Triage(path, Route.CALIBRATION, 0.0, None, True, (), sharp,
                      "caliper present, no complete card",
                      "usable for scale calibration; set the opening and tap the jaw tips")
    return Triage(path, Route.ENVIRONMENT, 0.0, None, False, (), sharp,
                  "no card and no reference detected",
                  "lighting or equipment reference; not a measurement input")


@dataclass
class SessionReport:
    items: list[Triage]

    def by_route(self) -> dict[Route, list[Triage]]:
        out: dict[Route, list[Triage]] = {}
        for t in self.items:
            out.setdefault(t.route, []).append(t)
        return out

    def summary(self) -> str:
        groups = self.by_route()
        total = len(self.items)
        lines = [f"{total} images"]
        for route in Route:
            n = len(groups.get(route, []))
            if n:
                lines.append(f"  {route.value:<18} {n:4d}  ({100 * n / total:.0f}%)")

        measurable = [t for t in self.items if t.route.is_measurable]
        with_ref = [t for t in measurable if t.caliper_present]
        lines.append("")
        lines.append(f"measurable: {len(measurable)}, of which {len(with_ref)} have a metric reference")

        if measurable:
            ppm = np.array([t.px_per_mm for t in measurable if t.px_per_mm])
            if ppm.size:
                lines.append(
                    f"resolution: median {np.median(ppm):.1f} px/mm, "
                    f"range {ppm.min():.1f}-{ppm.max():.1f}"
                )

        cropped = len(groups.get(Route.RESHOOT_CROPPED, []))
        if cropped > 0.2 * total:
            lines.append("")
            lines.append(
                f"DOMINANT PROBLEM: {cropped} images ({100 * cropped / total:.0f}%) have the "
                "card running off the frame. Step back so all four edges are visible -- "
                "this single change would recover most of the session."
            )
        return "\n".join(lines)


def triage_session(paths: list[str]) -> SessionReport:
    return SessionReport([triage(p) for p in paths])
