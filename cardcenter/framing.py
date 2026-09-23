"""Spend the pixel budget on the card, not on the room around it.

Both measurement paths used to cap the whole frame at a fixed size before
measuring: the still path at 2400 px, the AR path at 1200. That is fine when
the card fills the frame. Across a shop counter it is not: a card 400 px wide
in a 4000 px photo arrives 240 px wide after the cap, and the border
measurement loses nearly half its resolution for nothing -- the discarded
pixels are all table and other people's stock.

``frame_card_for_measure`` locates the card first (or takes the outline the AR
tracker already has), crops to it with margin, and only then applies the cap.
The card keeps every pixel the sensor gave it, up to the cap.

Intrinsics travel with the crop. Cropping does not change the lens, so the
focal length in pixels is unchanged by the crop and scales with the resize;
what moves is the principal point, which is the frame centre before the crop
and (centre - crop origin) * scale after it. Building a CaptureSpec from the
cropped image's width instead would report a much shorter focal length and
quietly corrupt the pose used for the refraction correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .types import STANDARD_CARD_W_MM, CaptureSpec, DetectionError

# Detection is not monotonic in scale: on a 3000x4000 frame with the card 509
# px wide, find_card_quad returns the card at a 2400 px working size and the
# artwork panel at 1400 px (measured). So try both and keep the larger find --
# the card contains the artwork, never the other way round, which is the same
# nesting rule find_card_quad uses internally.
DETECT_SCALES = (1400, 2400)
MARGIN_FRAC = 0.18
COARSE_MARGIN_FRAC = 0.6

# find_card_quad ignores regions under 0.8% of the frame. That is a fraction,
# and on a big sensor it bites before the pixel floor does: a card at 4.5 px/mm
# (the measurable floor) is 0.9% of a 12 MP photo but 0.2% of a 50 MP one, so
# the same card at the same distance was "not found" on the better camera. The
# floor that matters is pixels on the card, so the area limit is derived from
# it: anything smaller than 80% of a card at FLOOR_PX_PER_MM is not worth
# finding, anything larger is.
FLOOR_PX_PER_MM = 4.5
DEFAULT_MIN_AREA_FRAC = 0.008
MIN_AREA_FRAC_FLOOR = 0.0015


def min_area_frac_for(shape) -> float:
    """The smallest card worth finding in a frame of this size, as a fraction."""
    from .types import STANDARD_CARD_H_MM

    h, w = shape[:2]
    card_px = (STANDARD_CARD_W_MM * FLOOR_PX_PER_MM) * (STANDARD_CARD_H_MM * FLOOR_PX_PER_MM)
    frac = 0.8 * card_px / max(1.0, float(h) * float(w))
    return float(min(DEFAULT_MIN_AREA_FRAC, max(MIN_AREA_FRAC_FLOOR, frac)))


@dataclass(frozen=True)
class Framing:
    """The image to measure, and what is true about it."""

    image: np.ndarray
    capture: CaptureSpec
    scale: float                       # source pixels -> framed pixels
    origin: tuple                      # crop origin in the source image
    cropped: bool
    px_per_mm: Optional[float]         # the card's scale in the framed image
    source_shape: tuple
    # The outline located in the SOURCE frame, mapped into the framed image.
    # measure_centering takes it as card_quad: re-detecting inside a tight crop
    # finds a worse boundary (its own docstring says so, and measured here it
    # picked up the card's shadow and reported 50.00 +/- 1.05 against a true
    # 56.67).
    quad: Optional[np.ndarray] = None
    residual_px: float = 0.0

    def to_source(self, points: np.ndarray) -> np.ndarray:
        """Map points measured in the framed image back to the source frame."""
        pts = np.asarray(points, dtype=np.float64)
        return pts / self.scale + np.asarray(self.origin, dtype=np.float64)


def _capture_for(fov_deg: float, source_shape, origin, scale: float,
                 parent=None) -> CaptureSpec:
    """Intrinsics of the framed image.

    ``parent`` = (x, y, full_w, full_h) when the uploaded image is itself a
    crop of a larger photo (the phone crops to the tracked card before it
    uploads). The lens belongs to the full photo, so the focal length comes
    from ITS width and the principal point from ITS centre.
    """
    h, w = source_shape[:2]
    px, py = 0.0, 0.0
    if parent is not None:
        px, py, w, h = (float(v) for v in parent)
    focal_full = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    cx = (w / 2.0 - px - origin[0]) * scale
    cy = (h / 2.0 - py - origin[1]) * scale
    return CaptureSpec(focal_px=focal_full * scale, principal_point=(cx, cy))


def _area(quad) -> float:
    return abs(cv2.contourArea(np.asarray(quad, dtype=np.float32)))


def _locate_in(image: np.ndarray, prefer_point=None, long_side: int = 2400,
               min_area_frac: Optional[float] = None):
    """locate_card on a copy no larger than ``long_side``, mapped back."""
    from .edge_information import locate_card

    h, w = image.shape[:2]
    if min_area_frac is None:
        min_area_frac = min_area_frac_for(image.shape)
    s = min(1.0, long_side / max(h, w))
    small = (cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
             if s < 1.0 else image)
    pp = (prefer_point if prefer_point is not None
          else (small.shape[1] / 2.0, small.shape[0] / 2.0))
    try:
        quad, _, residual, _, _ = locate_card(small, prefer_point=pp,
                                              min_area_frac=min_area_frac)
    except DetectionError:
        return None
    return np.asarray(quad, dtype=np.float64) / s, float(residual) / max(s, 1e-9)


def _bbox(quad, image_shape, margin_frac):
    h, w = image_shape[:2]
    sides = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    margin = margin_frac * max(sides)
    x0 = max(0, int(math.floor(quad[:, 0].min() - margin)))
    y0 = max(0, int(math.floor(quad[:, 1].min() - margin)))
    x1 = min(w, int(math.ceil(quad[:, 0].max() + margin)))
    y1 = min(h, int(math.ceil(quad[:, 1].max() + margin)))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return 0, 0, w, h
    return x0, y0, x1, y1


def locate_coarse_to_fine(image: np.ndarray, prefer_point=None):
    """Public name for the two-pass locate: (quad, residual_px) in the
    image's own pixels, or None."""
    return _locate(image, prefer_point)


def card_region(image_shape, quad, margin_frac: float = COARSE_MARGIN_FRAC):
    """(x0, y0, x1, y1): the card plus margin, clipped to the image."""
    return _bbox(np.asarray(quad, dtype=np.float64).reshape(4, 2), image_shape, margin_frac)


def _locate(image: np.ndarray, prefer_point=None):
    """Find the card, coarse to fine.

    Two passes, because neither alone is reliable on a distant card:

    * Across scales, because the detector is not monotonic in resolution (see
      DETECT_SCALES). The largest find wins.
    * Then again inside a generous crop of the ORIGINAL pixels, where the card
      is several times larger than it was in the whole frame. Without this a
      first pass that locked onto the artwork panel would have the measurement
      report the artwork's borders -- symmetric, confident and wrong (measured
      50.00 +/- 1.05 against a true 56.67).
    """
    best = None
    for long_side in DETECT_SCALES:
        found = _locate_in(image, prefer_point, long_side)
        if found is not None:
            quad, res = found
            if best is None or _area(quad) > 1.02 * _area(best[0]):
                best = (quad, res)
        if max(image.shape[:2]) <= long_side:
            break                        # already ran at native resolution
    if best is None:
        return None
    coarse, coarse_res = best

    x0, y0, x1, y1 = _bbox(coarse, image.shape, COARSE_MARGIN_FRAC)
    crop = image[y0:y1, x0:x1]
    if min(crop.shape[:2]) < 64:
        return coarse, coarse_res
    refined = _locate_in(crop, None, DETECT_SCALES[-1])
    if refined is None:
        return coarse, coarse_res
    fine, fine_res = refined
    fine = fine + np.array([x0, y0], dtype=np.float64)
    return ((fine, fine_res) if _area(fine) >= 0.95 * _area(coarse)
            else (coarse, coarse_res))


def frame_card_for_measure(
    image: np.ndarray,
    *,
    quad: Optional[np.ndarray] = None,
    fov_deg: float = 68.0,
    max_side: int = 2400,
    margin_frac: float = MARGIN_FRAC,
    prefer_point=None,
    parent=None,
) -> Framing:
    """Crop to the card, then cap. Falls back to a plain cap when no card is
    found, so a frame this cannot locate behaves exactly as before.

    ``parent`` = (x, y, full_w, full_h) says ``image`` is already a crop of a
    larger photo; only the intrinsics use it (see ``_capture_for``)."""
    h, w = image.shape[:2]
    residual = 0.0
    if quad is None:
        found = _locate(image, prefer_point)
        if found is not None:
            quad, residual = found
    if quad is not None:
        quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)

    if quad is None:
        scale = min(1.0, max_side / max(h, w))
        framed = (cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                  if scale < 1.0 else image)
        return Framing(framed, _capture_for(fov_deg, image.shape, (0.0, 0.0), scale, parent),
                       scale, (0.0, 0.0), False, None, image.shape[:2], None, 0.0)

    sides = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    x0, y0, x1, y1 = _bbox(quad, image.shape, margin_frac)
    crop = image[y0:y1, x0:x1]
    scale = min(1.0, max_side / max(crop.shape[:2]))
    framed = (cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
              if scale < 1.0 else crop)
    short_px = min(min(sides[0], sides[1]), min(sides[2], sides[3])) * scale
    framed_quad = (quad - np.array([x0, y0], dtype=np.float64)) * scale
    return Framing(framed, _capture_for(fov_deg, image.shape, (x0, y0), scale, parent),
                   scale, (float(x0), float(y0)),
                   (x0, y0, x1, y1) != (0, 0, w, h),
                   short_px / STANDARD_CARD_W_MM, image.shape[:2],
                   framed_quad, residual * scale)
