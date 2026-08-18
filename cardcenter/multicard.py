"""Scanning many cards at once, from a wide photo or a video pan.

The workflow this supports: walk a display case with the camera rolling, or take
one wide shot of a binder page, and get a centering ceiling for every card in
view. This is the mode where an automated tool genuinely beats a human -- not
because it grades better, but because it measures forty cards while you measure
one, and it never gets bored on card thirty.

Two things make it work:

DEDUPE BY APPEARANCE, NOT POSITION. A video pan moves every card in the frame,
so centroid tracking fails immediately. Cards are grouped by a perceptual hash
of their rectified crop, which is stable under the pan, the lighting change, and
the angle change, and which separates two different cards of the same set.

BEST FRAME WINS, NOT THE AVERAGE OF ALL FRAMES. Across a pan, a given card is
sharp in two frames and smeared in fifteen. Averaging everything drags the good
frames down. The scanner keeps the frames that pass the quality gate, measures
those, and combines them with inverse-variance weighting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from .capture import FrameQuality, RunningRatio, assess_frame
from .centering import measure_centering
from .geometry import enforce_portrait, order_quad, quad_candidates, refine_quad, rectify
from .types import (
    STANDARD_CARD_H_MM,
    STANDARD_CARD_W_MM,
    CaptureSpec,
    CenteringResult,
    DetectionError,
    Measured,
    resolve_holder,
)


# A card boundary located in a wide, lower-resolution frame carries about a
# pixel of placement bias that the line-fit residual does not capture: the
# residual measures how STRAIGHT the edge is, not whether it sits a pixel inside
# or outside the true edge, and a single thresholding pass can land either way
# depending on where the threshold cut the edge ramp. Measured against synthetic
# ground truth, omitting this term drops 95% interval coverage to 80%; including
# it restores full coverage at a z-score spread of 0.70, i.e. honest and mildly
# conservative. Raising it further only wastes precision.
SCENE_BOUNDARY_FLOOR_PX = 1.0


def dhash(image: np.ndarray, size: int = 8) -> int:
    """Difference hash of a card crop. Stable under exposure and mild blur."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(image, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for i, v in enumerate(diff.ravel()):
        if v:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _overlap(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> tuple[float, float]:
    """Return (IoU, fraction of `a` contained in `b`)."""
    ma = np.zeros(shape, dtype=np.uint8)
    mb = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(ma, a.astype(np.int32), 1)
    cv2.fillConvexPoly(mb, b.astype(np.int32), 1)
    inter = float((ma & mb).sum())
    union = float((ma | mb).sum())
    area_a = float(ma.sum())
    iou = inter / union if union > 0 else 0.0
    contained = inter / area_a if area_a > 0 else 0.0
    return iou, contained


@dataclass
class CardObservation:
    """One card seen in one frame."""

    quad: np.ndarray
    frame_index: int
    quality: FrameQuality
    result: Optional[CenteringResult] = None
    error: Optional[str] = None
    phash: int = 0


@dataclass
class ScannedCard:
    """One physical card, possibly seen across many frames."""

    card_id: int
    phash: int
    observations: list[CardObservation] = field(default_factory=list)
    horizontal: RunningRatio = field(default_factory=RunningRatio)
    vertical: RunningRatio = field(default_factory=RunningRatio)
    thumbnail: Optional[np.ndarray] = None

    @property
    def n_measured(self) -> int:
        return len(self.horizontal)

    @property
    def worst_ratio(self) -> Optional[Measured]:
        cands = [c for c in (self.horizontal.combined, self.vertical.combined) if c]
        return max(cands, key=lambda m: m.value) if cands else None

    @property
    def worst_axis(self) -> Optional[str]:
        h, v = self.horizontal.combined, self.vertical.combined
        if h is None and v is None:
            return None
        if v is None or (h is not None and h.value >= v.value):
            return "horizontal"
        return "vertical"

    @property
    def failure_reasons(self) -> list[str]:
        return sorted({o.error for o in self.observations if o.error})

    def summary(self) -> str:
        w = self.worst_ratio
        if w is None:
            reasons = self.failure_reasons
            why = reasons[0][:70] if reasons else "no frame passed the quality gate"
            return f"card {self.card_id}: NOT MEASURED -- {why}"
        lo, hi = w.interval()
        return (
            f"card {self.card_id}: {w.value:.1f}/{100 - w.value:.1f} "
            f"(95% CI {lo:.1f}-{hi:.1f}) on {self.worst_axis}, "
            f"{self.n_measured} frame(s)"
        )


def _crop_quad(image: np.ndarray, quad: np.ndarray, margin_px: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Crop a card out of a wide frame with margin, returning (crop, shifted_quad)."""
    x, y, w, h = cv2.boundingRect(quad.astype(np.int32))
    x0 = max(0, x - margin_px)
    y0 = max(0, y - margin_px)
    x1 = min(image.shape[1], x + w + margin_px)
    y1 = min(image.shape[0], y + h + margin_px)
    return image[y0:y1, x0:x1], quad - np.array([x0, y0], dtype=np.float64)


def detect_cards_in_frame(
    image: np.ndarray, min_area_frac: float = 0.004, max_cards: int = 60
) -> list[tuple[np.ndarray, float]]:
    """Find every card in a wide frame, deduplicated by overlap and nesting.

    Returns (quad, line_fit_residual_px) per card. The residual travels with the
    quad because it is the honest measure of how well that boundary is known,
    and it feeds straight into the border-width error budget.
    """
    found = quad_candidates(image, min_area_frac=min_area_frac)
    found.sort(key=lambda x: -x[0])

    kept: list[tuple[np.ndarray, float]] = []
    kept_raw: list[np.ndarray] = []
    shape = image.shape[:2]
    for area, quad, contour in found:
        if len(kept) >= max_cards:
            break
        # Reject duplicates and, critically, nested detections. A card's printed
        # frame is itself a card-shaped rectangle, so a naive detector reports
        # every card twice: once for the cut edge and once for the inner border.
        # Candidates are area-sorted, so anything largely inside something
        # already kept is the inner frame of a card we already have.
        redundant = False
        for k in kept_raw:
            iou, contained = _overlap(quad, k, shape)
            if iou > 0.35 or contained > 0.80:
                redundant = True
                break
        if redundant:
            continue
        try:
            refined, residual = refine_quad(contour, quad)
            refined = enforce_portrait(order_quad(refined))
        except DetectionError:
            continue
        if residual > 6.0:
            continue  # edges are not straight; this is not a cleanly visible card
        kept_raw.append(quad)
        kept.append((refined, float(residual)))
    return kept


@dataclass
class ScanReport:
    cards: list[ScannedCard]
    frames_processed: int
    frames_skipped: int
    detections: int

    def measured(self) -> list[ScannedCard]:
        return [c for c in self.cards if c.worst_ratio is not None]

    def unmeasured(self) -> list[ScannedCard]:
        return [c for c in self.cards if c.worst_ratio is None]

    def summary(self) -> str:
        lines = [
            f"frames processed : {self.frames_processed}",
            f"frames skipped   : {self.frames_skipped} (failed quality gate)",
            f"card detections  : {self.detections}",
            f"distinct cards   : {len(self.cards)}",
            f"measured         : {len(self.measured())}",
            f"not measured     : {len(self.unmeasured())}",
        ]
        return "\n".join(lines)


class MultiCardScanner:
    """Accumulates card observations across frames and groups them by identity."""

    def __init__(
        self,
        holder: str = "raw",
        capture: Optional[CaptureSpec] = None,
        hash_threshold: int = 12,
        enforce_quality: bool = True,
    ) -> None:
        self.holder_name = holder
        self.holder = resolve_holder(holder)
        self.capture = capture or CaptureSpec()
        self.hash_threshold = hash_threshold
        self.enforce_quality = enforce_quality
        self.cards: list[ScannedCard] = []
        self.frames_processed = 0
        self.frames_skipped = 0
        self.detections = 0

    def _match(self, phash: int, frame_index: int) -> Optional[ScannedCard]:
        """Find the card this observation belongs to.

        Two detections in the SAME frame are two different physical cards, no
        matter how alike they look -- a case of near-identical base cards would
        otherwise collapse into one entry. Identity matching only applies across
        frames.
        """
        best, best_d = None, self.hash_threshold + 1
        for c in self.cards:
            if c.observations and c.observations[-1].frame_index == frame_index:
                continue
            d = hamming(phash, c.phash)
            if d < best_d:
                best, best_d = c, d
        return best if best_d <= self.hash_threshold else None

    def add_frame(self, image: np.ndarray, frame_index: int = 0) -> int:
        """Detect, gate, measure and file every card in one frame."""
        self.frames_processed += 1
        quads = detect_cards_in_frame(image)
        measured_here = 0

        for quad, residual in quads:
            self.detections += 1
            crop, local_quad = _crop_quad(image, quad)
            if crop.size == 0:
                continue

            # Rectify first so the hash and the quality stats are computed on the
            # card itself rather than on whatever is behind it.
            try:
                rect, _ = rectify(crop, local_quad, px_per_mm=8.0)
            except Exception:
                continue

            side = 0.5 * (
                np.linalg.norm(quad[1] - quad[0]) / STANDARD_CARD_W_MM
                + np.linalg.norm(quad[3] - quad[0]) / STANDARD_CARD_H_MM
            )
            q = assess_frame(crop, local_quad, px_per_mm=float(side))
            phash = dhash(rect)

            card = self._match(phash, frame_index)
            if card is None:
                card = ScannedCard(card_id=len(self.cards) + 1, phash=phash)
                self.cards.append(card)
            if card.thumbnail is None or q.passed:
                card.thumbnail = cv2.resize(rect, (120, 168), interpolation=cv2.INTER_AREA)

            obs = CardObservation(
                quad=quad, frame_index=frame_index, quality=q, phash=phash
            )

            if self.enforce_quality and not q.passed:
                obs.error = q.describe()
                card.observations.append(obs)
                self.frames_skipped += 1
                continue

            try:
                res = measure_centering(
                    crop,
                    slab=self.holder,
                    capture=self.capture,
                    keep_rectified=False,
                    card_quad=local_quad,
                    quad_residual_px=max(residual, SCENE_BOUNDARY_FLOOR_PX),
                )
                obs.result = res
                card.horizontal.add(res.horizontal.ratio_pct)
                card.vertical.add(res.vertical.ratio_pct)
                measured_here += 1
            except DetectionError as exc:
                obs.error = str(exc).split("\n")[0][:160]
            card.observations.append(obs)

        return measured_here

    def report(self) -> ScanReport:
        return ScanReport(
            cards=self.cards,
            frames_processed=self.frames_processed,
            frames_skipped=self.frames_skipped,
            detections=self.detections,
        )


def iter_video_frames(
    path: str, stride: int = 6, max_frames: Optional[int] = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield every ``stride``-th frame. Consecutive video frames are nearly
    identical, so measuring all of them costs time and buys nothing."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise DetectionError(f"could not open video: {path}")
    try:
        idx = 0
        yielded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, frame
                yielded += 1
                if max_frames and yielded >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()


def scan_video(
    path: str,
    holder: str = "raw",
    capture: Optional[CaptureSpec] = None,
    stride: int = 6,
    max_frames: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> ScanReport:
    scanner = MultiCardScanner(holder=holder, capture=capture)
    for i, (idx, frame) in enumerate(iter_video_frames(path, stride, max_frames)):
        scanner.add_frame(frame, frame_index=idx)
        if progress:
            progress(i, len(scanner.cards))
    return scanner.report()


def scan_image(
    image: np.ndarray,
    holder: str = "raw",
    capture: Optional[CaptureSpec] = None,
    enforce_quality: bool = True,
) -> ScanReport:
    scanner = MultiCardScanner(
        holder=holder, capture=capture, enforce_quality=enforce_quality
    )
    scanner.add_frame(image, frame_index=0)
    return scanner.report()
