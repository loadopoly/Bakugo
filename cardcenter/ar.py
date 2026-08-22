"""AR session: continuous capture, and calibrating scale against a caliper.

THROUGHPUT, MEASURED BEFORE DESIGNING
--------------------------------------
Full-frame detection is what costs, not the measurement:

    long side   detect   full measure   accuracy
      2400 px   1771 ms      2086 ms     0.20 pp
      1600 px    565 ms       683 ms     0.08 pp
      1200 px    300 ms       364 ms     0.06 pp
       720 px    148 ms       198 ms     0.21 pp
       540 px     65 ms       119 ms     0.23 pp

Nothing here runs a full detection at video rate, so an AR loop that tries to is
a loop that drops frames and feels broken. The design follows the numbers:

    TRACK    every frame, on a crop around the last known quad. Cheap, because
             the search space is a band rather than the whole image. This is what
             keeps the overlay glued to the card and drives live guidance.
    MEASURE  only on frames that pass the quality gate, at 1200 px, at a few Hz.
             Results accumulate in the existing inverse-variance combiner, so the
             band tightens while the user holds still.

Accuracy is flat from 1200 px up, so measuring at full sensor resolution buys
nothing and costs 6x the time.

CALIBRATION AGAINST A HELD CALIPER
-----------------------------------
An AR session already carries metric scale from visual-inertial odometry, but
VIO scale is good to a few percent and it drifts. A few percent on a 63.5 mm card
is 1-2 mm, which is useless for absolute work and catastrophic for trim
detection. A caliper opened to a known reading and held in frame fixes that: it
is a length you can read to +/-0.02 mm, which is the caliper-grade tier.

THE THING THAT WILL RUIN IT, IF IT RUINS ANYTHING: COPLANARITY. Scale from a
reference object is a ratio of apparent size to true size, and apparent size goes
as 1/distance. Hold the caliper 10% closer than the card and every dimension
comes out 10% wrong -- 6.3 mm on a card, an error a hundred times larger than the
thing being measured. Resting the caliper on the same surface as the card is not
a nicety; it is the whole measurement. This module estimates the depth mismatch
where it can and refuses the calibration where it cannot.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from .capture import FrameQuality, RunningRatio, assess_frame
from .centering import measure_centering
from .geometry import enforce_portrait, find_card_quad, order_quad, refine_quad
from .types import (
    STANDARD_CARD_H_MM,
    STANDARD_CARD_W_MM,
    CaptureSpec,
    CenteringResult,
    DetectionError,
    Measured,
    resolve_holder,
)

TRACK_LONG_SIDE = 540
MEASURE_LONG_SIDE = 1200

# VIO scale drifts over a session. Widen a calibration's uncertainty with age so
# a stale one stops being trusted silently.
DEFAULT_DRIFT_PER_HOUR = 0.01  # 1% per hour, relative


# ---------------------------------------------------------------------------
# Scale calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleCalibration:
    """Pixels per millimetre, with a provenance and an age."""

    px_per_mm: float
    sigma: float
    method: str
    observed_at: float
    reference_mm: float = 0.0
    drift_per_hour: float = DEFAULT_DRIFT_PER_HOUR
    warnings: tuple[str, ...] = ()

    def current(self, now: Optional[float] = None) -> Measured:
        """Scale as of now, with uncertainty widened for elapsed drift."""
        now = now if now is not None else time.time()
        hours = max(0.0, (now - self.observed_at) / 3600.0)
        drift = self.px_per_mm * self.drift_per_hour * hours
        return Measured(self.px_per_mm, math.hypot(self.sigma, drift))

    def age_hours(self, now: Optional[float] = None) -> float:
        now = now if now is not None else time.time()
        return max(0.0, (now - self.observed_at) / 3600.0)

    def stale(self, now: Optional[float] = None, limit_hours: float = 2.0) -> bool:
        return self.age_hours(now) > limit_hours

    def describe(self, now: Optional[float] = None) -> str:
        m = self.current(now)
        lines = [
            f"scale {m.value:.3f} +/- {m.sigma:.3f} px/mm  "
            f"({self.method}, {self.age_hours(now):.1f}h old)",
            f"  relative uncertainty {100 * m.sigma / max(m.value, 1e-9):.3f}% "
            f"-> +/-{STANDARD_CARD_W_MM * m.sigma / max(m.value, 1e-9):.3f} mm "
            "on a card width",
        ]
        if self.stale(now):
            lines.append("  STALE: re-shoot the caliper before trusting absolute sizes")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def calibrate_with_coplanarity_check(
    image: np.ndarray,
    p1_px: Sequence[float],
    p2_px: Sequence[float],
    opening_mm: float,
    card_quad: np.ndarray,
    reference_quad: np.ndarray,
    working_distance_mm: float = 250.0,
    **kw,
) -> tuple[ScaleCalibration, "object"]:
    """Calibrate, and MEASURE the coplanarity rather than assuming it.

    `depth_mismatch_frac` was previously a parameter the caller had to supply
    from information it did not have, so in practice it was always None and the
    dominant error source went unchecked. Defocus supplies it: blur on each
    object's own edges gives depth separation directly.
    """
    from .defocus import check_coplanarity

    check = check_coplanarity(
        image, card_quad, reference_quad, working_distance_mm=working_distance_mm
    )
    cal = calibrate_from_points(
        p1_px, p2_px, opening_mm,
        depth_mismatch_frac=check.scale_error_frac,
        **kw,
    )
    return cal, check


def calibrate_from_points(
    p1_px: Sequence[float],
    p2_px: Sequence[float],
    opening_mm: float,
    opening_tolerance_mm: float = 0.02,
    localisation_sigma_px: float = 1.5,
    depth_mismatch_frac: Optional[float] = None,
) -> ScaleCalibration:
    """Scale from two points a known distance apart -- the caliper's jaw tips.

    Tapping the jaw tips is deliberately offered alongside automatic detection.
    Calibration happens once and governs everything after it, so a slower method
    that is reliable beats a faster one that occasionally locks onto the wrong
    edge and silently rescales the whole session.
    """
    p1 = np.asarray(p1_px, dtype=np.float64)
    p2 = np.asarray(p2_px, dtype=np.float64)
    px = float(np.linalg.norm(p2 - p1))
    if px < 20.0:
        raise DetectionError(
            f"the two points are only {px:.0f} px apart. Open the caliper wider "
            "or move closer -- a short baseline makes the scale very uncertain."
        )
    if opening_mm <= 0:
        raise ValueError("caliper opening must be positive")

    scale = px / opening_mm
    tol_term = opening_tolerance_mm / opening_mm
    loc_term = math.sqrt(2.0) * localisation_sigma_px / px
    rel = math.hypot(tol_term, loc_term)

    warnings: list[str] = []
    # A caliper reads to +/-0.02mm, but that precision is thrown away if its jaws
    # cannot be LOCATED to better than a pixel or two. At a 500 px baseline,
    # 1.5 px of localisation error is 0.42% -- worse than simply using a bank
    # card, whose published tolerance is 0.152%. The caliper only wins when its
    # gap spans enough pixels, which means calibrating at full sensor resolution
    # with the caliper filling the frame.
    # The threshold that matters is not "localisation exceeds tolerance" but
    # "this is worse than the bank card anyone already has in their wallet".
    BANK_CARD_REL = 0.13 / 85.60
    if rel > BANK_CARD_REL:
        needed = math.sqrt(2.0) * localisation_sigma_px / max(
            math.sqrt(max(BANK_CARD_REL**2 - tol_term**2, 1e-12)), 1e-12
        )
        warnings.append(
            f"jaw localisation ({loc_term * 100:.3f}%) dominates the caliper's own "
            f"tolerance ({tol_term * 100:.3f}%). Calibrate at full sensor "
            f"resolution with the gap spanning ~{needed:.0f} px to use the "
            "caliper's real precision; below that a bank card would do as well."
        )
    if depth_mismatch_frac is not None:
        rel = math.hypot(rel, abs(depth_mismatch_frac))
        if abs(depth_mismatch_frac) > 0.03:
            warnings.append(
                f"the caliper appears {abs(depth_mismatch_frac) * 100:.0f}% "
                f"{'nearer' if depth_mismatch_frac > 0 else 'further'} than the "
                "card. Scale goes as 1/distance, so this alone is a "
                f"{abs(depth_mismatch_frac) * STANDARD_CARD_W_MM:.1f} mm error on "
                "a card width. Rest the caliper on the same surface."
            )
    else:
        warnings.append(
            "no depth information, so coplanarity is assumed rather than "
            "checked. Rest the caliper on the same surface as the card."
        )

    return ScaleCalibration(
        px_per_mm=scale,
        sigma=scale * rel,
        method="caliper",
        observed_at=time.time(),
        reference_mm=opening_mm,
        warnings=tuple(warnings),
    )


def detect_caliper_gap(
    image: np.ndarray, roi: Optional[tuple[int, int, int, int]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Find the two jaw faces automatically, as a projection-profile problem.

    A caliper's jaws present two strong, near-parallel edges bounding a gap. The
    gradient projected onto the axis across the gap has two dominant peaks; their
    separation is the opening. Returns the two midpoints.

    Raises rather than guessing when the two peaks are not clean, because a
    mis-detected calibration corrupts every measurement taken afterwards.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if roi is not None:
        x, y, w, h = roi
        gray = gray[y : y + h, x : x + w]
        origin = np.array([x, y], dtype=np.float64)
    else:
        origin = np.array([0.0, 0.0])

    gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Whichever axis carries more gradient energy is the one across the jaws.
    energy_x = float(np.abs(gx).sum())
    energy_y = float(np.abs(gy).sum())
    horizontal = energy_x >= energy_y
    profile = np.abs(gx).mean(axis=0) if horizontal else np.abs(gy).mean(axis=1)

    if profile.size < 20:
        raise DetectionError("calibration region is too small to find caliper jaws")

    # Relative to the peak, not a percentile. On a clean caliper shot most
    # columns are flat background, so the 92nd percentile of the gradient is
    # still zero and every column reads as "above threshold" -- one giant group
    # and no gap found.
    peak_value = float(profile.max())
    if peak_value <= 1e-6:
        raise DetectionError("no edges at all in the calibration region")
    thresh = 0.20 * peak_value
    peaks = []
    i = 0
    while i < len(profile):
        if profile[i] >= thresh:
            j = i
            while j + 1 < len(profile) and profile[j + 1] >= thresh:
                j += 1
            peaks.append((float(profile[i : j + 1].sum()), (i + j) / 2.0))
            i = j + 1
        else:
            i += 1

    if len(peaks) < 2:
        raise DetectionError(
            "could not find two jaw edges. Frame just the caliper gap against a "
            "plain background, or tap the jaw tips instead."
        )

    # Each jaw has two edges -- an outer and an inner face -- so a clean caliper
    # shot yields four peaks, not two. The measurement is between the INNER
    # faces, which are the pair bracketing the widest flat region. Picking the
    # two strongest peaks instead can straddle a single jaw and silently halve
    # the scale.
    centres = sorted(p[1] for p in peaks)
    flat = profile < thresh * 0.45
    best_pair, best_gap = None, 0.0
    for a_c, b_c in zip(centres, centres[1:]):
        lo, hi = int(math.ceil(a_c)) + 1, int(math.floor(b_c))
        if hi - lo < 3:
            continue
        span = hi - lo
        uniform = float(flat[lo:hi].mean())
        if uniform < 0.75:
            continue
        if span > best_gap:
            best_gap, best_pair = span, (a_c, b_c)

    if best_pair is None:
        raise DetectionError(
            "found edges but no clean gap between them. Frame just the caliper "
            "opening against a plain background, or tap the jaw tips instead."
        )
    top = [(0.0, best_pair[0]), (0.0, best_pair[1])]

    a, b = top[0][1], top[1][1]
    mid = (gray.shape[0] / 2.0) if horizontal else (gray.shape[1] / 2.0)
    if horizontal:
        p1 = np.array([a, mid]) + origin
        p2 = np.array([b, mid]) + origin
    else:
        p1 = np.array([mid, a]) + origin
        p2 = np.array([mid, b]) + origin
    return p1, p2


def verify_calibration_against_card(
    calibration: ScaleCalibration,
    card_quad: np.ndarray,
    now: Optional[float] = None,
) -> tuple[bool, str]:
    """Sanity-check a calibration by measuring the card it will be used on.

    A standard card is 63.5 mm wide. If a fresh calibration says otherwise by
    more than a couple of millimetres, something is wrong -- most likely the
    caliper was not coplanar. This cannot distinguish a bad calibration from a
    genuinely trimmed card, and it says so rather than picking one.
    """
    q = order_quad(np.asarray(card_quad, dtype=np.float64).reshape(4, 2))
    w_px = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    h_px = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    if w_px > h_px:
        w_px = h_px
    m = calibration.current(now)
    width_mm = w_px / max(m.value, 1e-9)
    delta = width_mm - STANDARD_CARD_W_MM

    if abs(delta) <= 0.35:
        return True, f"card measures {width_mm:.2f} mm; calibration looks sound"
    if abs(delta) <= 2.0:
        return False, (
            f"card measures {width_mm:.2f} mm against a nominal "
            f"{STANDARD_CARD_W_MM} mm ({delta:+.2f} mm). Either the caliper was "
            "not coplanar with the card, or this card is genuinely off-size. "
            "Re-calibrate with the caliper resting on the same surface; if the "
            "number persists, it is the card."
        )
    return False, (
        f"card measures {width_mm:.2f} mm ({delta:+.2f} mm off nominal). That is "
        "far too large to be a real card, so the calibration is wrong -- almost "
        "certainly a depth mismatch between the caliper and the card."
    )


# ---------------------------------------------------------------------------
# The session loop
# ---------------------------------------------------------------------------


def _resize_long(image: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= long_side:
        return image, 1.0
    s = long_side / longest
    return cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


def track_quad(
    image: np.ndarray, previous: np.ndarray, search_px: float = 14.0, samples: int = 22
) -> np.ndarray:
    """Re-find the card near where it was last frame, by searching edge normals.

    Cropping to a window around the previous quad and re-running full detection
    only bought 1.4x, because the card fills most of the frame and the crop is
    therefore most of the image. The cost is in the multi-strategy contour
    search, not in the pixel count.

    Between consecutive AR frames the card moves a few pixels, so the whole
    detection machinery is unnecessary. Sampling along each edge and stepping a
    short distance either way to find the strongest gradient turns detection into
    a line fit over a few hundred samples, which is what makes a per-frame
    overlay affordable.
    """
    q = np.asarray(previous, dtype=np.float64).reshape(4, 2)

    # Scale the normal search to the card's size in frame. A fixed radius that
    # suits a large card overshoots a small one: at 222x303 px a 14 px search
    # reaches a fifth of the way across the card and can lock onto the printed
    # frame instead of the cut edge, which showed up as a 10.4 px disagreement
    # with full detection where 3 px is the tolerance.
    shortest = min(
        float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)
    )
    search_px = float(np.clip(0.02 * shortest, 3.0, search_px))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    grad = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    h, w = grad.shape
    offsets = np.arange(-search_px, search_px + 1e-9, 1.0)

    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        edge = b - a
        L = float(np.linalg.norm(edge))
        if L < 8.0:
            raise DetectionError("previous quad is degenerate")
        n = np.array([-edge[1], edge[0]]) / L
        ts = np.linspace(0.12, 0.88, samples)
        base = a[None, :] + ts[:, None] * edge[None, :]

        found = []
        for pt in base:
            cand = pt[None, :] + offsets[:, None] * n[None, :]
            xi = np.round(cand[:, 0]).astype(int)
            yi = np.round(cand[:, 1]).astype(int)
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            if ok.sum() < 5:
                continue
            vals = np.where(ok, grad[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)], -1.0)
            k = int(np.argmax(vals))
            if vals[k] <= 0:
                continue
            found.append(pt + offsets[k] * n)

        if len(found) < 6:
            raise DetectionError("lost the card edge while tracking")
        pts = np.array(found)
        # Robust line fit: drop the worst quarter, which are usually a glare
        # streak or a neighbouring card's edge caught by the normal search.
        mean = pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts - mean, full_matrices=False)
        normal = np.array([-Vt[0][1], Vt[0][0]])
        resid = np.abs((pts - mean) @ normal)
        keep = pts[resid <= np.quantile(resid, 0.75)]
        if len(keep) < 4:
            keep = pts
        mean = keep.mean(axis=0)
        _, _, Vt = np.linalg.svd(keep - mean, full_matrices=False)
        normal = np.array([-Vt[0][1], Vt[0][0]])
        lines.append((normal[0], normal[1], -float(normal @ mean)))

    corners = []
    for i in range(4):
        l1, l2 = lines[(i - 1) % 4], lines[i]
        det = l1[0] * l2[1] - l2[0] * l1[1]
        if abs(det) < 1e-9:
            raise DetectionError("tracked edges are parallel")
        x = (l1[1] * l2[2] - l2[1] * l1[2]) / det
        y = (l1[2] * l2[0] - l2[2] * l1[0]) / det
        corners.append([x, y])
    out = np.array(corners, dtype=np.float64)

    # A tracker that has drifted onto something else is worse than one that
    # admits it lost the card, because the session would keep averaging.
    if float(np.abs(out - q).max()) > 4.0 * search_px:
        raise DetectionError("tracking drifted too far; re-detecting")
    return out


@dataclass
class ARStatus:
    """What the overlay should say right now."""

    tracking: bool
    quad: Optional[np.ndarray]
    guidance: tuple[str, ...]
    measured_frames: int
    seen_frames: int
    ratio: Optional[Measured]
    settled: bool
    scale: Optional[Measured] = None

    def headline(self) -> str:
        if not self.tracking:
            return "point at a card"
        if self.ratio is None:
            return "; ".join(self.guidance) or "hold steady"
        lo, hi = self.ratio.interval()
        tag = " (settled)" if self.settled else ""
        return (
            f"{self.ratio.value:.1f}/{100 - self.ratio.value:.1f}  "
            f"CI {lo:.1f}-{hi:.1f}{tag}"
        )


@dataclass
class ARSession:
    """Continuous measurement of one card while the camera is pointed at it.

    Two changes driven by real multi-view data:

    CONSISTENCY BEFORE COMBINATION. Two real views of the same card measured
    54.1 and 66.6 while each claimed +/-1.67. Naive inverse-variance pooling
    reported 54.18 +/- 0.288 -- a six-fold tightening onto an answer at most one
    input supports. The session now runs a chi2 test and refuses to pool
    inconsistent views.

    SEQUENTIAL STOPPING. A fixed frame count is wrong in both directions. A card
    at 68/32 against a 55/45 boundary is decided by the FIRST view; a card at
    55.0 is never decided and the user should be told that rather than handed a
    coin flip. SPRT stops as soon as the answer is settled.
    """

    holder: str = "raw"
    fov_deg: float = 68.0
    measure_interval_s: float = 0.35
    boundary: float = 55.0
    calibration: Optional[ScaleCalibration] = None
    horizontal: RunningRatio = field(default_factory=RunningRatio)
    vertical: RunningRatio = field(default_factory=RunningRatio)
    _measurements: list = field(default_factory=list)
    _sprt: Optional[object] = None
    _last_quad: Optional[np.ndarray] = None
    _last_measure: float = 0.0
    seen: int = 0
    measured: int = 0
    last_result: Optional[CenteringResult] = None

    def reset(self) -> None:
        """Start a new card. Combining frames across two different cards would
        produce a confident average of two unrelated things."""
        self.horizontal = RunningRatio()
        self.vertical = RunningRatio()
        self._last_quad = None
        self.measured = 0
        self.seen = 0
        self.last_result = None
        self._measurements = []
        self._sprt = None

    @property
    def worst_ratio(self) -> Optional[Measured]:
        cands = [c for c in (self.horizontal.combined, self.vertical.combined) if c]
        return max(cands, key=lambda m: m.value) if cands else None

    @property
    def fusion(self):
        """Consistency-checked combination of every view so far."""
        from .evidence import fuse

        return fuse(self._measurements)

    @property
    def verdict(self):
        from .evidence import Verdict

        f = self.fusion
        if f.n_views and not f.trustworthy:
            return Verdict.INCONSISTENT
        return self._sprt.verdict if self._sprt else Verdict.UNDECIDED

    @property
    def settled(self) -> bool:
        """Stop when the decision is made, not at an arbitrary frame count."""
        f = self.fusion
        if f.n_views < 2:
            return False
        if not f.trustworthy:
            return False  # disagreement is not settlement
        return bool(self._sprt and self._sprt.decided)

    @property
    def worth_continuing(self) -> bool:
        """Is another view likely to change the answer?

        Fisher information for a boundary decision peaks AT the boundary, so a
        card far from it is already decided and further capture is wasted.
        """
        from .evidence import information_value

        w = self.worst_ratio
        if w is None:
            return True
        if self.settled:
            return False
        return information_value(w, self.boundary) > 0.02

    def push(self, frame: np.ndarray, now: Optional[float] = None) -> ARStatus:
        """Feed one camera frame. Cheap unless the frame is worth measuring."""
        now = now if now is not None else time.time()
        self.seen += 1

        track_img, track_scale = _resize_long(frame, TRACK_LONG_SIDE)
        quad_small: Optional[np.ndarray] = None
        try:
            if self._last_quad is not None:
                quad_small = track_quad(track_img, self._last_quad * track_scale)
            else:
                quad_small, _, _ = find_card_quad(track_img)
        except DetectionError:
            try:
                quad_small, _, _ = find_card_quad(track_img)
            except DetectionError:
                self._last_quad = None
                return ARStatus(
                    tracking=False,
                    quad=None,
                    guidance=("point at a card, all four edges in frame",),
                    measured_frames=self.measured,
                    seen_frames=self.seen,
                    ratio=self.worst_ratio,
                    settled=self.settled,
                    scale=self.calibration.current(now) if self.calibration else None,
                )

        self._last_quad = quad_small / track_scale
        # The gate must judge the resolution the MEASUREMENT will have, not the
        # tracker's. Tracking runs at 540 px where a card is ~5 px/mm, which is
        # below the usable floor -- gating on that rejects every frame while the
        # measurement at 1200 px would have been comfortably fine.
        measure_scale = min(1.0, MEASURE_LONG_SIDE / max(frame.shape[:2]))
        full_px_per_mm = 0.5 * (
            np.linalg.norm(quad_small[1] - quad_small[0]) / STANDARD_CARD_W_MM
            + np.linalg.norm(quad_small[3] - quad_small[0]) / STANDARD_CARD_H_MM
        ) / track_scale
        quality = assess_frame(
            track_img, quad_small, px_per_mm=float(full_px_per_mm) * measure_scale
        )

        due = (now - self._last_measure) >= self.measure_interval_s
        if quality.passed and due:
            self._last_measure = now
            small, _ = _resize_long(frame, MEASURE_LONG_SIDE)
            try:
                res = measure_centering(
                    small,
                    slab=resolve_holder(self.holder),
                    capture=CaptureSpec.from_fov(self.fov_deg, small.shape),
                    keep_rectified=False,
                )
                self.horizontal.add(res.horizontal.ratio_pct)
                self.vertical.add(res.vertical.ratio_pct)
                self.last_result = res
                self.measured += 1

                from .evidence import SequentialBoundaryTest

                self._measurements.append(res.worst_ratio)
                if self._sprt is None:
                    self._sprt = SequentialBoundaryTest(threshold=self.boundary)
                self._sprt.update(res.worst_ratio)
            except DetectionError as exc:
                quality = FrameQuality(
                    sharpness=quality.sharpness,
                    glare_frac=quality.glare_frac,
                    clipped_frac=quality.clipped_frac,
                    dark_frac=quality.dark_frac,
                    px_per_mm=quality.px_per_mm,
                    tilt_deg=quality.tilt_deg,
                    passed=False,
                    guidance=(str(exc).split("\n")[0][:110],),
                )

        return ARStatus(
            tracking=True,
            quad=self._last_quad,
            guidance=quality.guidance,
            measured_frames=self.measured,
            seen_frames=self.seen,
            ratio=self.worst_ratio,
            settled=self.settled,
            scale=self.calibration.current(now) if self.calibration else None,
        )
