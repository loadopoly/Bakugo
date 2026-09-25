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
from .framing import frame_card_for_measure
from .edge_information import locate_card
from .geometry import (
    compute_edge_gradient,
    enforce_portrait,
    find_card_quad,
    order_quad,
    refine_quad,
    touches_frame_boundary,
)
from .types import (
    STANDARD_CARD_H_MM,
    STANDARD_CARD_W_MM,
    CaptureSpec,
    CenteringResult,
    DetectionError,
    Measured,
    resolve_holder,
)

def _default_gate_config():
    from .confidence import GateConfig

    return GateConfig()


TRACK_LONG_SIDE = 540
MEASURE_LONG_SIDE = 1200
FRAME_MARGIN_FRAC = 0.18

# Live-loop timing. Frames arrive a few times a second from a phone on shop
# signal, not at video rate, so between two pushes a hand-held phone moves the
# card well past the tracker's search band.
REDETECT_S = 1.0   # re-find the card from scratch at least this often
RECENT_S = 0.7     # a card seen this recently is the first place to look again
AIM_TTL_S = 3.0    # a tap names a card in the frames just after it, not forever

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
# Temporal Filtering (1€ Filter)
# ---------------------------------------------------------------------------


class LowPassFilter:
    """First-order low-pass exponential smoothing filter."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self._y: Optional[np.ndarray] = None

    def filter(self, val: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        if alpha is not None:
            self.alpha = float(np.clip(alpha, 0.0, 1.0))
        val_arr = np.asarray(val, dtype=np.float64)
        if self._y is None:
            self._y = val_arr.copy()
        else:
            self._y = self.alpha * val_arr + (1.0 - self.alpha) * self._y
        return self._y.copy()

    @property
    def last(self) -> Optional[np.ndarray]:
        return self._y

    def reset(self) -> None:
        self._y = None


class OneEuroFilter:
    """1€ Filter: Adaptive low-pass filter for interactive AR tracking.

    Minimizes jitter when stationary (low cutoff frequency fc_min) while eliminating
    lag during fast movement (cutoff increases dynamically with velocity).
    Casiez, Roussel, & Vogel (CHI 2012).
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.015,
        d_cutoff: float = 1.0,
    ):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.last_time: Optional[float] = None

    def _alpha(self, rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-4))
        te = 1.0 / max(rate, 1e-4)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if self.last_time is None or timestamp is None:
            rate = self.freq
        else:
            dt = timestamp - self.last_time
            rate = 1.0 / dt if dt > 1e-5 else self.freq
        self.last_time = timestamp

        prev_x = self.x_filt.last
        dx = np.zeros_like(x_arr) if prev_x is None else (x_arr - prev_x) * rate
        edx = self.dx_filt.filter(dx, self._alpha(rate, self.d_cutoff))
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(edx)
        a = self._alpha(rate, cutoff)
        return self.x_filt.filter(x_arr, a)

    def reset(self) -> None:
        self.x_filt.reset()
        self.dx_filt.reset()
        self.last_time = None


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


@dataclass(frozen=True)
class TrackStats:
    """How well the tracked sides fit their lines: the *achieved* uncertainty,
    to set against the Cramer-Rao floor from ``edge_information``."""

    side_rms_px: tuple
    side_points: tuple
    search_px: float

    @property
    def corner_sigma_px(self) -> float:
        """Worst corner: offset sigma per side (rms / sqrt(n)), extrapolated to
        the side's end, combined in quadrature for the two sides meeting there.
        Integer normal-search steps put a 1/sqrt(12) px quantisation floor
        under each point."""
        q = 1.0 / math.sqrt(12.0)
        end = [
            math.sqrt(max(r, q) ** 2 / max(n, 1)) * math.sqrt(1.0 + 3.0 * (0.5 / 0.38) ** 2)
            for r, n in zip(self.side_rms_px, self.side_points)
        ]
        return max(math.hypot(end[i - 1], end[i]) for i in range(4))


def track_quad(
    image: np.ndarray,
    previous: np.ndarray,
    search_px: float = 14.0,
    samples: int = 22,
    return_stats: bool = False,
):
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

    grad = compute_edge_gradient(image)
    h, w = grad.shape[:2]
    offsets = np.arange(-search_px, search_px + 1e-9, 1.0)

    lines = []
    side_rms: list[float] = []
    side_n: list[int] = []
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
        kept_resid = (keep - mean) @ normal
        side_rms.append(float(np.sqrt(np.mean(kept_resid**2))))
        side_n.append(int(len(keep)))

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
    if touches_frame_boundary(out, h, w):
        raise DetectionError("tracked quad touches frame boundary; re-detecting")
    if return_stats:
        return out, TrackStats(tuple(side_rms), tuple(side_n), float(search_px))
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
    grade_ceiling: Optional[str] = None
    bands: Optional[dict[str, str]] = None
    grade_estimate: Optional[str] = None
    grade_confidence: Optional[float] = None
    # Information floor of the tracked outline (edge_information.QuadInformation
    # .to_dict(), full-frame pixels) and the auto-accept decision from
    # confidence.gate on the fused ratio.
    information: Optional[dict] = None
    decision: Optional[str] = None
    decision_reason: str = ""
    # the tracked card in the pushed frame: its px/mm, and how much of the
    # frame's width it takes (the phone zooms in on the first when it is too
    # coarse to measure live)
    px_per_mm: Optional[float] = None
    card_frac: Optional[float] = None

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
    gate_config: "GateConfig" = field(default_factory=lambda: _default_gate_config())
    calibration: Optional[ScaleCalibration] = None
    horizontal: RunningRatio = field(default_factory=RunningRatio)
    vertical: RunningRatio = field(default_factory=RunningRatio)
    _measurements: list = field(default_factory=list)
    _sprt: Optional[object] = None
    _last_quad: Optional[np.ndarray] = None
    _quad_filter: OneEuroFilter = field(
        default_factory=lambda: OneEuroFilter(freq=30.0, min_cutoff=1.2, beta=0.02)
    )
    _last_measure: float = 0.0
    # Tracker settings driven by the measured uncertainty (see push()).
    _search_px: float = 14.0
    _sigma_meas_px: Optional[float] = None
    _last_info: Optional[object] = None
    _last_gate: Optional[object] = None
    seen: int = 0
    measured: int = 0
    last_result: Optional[CenteringResult] = None
    # camera pixels per pushed pixel (see push); 1.0 until the phone says
    source_scale: float = 1.0
    # the phone's camera zoom: the card looks this much bigger than it is
    # close, so "too close to focus" divides it out
    zoom: float = 1.0
    _aim_norm: Optional[tuple] = None
    _aim_at: Optional[float] = None
    # the outline as measured (unsmoothed): where the tracker starts next time
    _last_raw: Optional[np.ndarray] = None
    _last_valid_at: float = float("-inf")
    _last_detect_at: float = float("-inf")
    _lost_reason: str = ""
    # the last frame, small and grey, to measure how the view moved since
    _last_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Start a new card. Combining frames across two different cards would
        produce a confident average of two unrelated things."""
        self.horizontal = RunningRatio()
        self.vertical = RunningRatio()
        self._last_quad = None
        self._last_raw = None
        self._last_valid_at = float("-inf")
        self._last_detect_at = float("-inf")
        self._quad_filter.reset()
        self.measured = 0
        self.seen = 0
        self.last_result = None
        self._measurements = []
        self._sprt = None
        self._search_px = 14.0
        self._sigma_meas_px = None
        self._last_info = None
        self._last_gate = None
        # Where the user pointed (0..1 of the pushed frame), when they tapped a
        # card rather than centring it: a shop counter has several.
        self._aim_norm = None
        self._aim_at = None

    def select(self, aim_norm) -> None:
        """Start a new card at the point the user tapped (0..1 of the frame)."""
        self.reset()
        x, y = (float(v) for v in aim_norm)
        self._aim_norm = (min(1.0, max(0.0, x)), min(1.0, max(0.0, y)))

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

    def _new_card(self) -> None:
        """The outline now on screen is a different card: its views must not
        be pooled with the last card's."""
        self.horizontal = RunningRatio()
        self.vertical = RunningRatio()
        self.measured = 0
        self.last_result = None
        self._measurements = []
        self._sprt = None
        self._last_info = None
        self._last_gate = None

    def _view_motion(self, img: np.ndarray):
        """How far the view moved since the last frame (tracking pixels), or
        None when that cannot be told. Phase correlation on a half-size grey
        frame: ~3 ms, and the phone's own movement is the one motion that
        moves everything in the frame at once."""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        g = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
        prev, self._last_gray = self._last_gray, g
        if prev is None or prev.shape != g.shape:
            return None
        win = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(prev, g, win)
        if response < 0.08 or not (np.isfinite(dx) and np.isfinite(dy)):
            return None
        return np.array([2.0 * dx, 2.0 * dy])

    def _acquire(self, img: np.ndarray, scale: float, reticle, now: float, motion=None):
        """Where the card is in this (tracking-size) frame: (quad, sigma_px),
        or (None, None) with ``_lost_reason`` set.

        The field frames of 2.17.0 showed the old loop's failure: the tracker
        only ever asked "is there an edge near where the card was?", and on a
        soft frame from a moving phone there always is. It held an outline in
        the same place on the screen while the card moved away, onto a
        different card, and onto a different table, and called that TRACKING.
        So every tracked outline is now judged against the frame as a card
        (the scene search's judge: card-shaped through the perspective, edges
        along its sides that stop at its corners, a face unlike what is
        around it), and the card is found afresh at least every REDETECT_S.
        A frame where neither finds a card is reported as no card."""
        from .edge_information import _quad_iou
        from .scene import SceneSearch

        built = []

        def scene():
            if not built:
                built.append(SceneSearch(img))
            return built[0]

        shape = img.shape[:2]
        prev = self._last_raw * scale if self._last_raw is not None else None
        if prev is not None and motion is not None:
            # A phone moves the card tens of pixels between two pushes a few
            # hundred ms apart; the edge search reaches 14. Start it where
            # the whole view says the card went.
            prev = prev + motion[None, :]
        if prev is not None:
            try:
                q, stats = track_quad(img, prev, search_px=self._search_px, return_stats=True)
                hit = scene().judge(q)
                if hit is None:
                    raise DetectionError("the tracked outline is no longer a card")
                if now - self._last_detect_at >= REDETECT_S:
                    self._last_detect_at = now
                    fresh = scene().card_at(tuple(q.mean(axis=0)), nearest=False)
                    if (fresh is not None and fresh.score > hit.score
                            and _quad_iou(fresh.quad, q, shape) < 0.75):
                        q2 = enforce_portrait(order_quad(fresh.quad))
                        return self._adopt(q2, float(fresh.residual_px), prev, shape, now)
                self._last_valid_at = now
                return q, stats.corner_sigma_px
            except DetectionError:
                pass
        # Find it afresh: where it was a moment ago, if it was, else under the
        # reticle (or where the user tapped).
        anchors = []
        if prev is not None and now - self._last_valid_at <= RECENT_S:
            anchors.append(tuple(prev.mean(axis=0)))
        anchors.append(tuple(reticle))
        err = None
        for a in anchors:
            try:
                q, _, resid, _, _ = locate_card(img, prefer_point=a, search=scene(),
                                                judged_only=True)
            except DetectionError as exc:
                err = exc
                continue
            self._last_detect_at = now
            return self._adopt(q, float(resid), prev, shape, now)
        self._last_raw = None
        self._last_quad = None
        self._quad_filter.reset()
        self._lost_reason = " ".join(str(err).split()) if err is not None else ""
        return None, None

    def _adopt(self, q, resid, prev, shape, now):
        from .edge_information import _quad_iou

        if prev is not None and _quad_iou(q, prev, shape) < 0.3:
            self._new_card()
        # no smoothing across two different outlines
        self._quad_filter.reset()
        self._last_valid_at = now
        # the tap has done its job: later re-finds follow the card
        self._aim_norm = None
        self._aim_at = None
        return q, resid

    @staticmethod
    def _focus_hint(message: str, card_frac: float) -> str:
        """A soft frame with the card filling most of the view is usually the
        camera failing to focus that close (four of the five counter_* field
        frames: the card half the frame wide and every edge soft, the phone
        held still and level). Holding steadier does not fix that; a little
        more distance does."""
        if not message.startswith("hold steadier") or card_frac < 0.45:
            return message
        # The 2.21.0 screenshots: the card 50-80% of the view wide and every
        # frame soft for minutes. "Lift it a little" was cut off on screen
        # and did not say how far; about a third of the view wide is
        # 12-15 cm on a phone's main camera, past its closest focus, and
        # still ~12 px/mm in the Freeze frame.
        return ("too close to focus -- lift the phone until the card is about "
                "a third of the screen wide")

    def _photo_hint(self, message: str, frame_px_per_mm: float) -> str:
        """'Too far away' is about the LIVE frame, which the phone sends at
        540 px across. At a shop that is almost always too coarse to measure
        live, while the full-resolution photo from Measure Card has 4x the
        pixels -- so say which of the two it is."""
        if not message.startswith("too far away") or self.source_scale <= 1.0:
            return message
        from .capture import MIN_PX_PER_MM

        photo = frame_px_per_mm * self.source_scale
        if photo >= MIN_PX_PER_MM:
            return (f"live view too coarse here ({frame_px_per_mm:.1f} px/mm) -- zoom in (+) "
                    f"from where you are, or hold still and tap Freeze (photo ~{photo:.1f} px/mm)")
        return (f"too far even for a photo (~{photo:.1f} px/mm, needs {MIN_PX_PER_MM:.1f}) -- "
                "move closer or zoom in")

    def push(self, frame: np.ndarray, now: Optional[float] = None,
             source_scale: Optional[float] = None, zoom: Optional[float] = None) -> ARStatus:
        """Feed one camera frame. Cheap unless the frame is worth measuring.

        ``source_scale`` is how many camera pixels each pushed pixel stands
        for (the phone downsizes its 4K preview to 540 across). It does not
        change the live measurement, which can only use what was pushed; it
        lets the guidance say what a full-resolution photo would have."""
        now = now if now is not None else time.time()
        self.seen += 1
        if source_scale is not None and 1.0 <= source_scale <= 16.0:
            self.source_scale = float(source_scale)
        if zoom is not None and 1.0 <= zoom <= 30.0:
            self.zoom = float(zoom)

        track_img, track_scale = _resize_long(frame, TRACK_LONG_SIDE)
        th, tw = track_img.shape[:2]
        frame_centre = (tw / 2.0, th / 2.0)
        if self._aim_norm is not None and self._aim_at is None:
            self._aim_at = now
        if self._aim_norm is not None and now - self._aim_at > AIM_TTL_S:
            self._aim_norm = None          # a tap is about the frame it was made on
        if self._aim_norm is not None:
            frame_centre = (self._aim_norm[0] * tw, self._aim_norm[1] * th)
        motion = self._view_motion(track_img)
        quad_small, achieved_px = self._acquire(track_img, track_scale, frame_centre, now, motion)
        if quad_small is None:
            msg = self._lost_reason
            return ARStatus(
                tracking=False,
                quad=None,
                guidance=(msg[:160],) if msg else ("point at a card, all four edges in frame",),
                measured_frames=self.measured,
                seen_frames=self.seen,
                ratio=self.worst_ratio,
                settled=self.settled,
                scale=self.calibration.current(now) if self.calibration else None,
            )

        # INFORMATION FLOOR ON THE TRACKING FRAME.
        #
        # The tracker's measurement noise is what the sides actually achieved
        # (line-fit residuals), bounded below by the Cramer-Rao floor of the
        # channel that carried them. That one number sets both the 1-euro
        # filter's smoothing (noisy corners -> lower cutoff) and the next
        # frame's normal-search radius (a few sigma plus motion headroom),
        # instead of fixed constants tuned for one kind of scene.
        info_small = None
        try:
            from .edge_information import snap_to_information

            quad_small, info_small, moved = snap_to_information(
                track_img, quad_small, samples=24
            )
            if moved:
                quad_small = enforce_portrait(order_quad(quad_small))
        except ValueError:
            info_small = None
        floor_px = info_small.worst_corner_sigma_px if info_small is not None else 0.0
        sigma_meas = max(achieved_px or 0.0, floor_px, 1.0 / math.sqrt(12.0))
        self._sigma_meas_px = sigma_meas
        self._quad_filter.min_cutoff = float(np.clip(0.6 / sigma_meas, 0.3, 3.0))
        self._search_px = float(np.clip(4.0 * sigma_meas + 3.0, 3.0, 14.0))

        smoothed_quad = self._quad_filter.filter(quad_small, timestamp=now)
        self._last_quad = smoothed_quad / track_scale
        # The tracker starts from what was MEASURED, not from the smoothed
        # overlay: the filter lags a moving phone, and a tracker fed its own
        # lagged output searches where the card was, finds something there,
        # and the lag becomes the track.
        self._last_raw = quad_small / track_scale
        # The gate must judge the resolution the MEASUREMENT will have, not the
        # tracker's. Tracking runs at 540 px where a card is ~5 px/mm, which is
        # below the usable floor -- gating on that rejects every frame while the
        # measurement at 1200 px would have been comfortably fine.
        full_px_per_mm = 0.5 * (
            np.linalg.norm(quad_small[1] - quad_small[0]) / STANDARD_CARD_W_MM
            + np.linalg.norm(quad_small[3] - quad_small[0]) / STANDARD_CARD_H_MM
        ) / track_scale
        # The measurement is taken on a crop around the card, so its resolution
        # is set by the CROP's long side, not the frame's. Gating on the frame
        # would refuse shots the crop measures comfortably.
        card_long_px = max(
            float(np.linalg.norm(quad_small[2] - quad_small[1])),
            float(np.linalg.norm(quad_small[1] - quad_small[0])),
        ) / track_scale
        crop_long_px = card_long_px * (1.0 + 2.0 * FRAME_MARGIN_FRAC)
        measure_scale = min(1.0, MEASURE_LONG_SIDE / max(crop_long_px, 1.0))
        quality = assess_frame(
            track_img, quad_small, px_per_mm=float(full_px_per_mm) * measure_scale
        )

        due = (now - self._last_measure) >= self.measure_interval_s
        info_full = None
        if quality.passed and due:
            # Judge the outline at MEASUREMENT resolution before spending a
            # measurement on it: a side with no resolvable edge, or a frame
            # whose floor already exceeds what the grade gate can use, is
            # refused with the change that would help most.
            small, small_scale = _resize_long(frame, MEASURE_LONG_SIDE)
            try:
                from .edge_information import quad_information

                info_full = quad_information(small, self._last_quad * small_scale)
            except ValueError:
                info_full = None
            self._last_info = info_full
            blocked = None
            if info_full is not None and not info_full.resolvable:
                blocked = (
                    f"{', '.join(info_full.unresolved_sides)} edge not visible "
                    "against the background",
                )
            elif info_full is not None and info_full.sigma_cr_pp > self.gate_config.max_sigma_pp:
                blocked = tuple(a.message for a in info_full.advice[:2]) or (
                    "frame carries too little edge information to measure",
                )
            if blocked:
                quality = FrameQuality(
                    sharpness=quality.sharpness,
                    glare_frac=quality.glare_frac,
                    clipped_frac=quality.clipped_frac,
                    dark_frac=quality.dark_frac,
                    px_per_mm=quality.px_per_mm,
                    tilt_deg=quality.tilt_deg,
                    passed=False,
                    guidance=blocked,
                )
        if quality.passed and due:
            self._last_measure = now
            # The tracker already knows where the card is, so the measurement
            # frame is a crop around it rather than the whole downscaled frame.
            # On a distant card that is the difference between measuring 200 px
            # of card and measuring 600.
            framed = frame_card_for_measure(
                frame, quad=self._last_quad, fov_deg=self.fov_deg,
                max_side=MEASURE_LONG_SIDE,
            )
            try:
                # Detection is re-run INSIDE the crop rather than reusing the
                # tracked outline: the tracker's quad is 1-euro filtered and
                # measured worse on every synthetic view tried (err up to 4.3 pp
                # against 0.7 for a fresh detection in the crop). It is the same
                # locate_card the tracker acquired with -- the contour detector
                # alone does not find a card on a light counter.
                aim = (tuple(np.asarray(framed.quad).mean(axis=0))
                       if framed.quad is not None else None)
                try:
                    q_in, _, resid_in, _, _ = locate_card(framed.image, prefer_point=aim)
                except DetectionError:
                    q_in, resid_in = framed.quad, framed.residual_px
                res = measure_centering(
                    framed.image,
                    slab=resolve_holder(self.holder),
                    capture=framed.capture,
                    keep_rectified=False,
                    card_quad=q_in,
                    quad_residual_px=float(resid_in or 0.0),
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
                self._last_gate = self._gate(res, info_full)
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

        grade_ceil = None
        bands_dict = None
        grade_est = None
        grade_conf = None
        if self.worst_ratio is not None:
            try:
                from .grading import grade_band, predict_overall_grade
                psa_band = grade_band(self.worst_ratio, "PSA", "front")
                grade_ceil = psa_band.best if psa_band.is_single else f"{psa_band.worst}–{psa_band.best}"
                bands_dict = {
                    g: (b.best if b.is_single else f"{b.worst}–{b.best}")
                    for g, b in {
                        name: grade_band(self.worst_ratio, name, "front")
                        for name in ("PSA", "BGS", "CGC")
                    }.items()
                }
                # grade_ceil above is the honest worst-case range: it can only
                # narrow as more views accumulate and stays wide (e.g. "7-10")
                # on early frames by design, which reads as worthless on its
                # own. predict_overall_grade already exists for the still-photo
                # path and turns the same ratio into a single most-likely
                # grade plus a probability, using edge/corner quality signal
                # this session already measured -- wire it into the live loop
                # too instead of showing only the conservative range.
                quality_hint = self.last_result.quality if self.last_result else None
                pred = predict_overall_grade(
                    self.worst_ratio, quality=quality_hint, grader="PSA", face="front"
                )
                grade_est = pred.grade_label
                grade_conf = float(pred.confidence)
            except Exception:
                pass

        info_dict = None
        shown = self._last_info
        if shown is None and info_small is not None:
            shown = info_small.scaled(1.0 / track_scale)
        if shown is not None:
            info_dict = shown.to_dict()
            info_dict["tracking_sigma_px"] = round(sigma_meas / track_scale, 3)
            info_dict["search_px"] = round(self._search_px, 2)
        # how much of the frame's width the card takes: past about half, a
        # phone's main camera is near the closest it can focus
        xs = quad_small[:, 0]
        card_frac = float(xs.max() - xs.min()) / max(float(tw), 1.0)
        guidance = tuple(self._focus_hint(self._photo_hint(g, full_px_per_mm),
                                          card_frac / max(self.zoom, 1.0))
                         for g in quality.guidance)
        if quality.passed and shown is not None and shown.advice and not self.settled:
            guidance = tuple(guidance) + (shown.advice[0].message,)
        gate = self._last_gate

        return ARStatus(
            tracking=True,
            quad=self._last_quad,
            guidance=guidance,
            measured_frames=self.measured,
            seen_frames=self.seen,
            ratio=self.worst_ratio,
            settled=self.settled,
            scale=self.calibration.current(now) if self.calibration else None,
            grade_ceiling=grade_ceil,
            bands=bands_dict,
            grade_estimate=grade_est,
            grade_confidence=grade_conf,
            information=info_dict,
            decision=gate.decision.value if gate is not None else None,
            decision_reason=gate.reason if gate is not None else "",
            px_per_mm=float(full_px_per_mm),
            card_frac=card_frac,
        )

    def _gate(self, res: CenteringResult, info) -> Optional[object]:
        """confidence.gate on the fused worst-axis ratio, with the Cramer-Rao
        floor from the measured channel and the frame's shot-noise ratio."""
        from .confidence import gate
        from .information import cramer_rao_ratio_pp

        fused = self.worst_ratio
        if fused is None:
            return None
        pair = res.worst_axis
        if res.channel is not None:
            sigma_cr = cramer_rao_ratio_pp(
                res.channel, pair.low_mm.value, pair.high_mm.value, res.px_per_mm
            )
        elif info is not None:
            sigma_cr = info.sigma_cr_pp
        else:
            return None
        shot = info.shot_ratio if info is not None else float("inf")
        return gate(
            fused.value, self.boundary, sigma_cr, shot,
            effective_rows=float(self.fusion.n_views), cfg=self.gate_config,
        )
