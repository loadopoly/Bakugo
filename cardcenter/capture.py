"""Live capture: deciding which frames are worth measuring, and combining them.

Shooting through a display case, most frames are unusable. Glass throws
specular highlights, you cannot control the store's lighting, you are handheld,
and the card sits at whatever angle the shop chose. The useful product is not a
measurement on every frame -- it is a gate that refuses bad frames and tells you
what to change, plus an estimate that tightens as good frames accumulate.

RESOLUTION REALITY CHECK
------------------------
At a display case you get roughly:

    leaning on the glass, 1x     ~8 px/mm    125 um per pixel
    leaning on the glass, 2.5x  ~18 px/mm     55 um per pixel
    arm's length, 1x             ~4 px/mm    250 um per pixel

Edge whitening and corner wear are 50-200um features. At one to two pixels per
defect you cannot assess corners, edges, or surface, and no amount of software
changes that. Centering survives because it is a millimetre-scale geometric
measurement across the whole card, not a micrometre-scale texture judgement.

So live capture reports a centering CEILING and nothing else. It can rule a card
out. It can never rule a card in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .types import Measured

# Below this, the rectified card has too few pixels per millimetre for the
# border transition to be located to a useful fraction of a millimetre.
MIN_PX_PER_MM = 4.5
GOOD_PX_PER_MM = 9.0


@dataclass(frozen=True)
class FrameQuality:
    """Per-frame verdict, with guidance the user can act on immediately."""

    sharpness: float
    glare_frac: float
    clipped_frac: float
    dark_frac: float
    px_per_mm: float
    tilt_deg: Optional[float]
    passed: bool
    guidance: tuple[str, ...]

    def describe(self) -> str:
        state = "USE" if self.passed else "SKIP"
        return f"[{state}] " + ("; ".join(self.guidance) if self.guidance else "frame is good")


def assess_frame(
    image: np.ndarray,
    card_quad: Optional[np.ndarray] = None,
    px_per_mm: Optional[float] = None,
    tilt_deg: Optional[float] = None,
    sharpness_floor: float = 55.0,
    glare_ceiling: float = 0.030,
) -> FrameQuality:
    """Judge whether a frame is worth measuring.

    If ``card_quad`` is given, all statistics are computed inside the card only,
    which matters because a bright shop ceiling outside the card should not
    condemn an otherwise clean frame.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    mask = None
    if card_quad is not None:
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.asarray(card_quad, dtype=np.int32), 255)

    if mask is not None and mask.any():
        pixels = gray[mask > 0]
        x, y, w, h = cv2.boundingRect(np.asarray(card_quad, dtype=np.int32))
        x, y = max(0, x), max(0, y)
        region = gray[y : y + h, x : x + w]
    else:
        pixels = gray.ravel()
        region = gray

    # Sharpness: Laplacian variance. Sensitive to focus and motion blur alike,
    # which is what we want -- both make the border transition unmeasurable.
    sharpness = float(cv2.Laplacian(region, cv2.CV_64F).var()) if region.size else 0.0

    # Glare: specular highlights from the glass. Near-saturated AND locally flat
    # (a blown highlight has no texture), which distinguishes it from legitimate
    # bright white borders.
    if pixels.size:
        clipped_frac = float((pixels >= 250).mean())
        dark_frac = float((pixels <= 8).mean())
    else:
        clipped_frac = dark_frac = 0.0

    if region.size:
        bright = (region >= 244).astype(np.uint8)
        local_var = cv2.blur((region.astype(np.float32)) ** 2, (9, 9)) - (
            cv2.blur(region.astype(np.float32), (9, 9)) ** 2
        )
        flat = (local_var < 25.0).astype(np.uint8)
        glare_frac = float((bright & flat).mean())
    else:
        glare_frac = 0.0

    guidance: list[str] = []
    passed = True

    if sharpness < sharpness_floor:
        passed = False
        guidance.append("hold steadier or let it refocus -- frame is soft")
    if glare_frac > glare_ceiling:
        passed = False
        guidance.append(
            "glare on the glass -- shift your angle or shade the case with your body"
        )
    if clipped_frac > 0.12:
        passed = False
        guidance.append("highlights blown out -- reduce exposure")
    if dark_frac > 0.25:
        passed = False
        guidance.append("too dark -- get closer or raise exposure")
    if px_per_mm is not None:
        if px_per_mm < MIN_PX_PER_MM:
            passed = False
            guidance.append(
                f"too far away ({px_per_mm:.1f} px/mm) -- move closer or zoom in"
            )
        elif px_per_mm < GOOD_PX_PER_MM:
            guidance.append(
                f"usable but coarse ({px_per_mm:.1f} px/mm) -- closer would tighten the band"
            )
    if tilt_deg is not None and tilt_deg > 38.0:
        passed = False
        guidance.append(f"viewing angle too steep ({tilt_deg:.0f} deg) -- square up")

    return FrameQuality(
        sharpness=sharpness,
        glare_frac=glare_frac,
        clipped_frac=clipped_frac,
        dark_frac=dark_frac,
        px_per_mm=px_per_mm or 0.0,
        tilt_deg=tilt_deg,
        passed=passed,
        guidance=tuple(guidance),
    )


@dataclass
class RunningRatio:
    """Inverse-variance combination of repeated measurements of one card.

    Uses the Particle Data Group's scale-factor convention: if the individual
    measurements scatter by more than their own error bars allow, the combined
    uncertainty is inflated by sqrt(chi2/dof) rather than shrinking as 1/sqrt(N).
    Averaging inconsistent measurements into a tight number is how a tool ends up
    confidently wrong, and repeated frames of the same card through glass are
    exactly where that happens -- a glare streak or a misdetected edge produces a
    confident outlier, not a noisy one.
    """

    values: list[float] = field(default_factory=list)
    sigmas: list[float] = field(default_factory=list)

    def add(self, m: Measured) -> None:
        if m.sigma <= 0:
            return
        self.values.append(m.value)
        self.sigmas.append(m.sigma)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def combined(self) -> Optional[Measured]:
        if not self.values:
            return None
        v = np.asarray(self.values)
        s = np.asarray(self.sigmas)
        w = 1.0 / s**2
        mean = float((w * v).sum() / w.sum())
        sigma = float(math.sqrt(1.0 / w.sum()))

        if len(v) > 1:
            chi2 = float((w * (v - mean) ** 2).sum())
            dof = len(v) - 1
            scale = math.sqrt(max(1.0, chi2 / dof))
            sigma *= scale
        return Measured(mean, sigma)

    @property
    def consistency(self) -> Optional[float]:
        """chi2/dof. Above ~2 means the frames disagree with each other."""
        if len(self.values) < 2:
            return None
        v = np.asarray(self.values)
        s = np.asarray(self.sigmas)
        w = 1.0 / s**2
        mean = float((w * v).sum() / w.sum())
        return float((w * (v - mean) ** 2).sum() / (len(v) - 1))


@dataclass
class LiveSession:
    """Accumulates measurements of one card across frames."""

    horizontal: RunningRatio = field(default_factory=RunningRatio)
    vertical: RunningRatio = field(default_factory=RunningRatio)
    frames_seen: int = 0
    frames_used: int = 0
    last_guidance: tuple[str, ...] = ()

    def observe(self, result, quality: FrameQuality) -> None:
        self.frames_seen += 1
        self.last_guidance = quality.guidance
        if not quality.passed or result is None:
            return
        self.frames_used += 1
        self.horizontal.add(result.horizontal.ratio_pct)
        self.vertical.add(result.vertical.ratio_pct)

    @property
    def worst_ratio(self) -> Optional[Measured]:
        h = self.horizontal.combined
        v = self.vertical.combined
        candidates = [c for c in (h, v) if c is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.value)

    @property
    def settled(self) -> bool:
        """Enough consistent frames that more will not help much."""
        w = self.worst_ratio
        if w is None or self.frames_used < 4:
            return False
        c = max(
            [x for x in (self.horizontal.consistency, self.vertical.consistency) if x],
            default=0.0,
        )
        return w.sigma < 0.6 and c < 2.5

    def status(self) -> str:
        w = self.worst_ratio
        if w is None:
            hint = "; ".join(self.last_guidance) or "looking for a card"
            return f"no usable frames yet ({self.frames_seen} seen) -- {hint}"
        lo, hi = w.interval()
        line = (
            f"{w.value:.1f}/{100 - w.value:.1f}  (95% CI {lo:.1f}-{hi:.1f})  "
            f"from {self.frames_used}/{self.frames_seen} frames"
        )
        c = max(
            [x for x in (self.horizontal.consistency, self.vertical.consistency) if x],
            default=0.0,
        )
        if c > 2.5:
            line += f"  [frames disagree, chi2/dof={c:.1f} -- error bar inflated]"
        elif self.settled:
            line += "  [settled]"
        return line
