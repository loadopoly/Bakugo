"""Finding the printed border edge on a rectified card.

This is the part that fails. Outer-boundary detection is easy -- a card is a
high-contrast rectangle. Finding where the *printed* frame begins is not, and
any tool that claims to do it silently on arbitrary cards is overclaiming.

Approach: for each of the four sides, build a strip running inward from the
card edge and measure, per column, the depth at which the colour stops matching
the border. Two things make this robust:

  1. Work in CIELAB and threshold on deltaE, not RGB distance. Border-to-art
     transitions are often hue changes at similar luminance, which RGB
     Euclidean distance handles badly.
  2. Take a per-column transition and then a *median* across columns. Text,
     holo patches, set symbols and copyright lines all intrude into borders,
     but they rarely occupy more than half a side. A median survives them; a
     mean does not.

The spread of the per-column transitions is not thrown away -- it becomes the
uncertainty on that border width, and its slope becomes a print-rotation
warning. A card whose printing is rotated inside its cut has genuinely
ambiguous centering, and the user should be told rather than handed a number.

Cards with no border at all (full-bleed art) raise DetectionError. There is
nothing to measure and we say so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .types import DetectionError

SIDES = ("left", "top", "right", "bottom")


@dataclass
class SideProfile:
    side: str
    depth_mm: float
    sigma_mm: float
    confidence: float
    slope_mm_per_mm: float
    n_columns: int
    rejected_frac: float


def _oriented_strip(
    lab: np.ndarray, side: str, max_depth_px: int, edge_inset_px: int
) -> np.ndarray:
    """Return a strip with axis 0 = depth inward, axis 1 = position along edge.

    Every side is rotated into the same frame so downstream code is written
    once instead of four times with sign errors.
    """
    if side == "left":
        strip = lab[:, edge_inset_px : edge_inset_px + max_depth_px, :]
        return np.transpose(strip, (1, 0, 2))
    if side == "right":
        strip = lab[:, lab.shape[1] - edge_inset_px - max_depth_px : lab.shape[1] - edge_inset_px, :]
        return np.transpose(strip[:, ::-1, :], (1, 0, 2))
    if side == "top":
        return lab[edge_inset_px : edge_inset_px + max_depth_px, :, :]
    if side == "bottom":
        return lab[lab.shape[0] - edge_inset_px - max_depth_px : lab.shape[0] - edge_inset_px, :, :][::-1, :, :]
    raise ValueError(f"unknown side {side}")


def _column_transitions(
    dist: np.ndarray, thresh: float, min_run: int
) -> np.ndarray:
    """First sustained threshold crossing per column, with linear subpixel refinement.

    ``dist`` is (depth, position). Returns depth per column, NaN where no
    sustained crossing exists.
    """
    n_depth, n_pos = dist.shape
    above = dist > thresh

    # A crossing counts only if it stays above threshold for min_run samples.
    # Cumulative-sum trick gives the run length starting at each depth.
    csum = np.cumsum(above, axis=0)
    padded = np.vstack([np.zeros((1, n_pos)), csum])
    window_end = np.minimum(np.arange(1, n_depth + 1) + min_run, n_depth)
    sustained = np.zeros_like(above)
    for d in range(n_depth):
        end = int(window_end[d])
        run = padded[end] - padded[d]
        sustained[d] = run >= min(min_run, end - d)

    valid = above & sustained
    out = np.full(n_pos, np.nan)
    first_idx = np.argmax(valid, axis=0)
    has_any = valid.any(axis=0)

    for j in range(n_pos):
        if not has_any[j]:
            continue
        i = int(first_idx[j])
        if i == 0:
            out[j] = 0.0
            continue
        d0, d1 = dist[i - 1, j], dist[i, j]
        if abs(d1 - d0) < 1e-9:
            out[j] = float(i)
        else:
            frac = (thresh - d0) / (d1 - d0)
            out[j] = (i - 1) + min(1.0, max(0.0, float(frac)))
    return out


def _profile_pass(
    strip: np.ndarray, ref_depth_px: int, px_per_mm: float
) -> tuple[np.ndarray, float, np.ndarray, float, float]:
    """One detection pass. Returns (dist, thresh, transitions, signal, noise)."""
    ref_depth_px = max(2, min(ref_depth_px, strip.shape[0] - 2))
    ref = np.median(strip[:ref_depth_px].reshape(-1, 3), axis=0)
    dist = np.sqrt(((strip - ref[None, None, :]) ** 2).sum(axis=2))  # deltaE76

    # Otsu over the deltaE field separates "still border" from "no longer
    # border" without hand-tuned constants.
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        raise DetectionError("empty colour-distance field")
    ref_block = dist[:ref_depth_px]
    noise = float(np.median(np.abs(ref_block - np.median(ref_block))))
    interior = dist[int(strip.shape[0] * 0.6) :]
    signal = float(np.median(interior)) if interior.size else 0.0

    dmax = max(float(finite.max()), 1e-6)
    scaled = np.clip(finite / dmax * 255.0, 0, 255).astype(np.uint8)
    otsu_norm, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = max(float(otsu_norm) / 255.0 * dmax, 6.0)  # deltaE < ~6 is not real

    # Otsu splits the deltaE field on its own histogram, which assumes the two
    # classes are comparably populated. In a border strip they are not: most of
    # the scan depth is artwork, so Otsu lands near the artwork mode rather than
    # between border and artwork. Measured on a real bordered card at 4.7 px/mm:
    # signal 14-27 but Otsu threshold 31-38, so no column ever crossed and all
    # four sides silently failed. Across 75 real cards this produced 0/4 sides
    # detectable on 85% of them.
    #
    # The transition is what we want, so anchor the threshold to the measured
    # step instead: halfway between the border reference and the interior. Otsu
    # is kept as an upper bound because it is the better estimate when the two
    # classes ARE balanced.
    ref_block = dist[:ref_depth_px]
    border_level = float(np.median(ref_block))
    midpoint = border_level + 0.5 * (signal - border_level)
    thresh = max(6.0, min(thresh, midpoint))

    min_run = max(2, int(round(0.6 * px_per_mm)))
    trans = _column_transitions(dist, thresh, min_run)
    return dist, thresh, trans, signal, noise


def detect_side_border(
    rect_bgr: np.ndarray,
    side: str,
    px_per_mm: float,
    max_depth_mm: float = 22.0,
    edge_inset_mm: float = 0.35,
    end_trim_frac: float = 0.18,
) -> SideProfile:
    """Measure one border width, in mm, from the rectified card."""
    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    # OpenCV packs 8-bit Lab as L in 0..255 and a,b offset by 128.
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1] -= 128.0
    lab[..., 2] -= 128.0

    max_depth_px = int(round(max_depth_mm * px_per_mm))
    inset_px = int(round(edge_inset_mm * px_per_mm))
    strip = _oriented_strip(lab, side, max_depth_px, inset_px)
    if strip.shape[0] < 8 or strip.shape[1] < 16:
        raise DetectionError(f"side '{side}': rectified strip too small to measure")

    # Trim the ends: corners are where all four borders overlap and where
    # rounded die-cut radius distorts everything.
    n_pos = strip.shape[1]
    t0 = int(n_pos * end_trim_frac)
    t1 = n_pos - t0
    strip = strip[:, t0:t1, :]

    # Smooth lightly along position to suppress print dithering / JPEG noise.
    k = max(1, int(round(0.4 * px_per_mm)) | 1)
    strip = cv2.GaussianBlur(strip, (k, 1), 0)

    # PASS 1. The reference colour must be sampled from inside the border, but
    # we do not yet know how wide the border is. Start with a very shallow band.
    # 0.35mm is narrower than any printed border worth measuring, including the
    # ~1mm slivers on badly cut cards. Sampling deeper is tempting -- a wider
    # reference is a better colour estimate -- but on a narrow border it reaches
    # into the artwork, which drags the reference toward the art, weakens the
    # deltaE at the true transition, and pushes the detected edge DEEPER. The
    # bias runs one way: narrow borders get over-measured, which makes an
    # off-centre card look better centred than it is. Pass 2 widens the
    # reference once the border width is provisionally known.
    ref_px = max(3, int(round(0.35 * px_per_mm)))
    try:
        dist, thresh, trans, signal, noise = _profile_pass(strip, ref_px, px_per_mm)
    except DetectionError as exc:
        raise DetectionError(f"side '{side}': {exc}") from exc

    # PASS 2. Now that we have a provisional border width, re-sample the
    # reference over most of it. A wider reference is a better colour estimate
    # and a better noise estimate, which tightens the threshold and the
    # confidence score -- but only if it stays inside the border.
    provisional = np.nanmedian(trans) if np.isfinite(trans).any() else np.nan
    if np.isfinite(provisional) and provisional > 2.0 * ref_px:
        refined_px = int(max(ref_px, 0.6 * provisional))
        try:
            dist2, thresh2, trans2, signal2, noise2 = _profile_pass(
                strip, refined_px, px_per_mm
            )
            if np.isfinite(trans2).mean() >= np.isfinite(trans).mean() * 0.9:
                dist, thresh, trans, signal, noise = (
                    dist2,
                    thresh2,
                    trans2,
                    signal2,
                    noise2,
                )
        except DetectionError:
            pass  # keep pass-1 result

    border_noise = noise
    if signal < max(8.0, 4.0 * border_noise):
        raise DetectionError(
            f"side '{side}': no measurable border. The card may be full-bleed "
            f"(no printed frame), or the border and artwork are too similar in "
            f"colour (deltaE signal {signal:.1f} vs noise {border_noise:.1f})."
        )

    good = np.isfinite(trans)
    if good.sum() < 0.4 * len(trans):
        raise DetectionError(
            f"side '{side}': border edge found on only {100 * good.mean():.0f}% of "
            "the side; detection is not trustworthy"
        )

    vals = trans[good]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    robust_sigma = 1.4826 * mad

    # Reject columns more than 3 robust sigma out (logos, holo bleed, notches).
    tol = max(3.0 * robust_sigma, 0.5 * px_per_mm)
    inliers = np.abs(vals - med) <= tol
    rejected_frac = 1.0 - float(inliers.mean())
    kept = vals[inliers]
    if len(kept) < 8:
        raise DetectionError(f"side '{side}': too few consistent columns after outlier rejection")

    depth_px = float(np.median(kept))
    # Standard error of the median, inflated: adjacent columns are correlated
    # by the smoothing kernel, so the effective sample count is lower than the
    # raw column count. Divide by the kernel width to be conservative.
    n_eff = max(2.0, len(kept) / max(1.0, k))
    sigma_px = 1.2533 * (1.4826 * float(np.median(np.abs(kept - depth_px)))) / math.sqrt(n_eff)
    sigma_px = max(sigma_px, 0.25)  # floor: no subpixel claim below a quarter pixel

    positions = np.arange(len(trans))[good][inliers].astype(np.float64)
    if len(positions) >= 8 and float(np.ptp(positions)) > 1e-6:
        slope_px_per_px = float(np.polyfit(positions, kept, 1)[0])
    else:
        slope_px_per_px = 0.0

    confidence = float(np.clip(signal / (signal + 6.0 * border_noise + 4.0), 0.0, 1.0))
    confidence *= float(np.clip(1.0 - rejected_frac, 0.0, 1.0))
    confidence *= float(np.clip(1.0 - (robust_sigma / max(px_per_mm, 1e-6)) / 1.5, 0.0, 1.0))

    depth_mm_total = (depth_px / px_per_mm) + (inset_px / px_per_mm)
    # Very narrow borders sit near the limit of what the scan geometry can
    # resolve: the inset and the reference band together occupy a large
    # fraction of them, so residual bias is possible even after pass 2. Widen
    # the error bar rather than pretend the number is as good as a wide border.
    if depth_mm_total < 1.5:
        sigma_px = math.hypot(sigma_px, 0.18 * px_per_mm)
        confidence *= 0.8

    return SideProfile(
        side=side,
        depth_mm=depth_mm_total,
        sigma_mm=sigma_px / px_per_mm,
        confidence=confidence,
        slope_mm_per_mm=slope_px_per_px,
        n_columns=int(len(kept)),
        rejected_frac=rejected_frac,
    )


def detect_all_borders(rect_bgr: np.ndarray, px_per_mm: float) -> dict[str, SideProfile]:
    profiles: dict[str, SideProfile] = {}
    errors: list[str] = []
    for side in SIDES:
        try:
            profiles[side] = detect_side_border(rect_bgr, side, px_per_mm)
        except DetectionError as exc:
            errors.append(str(exc))
    if errors:
        raise DetectionError(
            "border detection failed:\n  - " + "\n  - ".join(errors)
        )
    return profiles
