"""Quad handling and rectification to physical card coordinates.

The card's outer boundary is found as a contour, but a contour's own vertices
are quantised to whole pixels and sit wherever the approximation algorithm
happened to break the polygon. We do better: fit a straight line to each of the
four sides using all the contour points along that side, then intersect
adjacent lines. That gives subpixel corners *and* a residual we can turn into
an honest uncertainty, because a card edge really is straight -- any residual
is measurement noise or a bent card, and both are things the user should know.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from .types import STANDARD_CARD_H_MM, STANDARD_CARD_W_MM, DetectionError


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points consistently: start top-left, proceed clockwise.

    Sorting by angle about the centroid is robust to rotation, unlike the
    common x+y / x-y trick which breaks past ~30 degrees of roll.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    idx = np.argsort(ang)
    ordered = pts[idx]

    # Enforce clockwise in image coords (y down => positive shoelace is CW).
    area = 0.0
    for i in range(4):
        x1, y1 = ordered[i]
        x2, y2 = ordered[(i + 1) % 4]
        area += x1 * y2 - x2 * y1
    if area < 0:
        ordered = ordered[::-1]

    # Rotate so index 0 is the top-left-most corner.
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def enforce_portrait(quad: np.ndarray) -> np.ndarray:
    """Rotate the ordering so edge 0->1 is the card's short (width) side."""
    e01 = np.linalg.norm(quad[1] - quad[0])
    e12 = np.linalg.norm(quad[2] - quad[1])
    if e01 > e12:
        return np.roll(quad, -1, axis=0)
    return quad


def fit_line_tls(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Total-least-squares line fit. Returns ((a,b,c) for ax+by+c=0, rms_residual)."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        raise DetectionError("not enough points to fit a card edge")
    mean = pts.mean(axis=0)
    centred = pts - mean
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    direction = Vt[0]
    normal = np.array([-direction[1], direction[0]])
    c = -float(normal @ mean)
    resid = centred @ normal
    return np.array([normal[0], normal[1], c]), float(np.sqrt(np.mean(resid**2)))


# Raw approxPolyDP corners are quantised to whole pixels, so a quad reported
# without a successful line refinement is known to about a pixel -- not the
# subpixel precision a good fit gives, and not the tens of pixels a bad one
# implies.
RAW_QUAD_RESIDUAL_PX = 1.5

# How far refinement may move a corner before we stop believing it, as a
# fraction of the quad's short side. Line-intersection refinement is a
# subpixel correction on a boundary the shape filter already found; moving a
# corner a fifth of the card's width means the line fit found a different
# object, not a better estimate of this one.
MAX_REFINE_SHIFT_FRAC = 0.20


def _refinement_is_sane(raw: np.ndarray, refined: np.ndarray) -> bool:
    """Did refinement improve this quad, or replace it with something else?"""
    e01 = float(np.linalg.norm(refined[1] - refined[0]))
    e12 = float(np.linalg.norm(refined[2] - refined[1]))
    if min(e01, e12) < 1e-6 or not np.all(np.isfinite(refined)):
        return False
    r01 = float(np.linalg.norm(raw[1] - raw[0]))
    r12 = float(np.linalg.norm(raw[2] - raw[1]))
    if min(r01, r12) < 1e-6:
        return False
    scale = min(r01, r12)
    shift = float(np.abs(refined - raw).max())
    return shift <= MAX_REFINE_SHIFT_FRAC * scale


def fit_line_robust(
    pts: np.ndarray, k_sigma: float = 2.5, max_iter: int = 3, min_keep: int = 8
) -> tuple[np.ndarray, float, float]:
    """TLS with iterative MAD-based outlier rejection.

    ``fit_line_tls`` is least-squares, which has a breakdown point of zero: one
    contaminated cluster moves the line arbitrarily far. That is fine on a clean
    catalogue scan and wrong on a handheld photo, because the points assigned to
    a card side are routinely a MIXTURE of surfaces:

      - the card's own cut edge, which is what we want;
      - the penny sleeve's edge, a near-parallel line 1-3 mm outside it;
      - sleeve crinkles and specular glare streaks;
      - a neighbouring card in the bin or binder page, whose edge is simply the
        closest thing to that side.

    Measured on 162 real sleeved-card photographs: plain TLS returned side
    residuals of 46-68 px on cards whose raw quad was already a correct 1.33
    aspect, and the distorted refinement then failed the aspect gate or moved
    the boundary far enough that border detection found nothing. Not one of the
    162 produced a measurement.

    A mixture is not noise, so the fix is rejection rather than a wider error
    bar. Fit, measure residuals, drop everything beyond ``k_sigma`` robust
    sigma, refit. The dominant surface wins because it contributes the most
    points, and on a card photographed against anything the strongest, longest
    straight run along a side is the card edge itself.

    Returns (line, inlier_rms, rejected_frac). The RMS is over the SURVIVING
    points and is an honest uncertainty for the edge that was actually fitted:
    the rejected points were a different object, not scatter about this one.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        raise DetectionError("not enough points to fit a card edge")

    keep = np.ones(len(pts), dtype=bool)
    line, rms = fit_line_tls(pts)

    # ROBUST SEED FIRST, THEN MAD REJECTION.
    #
    # MAD rejection can only separate clusters if the line it measures against
    # is already roughly right, and the plain fit often is not: contamination
    # that covers only PART of a side (a sleeve edge over half its length, a
    # neighbour touching one end) tilts the fit rather than offsetting it, so
    # residuals then vary smoothly along the side instead of splitting into two
    # groups, and MAD sees one broad distribution with nothing to reject.
    #
    # Trimming the worst-fitting fraction outright does not need the line to be
    # right, only better than chance -- the same trick track_quad already uses
    # to shrug off a glare streak. Use it to get a direction, then let MAD
    # decide membership. On a clean side the trimmed points are collinear with
    # the rest, so the seed is identical and nothing is ultimately discarded.
    for _ in range(2):
        signed = pts @ line[:2] + line[2]
        resid = np.abs(signed - float(np.median(signed)))
        sel = resid <= float(np.quantile(resid, 0.70))
        if sel.sum() < max(min_keep, 2):
            break
        line, rms = fit_line_tls(pts[sel])

    for _ in range(max_iter):
        normal = line[:2]
        signed = pts @ normal + line[2]
        med = float(np.median(signed[keep]))
        mad = float(np.median(np.abs(signed[keep] - med)))
        # Floor the scale rather than bail out when the MAD collapses. A MAD of
        # zero does not mean "no outliers to find" -- it means the dominant
        # surface is DEAD straight, which is exactly the case where a second
        # edge alongside it should be rejected most decisively. Bailing out
        # there left the fit sitting between the two edges, on neither. The
        # floor is sub-pixel, so on a genuinely clean side it is wider than the
        # scatter and nothing is discarded.
        sigma = max(1.4826 * mad, 0.5)
        new_keep = np.abs(signed - med) <= k_sigma * sigma
        if new_keep.sum() < max(min_keep, 2):
            break
        if new_keep.sum() == keep.sum() and bool(np.all(new_keep == keep)):
            break
        keep = new_keep
        line, rms = fit_line_tls(pts[keep])

    return line, float(rms), float(1.0 - keep.mean())


def intersect_lines(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-12:
        raise DetectionError("card edges are parallel; corner is undefined")
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.array([x, y])


def refine_quad(
    contour: np.ndarray, quad: np.ndarray, trim_frac: float = 0.15
) -> tuple[np.ndarray, float]:
    """Refine corners by fitting lines to each side of the contour.

    ``trim_frac`` drops points near the corners, where the physical card has a
    rounded die-cut radius that would bias a straight-line fit inward.
    """
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)

    # Assign each contour point to the side it is closest to. Vectorised over
    # all points against all four segments at once.
    dists = np.empty((4, len(pts)))
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-12:
            dists[i] = np.inf
            continue
        t = ((pts - a) @ ab) / L2
        t = np.clip(t, 0.0, 1.0)
        proj = a[None, :] + t[:, None] * ab[None, :]
        dists[i] = np.linalg.norm(pts - proj, axis=1)
    assignment = np.argmin(dists, axis=0)
    sides: list[np.ndarray] = [pts[assignment == i] for i in range(4)]

    lines, residuals = [], []
    for i in range(4):
        side_pts = sides[i]
        if len(side_pts) < 8:
            raise DetectionError(
                f"card side {i} has only {len(side_pts)} contour points; "
                "the outer boundary was not cleanly detected"
            )
        a, b = quad[i], quad[(i + 1) % 4]
        ab = b - a
        L2 = float(ab @ ab)
        t = ((side_pts - a) @ ab) / max(L2, 1e-12)
        keep = (t > trim_frac) & (t < 1.0 - trim_frac)
        chosen = side_pts[keep] if keep.sum() >= 8 else side_pts
        # Robust, not least-squares: a side's points are usually a mixture of
        # the card edge and whatever lies alongside it (sleeve edge, a
        # neighbouring card). See fit_line_robust.
        line, res, rejected = fit_line_robust(chosen)
        lines.append(line)
        # A side where most points were rejected is not a cleanly observed
        # edge, whatever the surviving points' RMS says. Charge that back so
        # the boundary uncertainty stays honest and the selection ladder can
        # still tell a crisp edge from a salvaged one.
        residuals.append(res * (1.0 + 2.0 * rejected))

    corners = np.array(
        [intersect_lines(lines[(i - 1) % 4], lines[i]) for i in range(4)]
    )
    return corners, float(np.mean(residuals))


def _solidify(cnt: np.ndarray, shape: tuple[int, int]) -> Optional[np.ndarray]:
    """Return a dense, solid outer boundary for a possibly hollow contour.

    A thresholded card often comes back as a *ring*: the printed border is above
    threshold and the darker artwork inside is not. Such a contour traces around
    the band rather than around the card, so its area is the band's area and its
    points lie on both the outer and inner edges. Fitting lines to that mixture
    biases every border inward.

    Filling the contour into a local mask and re-extracting the external
    boundary collapses it to the outer edge, densely sampled, whatever the
    interior looked like.
    """
    x, y, w, h = cv2.boundingRect(cnt.astype(np.int32))
    if w < 8 or h < 8:
        return None
    pad = 3
    mask = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.uint8)
    shifted = cnt.astype(np.int32).reshape(-1, 1, 2) - np.array(
        [[x - pad, y - pad]], dtype=np.int32
    )
    cv2.drawContours(mask, [shifted], -1, 255, -1)
    # A small kernel only. This exists to bridge one- or two-pixel breaks in a
    # ring, not to reshape the card: a large kernel rounds the corners and
    # displaces the straight edges, which shows up directly as line-fit residual
    # and therefore as inflated uncertainty on every border.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not found:
        return None
    big = max(found, key=cv2.contourArea)
    return big.reshape(-1, 2).astype(np.float64) + np.array([x - pad, y - pad])


def _gather_contours(image: np.ndarray) -> list[np.ndarray]:
    """Contours from several independent strategies.

    No single threshold works across a case full of differently coloured cards:
    a global Otsu split that isolates a yellow-bordered card merges a
    white-bordered one into the background. Running complementary strategies and
    letting the shape filter arbitrate is more robust than tuning one of them.
    RETR_LIST rather than RETR_EXTERNAL because a card inside a slab inside a
    display case is genuinely nested.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    blurred = cv2.bilateralFilter(gray, 9, 60, 60)
    out: list[np.ndarray] = []

    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binaries = [th, cv2.bitwise_not(th)]

    block = max(31, (min(gray.shape) // 12) | 1)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 5
    )
    binaries += [adaptive, cv2.bitwise_not(adaptive)]

    med = float(np.median(blurred))
    edges = cv2.Canny(blurred, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)))
    binaries.append(
        cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    )

    for b in binaries:
        cleaned = cv2.morphologyEx(
            b, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
        )
        cnts, _ = cv2.findContours(cleaned, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        out.extend(cnts)
    return out


def compute_edge_gradient(image: np.ndarray) -> np.ndarray:
    """Compute multi-channel chromatic edge gradient magnitude.

    Grayscale conversion (0.299R + 0.587G + 0.114B) loses edge contrast when
    yellow, silver, or white card borders rest on light wood, pine desks, or
    light quartz where the luminance difference is negligible. Computing the
    Sobel gradient per color channel and taking the element-wise maximum:
        mag = max(mag_B, mag_G, mag_R)
    preserves strong edge response across iso-luminant chromatic boundaries.
    """
    if image.ndim == 3 and image.shape[2] >= 3:
        mags = []
        for c in range(3):
            ch = image[..., c]
            gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
            mags.append(cv2.magnitude(gx, gy))
        return np.maximum.reduce(mags)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)


def edge_support(gradient: np.ndarray, quad: np.ndarray, samples: int = 240) -> float:
    """Fraction of the quad's perimeter that sits on a real intensity edge.

    Returns 0..1 and is directly interpretable: 0.9 means nearly the whole
    outline lies on a step in the image, 0.2 means it mostly floats in flat
    pixels.

    This is the defence against *halo* detections. A large-block adaptive
    threshold produces a contour tens of pixels outside the true card edge,
    sitting in blank background. A halo is LARGER than the card it surrounds, so
    area-based selection prefers it, and because it adds the same margin to both
    sides it makes an off-centre card measure as perfectly centred -- with a
    tight error bar, because nothing else about the measurement is wrong. Silent
    and confident is the worst failure mode a measurement tool can have, so this
    check is a hard gate rather than a soft score.
    """
    h, w = gradient.shape[:2]

    # THRESHOLD MUST BE LOCAL, NOT GLOBAL.
    #
    # A global percentile of the gradient is set by whatever is BUSIEST in the
    # frame, and on a trading card that is the card's own interior -- text,
    # artwork, holo foil. Measured on a clean catalogue scan of a plainly
    # bordered card: interior gradients reach 923 and p99.5 is 564, putting the
    # global threshold at 169, while the true card boundary has a median
    # gradient of 219. The real edge barely cleared a bar set by the art it
    # surrounds, so edge support came out 0.07-0.27 against a 0.55 requirement
    # and every genuinely bordered card was rejected.
    #
    # The question is not "is this edge strong compared to the whole image" but
    # "is this a step compared to its own immediate surroundings". Comparing each
    # sample against a local neighbourhood answers that and is invariant to how
    # busy the rest of the card is.
    per_side = max(8, samples // 4)
    hits = 0
    count = 0
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        ts = np.linspace(0.06, 0.94, per_side)
        pts = a[None, :] + ts[:, None] * (b - a)[None, :]
        edge = b - a
        n = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm

        # Sample a profile across the candidate edge. A real boundary peaks near
        # the centre of that profile and falls away on both sides.
        #
        # The search must be WIDE. approxPolyDP after morphological closing
        # returns a polygon whose corners sit a few pixels off the true edge and
        # rotated by around a degree, so a narrow perpendicular search walks
        # ALONG the boundary instead of across it and finds no step at all.
        # Measured on a clean catalogue scan: the true left edge has gradient
        # 769 against flanks of 2 -- a ratio of 504 -- yet edge support scored
        # 0.15 because the samples never landed on it.
        offsets = np.arange(-14.0, 14.01, 1.0)
        prof = np.zeros((len(pts), len(offsets)))
        for k, off in enumerate(offsets):
            q = pts + off * n[None, :]
            xi = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
            yi = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
            prof[:, k] = gradient[yi, xi]

        # Peak anywhere in the search window, against the window's own quiet
        # tails. This tolerates a few pixels of polygon offset while still
        # requiring a genuine step rather than ambient texture.
        centre = prof.max(axis=1)
        flank = np.concatenate([prof[:, :4], prof[:, -4:]], axis=1).mean(axis=1)
        # A step edge stands proud of its own flanks; interior texture does not.
        hits += int(np.sum(centre > np.maximum(flank * 2.0, 15.0)))
        count += len(pts)

    return hits / count if count else 0.0


def snap_quad_to_edges(
    gradient: np.ndarray,
    quad: np.ndarray,
    inward_frac: float = 0.16,
    outward_frac: float = 0.03,
    samples: int = 28,
) -> tuple[np.ndarray, float]:
    """Pull each side of an approximate quad onto the strongest nearby edge.

    Thresholding finds a card by contrast against its surroundings, which works
    on a desk and degrades badly in the setting this tool is actually used in: a
    card held over a bulk bin, a binder page, a display tray. There the
    surroundings ARE other cards, at similar brightness, so the traced contour
    routinely merges the subject with a neighbour and the resulting quad is a
    HALO -- correct in shape and aspect, but larger than the card.

    A halo is the dangerous kind of wrong. It adds roughly the same margin to
    opposite sides, so it pulls any measured ratio toward 50/50 and makes an
    off-centre card look well centred, with a tight error bar because nothing
    else about the measurement is inconsistent. ``edge_support`` is the existing
    defence and it cannot see this case: a halo boundary in a bin of cards still
    lies along real intensity steps, they are just the wrong cards' edges.

    So rather than judge the boundary, correct it. Sample along each side, walk
    the normal, and take the position of peak gradient. That the peak is the CUT
    edge rather than the printed frame a few millimetres inside it is not an
    assumption: a cut edge is a luminance step against the background, while the
    border-to-artwork transition is usually a hue change at similar luminance
    (the reason ``detect`` works in CIELAB and not in grey). The search is
    deliberately asymmetric -- far inward, barely outward -- because the failure
    being corrected is always an overshoot.

    Returns (snapped_quad, mean_inlier_residual_px). The caller decides whether
    to keep it; this function does not assume it improved anything.
    """
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    h, w = gradient.shape[:2]
    short = min(
        float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)
    )
    if short < 24.0:
        raise DetectionError("quad too small to snap")
    inward = max(3.0, inward_frac * short)
    outward = max(2.0, outward_frac * short)
    offsets = np.arange(-outward, inward + 1e-9, 1.0)

    lines, residuals = [], []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        edge = b - a
        L = float(np.linalg.norm(edge))
        if L < 8.0:
            raise DetectionError("degenerate side while snapping")
        n = np.array([-edge[1], edge[0]]) / L
        # Point the normal INWARD so positive offsets move toward the centre.
        if float(n @ (0.5 * (a + b) - q.mean(axis=0))) > 0:
            n = -n
        ts = np.linspace(0.10, 0.90, samples)
        base = a[None, :] + ts[:, None] * edge[None, :]

        hits = []
        for pt in base:
            cand = pt[None, :] + offsets[:, None] * n[None, :]
            xi = np.round(cand[:, 0]).astype(int)
            yi = np.round(cand[:, 1]).astype(int)
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            if ok.sum() < 6:
                continue
            vals = np.where(
                ok, gradient[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)], -1.0
            )
            k = int(np.argmax(vals))
            if vals[k] <= 0:
                continue
            hits.append(pt + offsets[k] * n)

        if len(hits) < 8:
            raise DetectionError(f"side {i}: too few edge hits while snapping")
        line, res, rejected = fit_line_robust(np.array(hits))
        lines.append(line)
        residuals.append(res * (1.0 + 2.0 * rejected))

    corners = np.array(
        [intersect_lines(lines[(i - 1) % 4], lines[i]) for i in range(4)]
    )
    if not np.all(np.isfinite(corners)):
        raise DetectionError("snapping produced a degenerate corner")
    return corners, float(np.mean(residuals))


def touches_frame_boundary(
    quad: np.ndarray, h: int, w: int, margin: Optional[float] = None
) -> bool:
    """Return True if the quad touches, spans, or runs along the sensor/image boundary.

    A collectible card being measured must have all four edges visible in frame.
    An edge coincident with the image boundary indicates a clipped card or a
    viewport/container boundary artifact.
    """
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    m = margin if margin is not None else max(4.0, 0.008 * min(h, w))
    min_x, max_x = float(q[:, 0].min()), float(q[:, 0].max())
    min_y, max_y = float(q[:, 1].min()), float(q[:, 1].max())

    # Spans essentially the full height or full width of the image
    if min_y <= m and max_y >= (h - 1) - m:
        return True
    if min_x <= m and max_x >= (w - 1) - m:
        return True

    # Check if any side of the quad lies along an image boundary
    for i in range(4):
        p1, p2 = q[i], q[(i + 1) % 4]
        # Top border
        if abs(p1[1]) <= m and abs(p2[1]) <= m:
            return True
        # Bottom border
        if abs(p1[1] - (h - 1)) <= m and abs(p2[1] - (h - 1)) <= m:
            return True
        # Left border
        if abs(p1[0]) <= m and abs(p2[0]) <= m:
            return True
        # Right border
        if abs(p1[0] - (w - 1)) <= m and abs(p2[0] - (w - 1)) <= m:
            return True

    return False


def detect_active_viewport(
    image: np.ndarray,
    black_threshold: float = 16.0,
    min_consecutive: int = 10,
    min_span_frac: float = 0.20,
) -> tuple[int, int, int, int]:
    """Detect (x, y, w, h) of the active camera viewport, excluding digital
    pillarbox (black bars on left/right) or letterbox (black bars on top/bottom)
    introduced by webcam drivers or letterboxed video streams.

    Returns (0, 0, w, h) if no digital padding is present.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    h, w = gray.shape[:2]

    # Sample middle 60% of rows to avoid any corner UI overlays/badges
    r_start, r_end = int(0.2 * h), max(int(0.2 * h) + 1, int(0.8 * h))
    mid_rows = gray[r_start:r_end, :]
    col_means = mid_rows.mean(axis=0)
    col_active = col_means >= black_threshold

    # Find active column boundaries with sustained content
    x1 = 0
    for i in range(w - min_consecutive):
        if np.all(col_active[i : i + min_consecutive]):
            x1 = i
            break

    x2 = w - 1
    for i in range(w - 1, min_consecutive - 1, -1):
        if np.all(col_active[i - min_consecutive + 1 : i + 1]):
            x2 = i
            break

    # Sample middle 60% of columns between x1 and x2
    span_w = max(1, x2 - x1)
    c_start = x1 + int(0.2 * span_w)
    c_end = max(c_start + 1, x1 + int(0.8 * span_w))
    mid_cols = gray[:, c_start:c_end]
    row_means = mid_cols.mean(axis=1)
    row_active = row_means >= black_threshold

    y1 = 0
    for i in range(h - min_consecutive):
        if np.all(row_active[i : i + min_consecutive]):
            y1 = i
            break

    y2 = h - 1
    for i in range(h - 1, min_consecutive - 1, -1):
        if np.all(row_active[i - min_consecutive + 1 : i + 1]):
            y2 = i
            break

    pad_w = x1 + (w - 1 - x2)
    pad_h = y1 + (h - 1 - y2)
    # Only treat as digital padding if padding is significant (> 4% of dimension)
    if pad_w < 0.04 * w:
        x1, x2 = 0, w - 1
    if pad_h < 0.04 * h:
        y1, y2 = 0, h - 1

    active_w = x2 - x1 + 1
    active_h = y2 - y1 + 1
    if active_w < min_span_frac * w or active_h < min_span_frac * h:
        return 0, 0, w, h

    return x1, y1, active_w, active_h


def quad_candidates(
    image: np.ndarray, min_area_frac: float = 0.03, min_edge_support: float = 0.55
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """All card-shaped quadrilaterals in the image.

    Returns (quad_area, ordered_quad, dense_boundary_points), unsorted. Area and
    aspect are judged on the *quad*, not on the raw traced contour, so a hollow
    ring is measured by the card it outlines rather than by the width of its own
    border.
    """
    h, w = image.shape[:2]
    img_area = float(h * w)
    expected = STANDARD_CARD_H_MM / STANDARD_CARD_W_MM

    gradient = compute_edge_gradient(image)

    found: list[tuple[float, np.ndarray, np.ndarray]] = []
    seen_boxes: list[tuple[int, int, int, int]] = []
    for cnt in _gather_contours(image):
        if len(cnt) < 32:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw < 8 or bh < 8:
            continue
        # The bounding box already bounds the quad, so anything whose box is too
        # small or too elongated cannot be a card. Rejecting here avoids running
        # the expensive solidify step on thousands of text and texture contours.
        if bw * bh < 0.55 * min_area_frac * img_area:
            continue
        box_aspect = max(bw, bh) / max(1, min(bw, bh))
        if box_aspect > 3.2:
            continue
        rounded = (bx // 8, by // 8, bw // 8, bh // 8)
        if rounded in seen_boxes:
            continue
        seen_boxes.append(rounded)

        solid = _solidify(cnt, (h, w))
        if solid is None or len(solid) < 32:
            continue

        pts = solid.reshape(-1, 1, 2).astype(np.float32)
        peri = cv2.arcLength(pts, True)
        approx = None
        for eps in (0.01, 0.02, 0.03, 0.045):
            a = cv2.approxPolyDP(pts, eps * peri, True)
            if len(a) == 4:
                approx = a
                break
        if approx is None:
            approx = cv2.boxPoints(cv2.minAreaRect(pts)).reshape(4, 1, 2)

        quad = order_quad(np.asarray(approx, dtype=np.float64).reshape(4, 2))
        quad_area = abs(cv2.contourArea(quad.astype(np.float32)))
        if quad_area < min_area_frac * img_area or quad_area > 0.995 * img_area:
            continue

        # A card being measured cannot touch or run along the sensor borders.
        if touches_frame_boundary(quad, h, w):
            continue

        # The quad must actually describe the shape, not just bound it.
        if cv2.contourArea(cv2.convexHull(pts)) < 0.80 * quad_area:
            continue

        e01 = np.linalg.norm(quad[1] - quad[0])
        e12 = np.linalg.norm(quad[2] - quad[1])
        if min(e01, e12) < 1e-6:
            continue
        aspect = max(e01, e12) / min(e01, e12)
        # A card face is 1.400. Perspective foreshortening stretches or compresses
        # this under handheld tilts (e.g. 1.40 * cos(45 deg) approx 0.99).
        # Expanding the aspect window to 0.92 < aspect < 1.88 admits natural tilted
        # viewing angles up to ~48 deg while still reliably excluding elongated objects.
        if not (0.92 < aspect < 1.88):
            continue

        if edge_support(gradient, quad) < min_edge_support:
            continue

        found.append((quad_area, quad, solid))
    return found


def find_card_quad(
    image: np.ndarray,
    min_area_frac: float = 0.008,
    prefer_point: Optional[tuple[float, float]] = None,
    check_viewport: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Locate a card in the image.

    ``min_area_frac`` defaulted to 0.03 for synthetic renders, where the card
    fills the frame by construction. On a real capture session the median card
    occupies 3.6% of the frame and many good shots are below 3% -- so the
    default was rejecting cards the detector had already found. Measured on 200
    real photographs: at 0.03 the pipeline measured 0 of 60; the candidate stage
    was finding valid 1.44-aspect quads and the area gate was discarding them.

    ``prefer_point``, if given, breaks ties among the quality-gated survivors by
    distance to that point instead of by area. A shop table is rarely one card:
    it is a spread, a display case, or a box lid, and "biggest card-shaped thing
    anywhere in the frame" silently measures whichever neighbour happens to be
    larger rather than the one the user framed. AR callers pass the reticle (the
    frame centre) on first acquisition and the last known position on
    re-acquisition after a lost track, so the tool measures what it was already
    looking at, not whatever else is on the table. Ungated candidates are never
    promoted by this -- it only orders within the survivors of the existing
    residual/edge-support ladder.

    Returns (refined_corners, contour, mean_line_residual_px).
    """
    if check_viewport:
        vx, vy, vw, vh = detect_active_viewport(image)
        h, w = image.shape[:2]
        if vw < w or vh < h:
            crop = image[vy : vy + vh, vx : vx + vw]
            pt = (
                (prefer_point[0] - vx, prefer_point[1] - vy)
                if prefer_point is not None
                else None
            )
            refined, contour, res = find_card_quad(
                crop,
                min_area_frac=min_area_frac,
                prefer_point=pt,
                check_viewport=False,
            )
            refined = refined.copy()
            refined[:, 0] += vx
            refined[:, 1] += vy
            contour = contour.copy()
            contour[..., 0] += vx
            contour[..., 1] += vy
            return refined, contour, res

    found = quad_candidates(image, min_area_frac=min_area_frac)
    if not found:
        raise DetectionError(
            "could not locate a card-shaped quadrilateral. Shoot the card "
            "against a plain contrasting background with all four edges visible."
        )

    gradient_full = compute_edge_gradient(image)

    # Largest-wins is not enough: a slightly larger candidate that fits its own
    # edges badly is a blob or a halo, not a card, and picking it silently
    # displaces every border measurement. But a product of exponential penalties
    # is worse -- it lets a small, crisp inner rectangle outscore the real card.
    # Use explicit gates instead, then take the largest survivor, relaxing the
    # gates only if nothing passes.
    #
    # When a caller supplies prefer_point, rank candidates for the refinement
    # cap by nearness to it rather than by size. A cluttered table can easily
    # hold more than 14 card-shaped regions, and without this the target card
    # -- smaller in frame than a neighbour, but the one under the reticle --
    # gets pushed out of the top 14 before it is ever scored.
    if prefer_point is not None:
        pp = np.asarray(prefer_point, dtype=np.float64)
        ppt = (float(pp[0]), float(pp[1]))

        def _cap_key(x):
            inside = (
                cv2.pointPolygonTest(x[1].astype(np.float32), ppt, False) >= 0
            )
            # Candidates containing the subject point first, largest first
            # within each group, so the card the user framed survives the cap
            # even in a frame crowded with other card-shaped things.
            return (0 if inside else 1, -x[0])

        found.sort(key=_cap_key)
    else:
        found.sort(key=lambda x: -x[0])
    evaluated: list[tuple[float, np.ndarray, np.ndarray, float, float]] = []
    for area, quad, contour in found[:14]:
        raw_portrait = enforce_portrait(order_quad(quad))
        try:
            refined, residual = refine_quad(contour, quad)
            refined = enforce_portrait(order_quad(refined))
            if not _refinement_is_sane(raw_portrait, refined):
                # Refinement re-fits corners by intersecting line fits, and
                # near-parallel or mis-assigned sides intersect far away: this
                # corpus produced refined aspects of 15, 20, even 409 from raw
                # quads that were a correct 1.3. Robust fitting makes that MORE
                # likely, not less, because rejecting hard enough can leave a
                # small spurious run of points that fits its own line tightly
                # -- so the residual looks fine and the old residual-only
                # decline never fires. Judge the refinement by how far it moved
                # the quad, which is the thing that actually went wrong.
                refined, residual = raw_portrait, max(residual, RAW_QUAD_RESIDUAL_PX)
        except DetectionError:
            # A side with too few contour points is a reason to distrust the
            # refinement, not to discard a candidate the shape filter already
            # accepted.
            refined, residual = raw_portrait, RAW_QUAD_RESIDUAL_PX
        e01 = float(np.linalg.norm(refined[1] - refined[0]))
        e12 = float(np.linalg.norm(refined[2] - refined[1]))
        if min(e01, e12) < 1e-6:
            continue
        support = edge_support(gradient_full, refined)
        evaluated.append((area, refined, contour, residual, support))

    # ASPECT IS A SELECTION SIGNAL, NOT JUST A FINAL VETO.
    #
    # The residual ladder below relaxes its straightness requirement until
    # something passes, then takes the largest survivor. That ordering assumes
    # residual separates cards from non-cards, which holds on a clean scan and
    # fails on a handheld photo of a sleeved card: the true card's sides are a
    # mixture of surfaces (sleeve edge, neighbouring cards) so even a robust fit
    # reports a large residual, while some small artifact elsewhere in the frame
    # -- a price tag, a logo panel, a gap between cards -- fits a line
    # beautifully. Measured on 162 real photographs: a 3.2%-of-frame blob
    # refining to aspect 1.01 was repeatedly chosen over the 82.6%-of-frame card
    # refining to 1.49, because the blob cleared an earlier rung of the ladder.
    # The aspect gate at the bottom of this function then rejected the blob and
    # the whole frame with it, so a correct detection was already in hand and
    # thrown away.
    #
    # A card face is 1.40. That is the most reliable thing we know about the
    # object we are looking for, so apply it BEFORE the ladder rather than
    # after. Judge on the raw quad OR the refined one: refinement can distort a
    # genuine card (that is exactly why the raw-quad fallback below exists), so
    # requiring both would discard the cards this is meant to rescue.
    def _aspect(q: np.ndarray) -> float:
        a = float(np.linalg.norm(q[1] - q[0]))
        b = float(np.linalg.norm(q[2] - q[1]))
        return max(a, b) / max(min(a, b), 1e-9)

    raw_aspect_by_contour = {id(c): _aspect(q) for (_, q, c) in found}
    card_shaped = [
        e
        for e in evaluated
        if 0.92 < _aspect(e[1]) < 1.88
        or 0.92 < raw_aspect_by_contour.get(id(e[2]), 0.0) < 1.88
    ]
    # If nothing is card-shaped, fall back to the full set so the existing
    # aspect error below still reports what was actually found.
    pool = card_shaped or evaluated

    # SUBJECT SELECTION: OUTERMOST WINS, RESIDUAL ONLY SETS TRUST.
    #
    # When a caller names the subject, the residual ladder must not be what
    # chooses between candidates, because it early-exits on its first populated
    # rung and a card's own artwork ALWAYS populates the strictest rung: a
    # rectangular patch of printed art has crisp, genuinely straight edges,
    # while the card's outer boundary -- against a sleeve, against other cards
    # -- does not. Measured on this corpus, that handed back a fragment of the
    # Pokemon's body instead of the card, on card after card.
    #
    # The nesting relation settles it, and multicard.py already relies on the
    # same fact for its dedupe: artwork detail and the printed frame are INSIDE
    # the cut edge, so among card-shaped candidates containing the subject
    # point, the OUTERMOST is the card. Residual does not choose here; it is
    # reported as the boundary uncertainty, which is where it belongs.
    #
    # edge_support still gates, because it is the defence against a halo -- a
    # contour floating in blank space outside the true edge, which would be
    # larger AND would make an off-centre card look perfectly centred.
    scored: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    subject: Optional[tuple[float, np.ndarray, np.ndarray, float]] = None

    if prefer_point is not None:
        ppt = (float(prefer_point[0]), float(prefer_point[1]))
        containing = [
            e
            for e in pool
            if cv2.pointPolygonTest(e[1].astype(np.float32), ppt, False) >= 0
        ]
        # The floor starts at "this is a real edge" (0.55, the same bar
        # quad_candidates uses), NOT at "this is a pristine edge". Starting
        # stricter re-creates the early-exit trap the residual ladder had: a
        # small patch of printed artwork clears a 0.75 support bar that the
        # card's own outer edge -- lying against another card -- does not, so
        # the artwork ends up alone in the first tier and wins on area against
        # no competition. Within a tier, largest still wins, because artwork
        # detail is nested inside the card.
        for min_support in (0.55, 0.35, 0.0):
            tier = [e for e in containing if e[4] >= min_support]
            if tier:
                a, r, c, res, _sup = max(tier, key=lambda x: x[0])
                subject = (a, r, c, res)
                break

    if subject is None:
        for max_resid, min_support in (
            (3.0, 0.75),
            (5.0, 0.55),
            (8.0, 0.35),
            (1e9, 0.0),
        ):
            scored = [
                (a, r, c, res)
                for (a, r, c, res, sup) in pool
                if res <= max_resid and sup >= min_support
            ]
            if scored:
                break

        if not scored:
            raise DetectionError(
                "found card-shaped regions but none had straight, consistent "
                "edges. The card may be obscured, bent, or overlapping another "
                "card."
            )
    else:
        scored = [subject]

    if subject is not None:
        _, refined, contour, residual = subject
    elif prefer_point is not None:
        # A subject point was given but nothing card-shaped contained it --
        # the user is aimed between cards, or at one only partly in frame.
        # Nearest centroid is the best remaining reading of their intent.
        pp = np.asarray(prefer_point, dtype=np.float64)
        _, refined, contour, residual = min(
            scored, key=lambda x: float(np.linalg.norm(x[1].mean(axis=0) - pp))
        )
    else:
        _, refined, contour, residual = max(scored, key=lambda x: x[0])

    # SNAP THE WINNER ONTO THE REAL CUT EDGE.
    #
    # Everything above chooses WHICH region is the card; this decides where its
    # boundary actually lies. The two are separate problems and the selection
    # stage is bad at the second one, because a contour traced through a bin of
    # touching cards can be the right shape in the right place and still sit
    # several millimetres outside the subject. See snap_quad_to_edges.
    #
    # Accept the snap only if it lands on stronger edges than the quad we
    # already had. A snap that finds less support has wandered onto texture,
    # and keeping it would trade a known-loose boundary for an unknown one.
    snapped_ok = False
    try:
        snapped, snap_res = snap_quad_to_edges(gradient_full, refined)
        snapped = enforce_portrait(order_quad(snapped))
        e01s = float(np.linalg.norm(snapped[1] - snapped[0]))
        e12s = float(np.linalg.norm(snapped[2] - snapped[1]))
        # Correcting a halo shrinks the quad modestly. Collapsing it does not:
        # if the normal search finds artwork edges instead of the cut edge the
        # sides can cross, and the result is a small sliver that may well sit on
        # STRONGER gradients than the true boundary -- printed art has crisper
        # edges than a card lying against another card. So bound the geometry
        # change as well as requiring the support to improve; either test alone
        # accepts a collapse.
        area_in = abs(cv2.contourArea(refined.astype(np.float32)))
        area_out = abs(cv2.contourArea(snapped.astype(np.float32)))
        plausible_shrink = area_in > 0 and 0.55 <= area_out / area_in <= 1.05
        if (
            min(e01s, e12s) > 1e-6
            and 0.92 < max(e01s, e12s) / min(e01s, e12s) < 1.88
            and plausible_shrink
            and edge_support(gradient_full, snapped)
            > edge_support(gradient_full, refined)
        ):
            refined, residual = snapped, snap_res
            snapped_ok = True
    except DetectionError:
        pass  # keep the unsnapped boundary

    # SUBPIXEL REFINEMENT CAN MAKE THE QUAD WORSE, AND MUST BE ABLE TO DECLINE.
    #
    # refine_quad re-fits corners by intersecting line fits through the contour
    # points assigned to each side. When something straight lies against the card
    # -- most often the caliper's steel beam, which is exactly what a good
    # measurement shot contains -- points from that object get assigned to a card
    # side and drag the fit off the true edge.
    #
    # Measured on a real caliper frame: a clean 1.45-aspect raw quad came back
    # from refinement at 1.89 with a 26.1 px line residual, and the downstream
    # aspect gate then rejected the whole frame. That silently discarded the BEST
    # data in the session -- bordered cards with a metric reference in shot.
    #
    # A large residual means the line model does not describe these points, so
    # the refinement is not trustworthy and the raw approximation is better.
    raw_candidates = {id(c): (a, q) for (a, q, c) in found}
    # A snapped boundary was measured against the image's own gradients and
    # already beat the alternative on edge support, so the raw contour
    # approximation is not a better answer than it -- only than a failed line
    # refinement.
    if residual > 8.0 and not snapped_ok:
        for area, quad, cont in found:
            if cont is contour:
                fallback = enforce_portrait(order_quad(quad))
                e01 = float(np.linalg.norm(fallback[1] - fallback[0]))
                e12 = float(np.linalg.norm(fallback[2] - fallback[1]))
                if min(e01, e12) > 1e-6 and 0.92 < max(e01, e12) / min(e01, e12) < 1.88:
                    # Raw polygon corners are pixel-quantised, so report a
                    # residual reflecting that rather than the failed fit.
                    return fallback, cont, 1.0
                break

    # Re-check aspect on the REFINED quad. The candidate filter tests the raw
    # approximation, but refine_quad re-fits corners from line intersections and
    # can reshape it substantially -- a caliper beam whose raw quad squeaked
    # through comes back at aspect 9.85. Without this the relaxation ladder
    # above also has no aspect term, so its last rung (1e9, 0.0) accepts
    # anything at all rather than refusing.
    e01 = float(np.linalg.norm(refined[1] - refined[0]))
    e12 = float(np.linalg.norm(refined[2] - refined[1]))
    if min(e01, e12) < 1e-6:
        raise DetectionError("degenerate card outline")
    final_aspect = max(e01, e12) / min(e01, e12)
    if not (0.92 < final_aspect < 1.88):
        raise DetectionError(
            f"best candidate has aspect {final_aspect:.2f}, which is not a card "
            "face (nominal card is 1.40). This is usually the caliper beam, a card "
            "seen edge-on, or a partial detection."
        )

    # CONTAINER GUARD: a winning quad can be the OUTER rim of several cards
    # pushed edge to edge -- a display case tray, a binder page, a stack of
    # pocket sleeves -- rather than one card. That rim is often a strong,
    # straight, plausibly card-aspect rectangle in its own right, so it
    # passes every gate above, and border detection downstream can even find
    # SOME signal along it (the tray's own bezel, a neighbouring card's edge)
    # and report a confident but meaningless measurement. That is worse than
    # refusing. If the winner contains two or more OTHER candidates that are
    # each already valid card-shaped quads on their own, meaningfully
    # smaller, and do not substantially overlap each other, this is a
    # container: prefer the sibling nearest prefer_point (the cell the
    # camera is actually aimed at) instead of measuring the whole tray as
    # one card. Skipped when only one candidate exists at all, which is the
    # overwhelming majority of real single-card shots.
    if len(found) > 1:
        mask_h, mask_w = image.shape[:2]

        def _mask(q: np.ndarray) -> np.ndarray:
            m = np.zeros((mask_h, mask_w), dtype=np.uint8)
            cv2.fillConvexPoly(m, np.round(q).astype(np.int32), 1)
            return m

        winner_area = abs(cv2.contourArea(refined.astype(np.float32)))
        winner_mask = _mask(refined)
        siblings: list[tuple[np.ndarray, np.ndarray]] = []
        for area, quad, cont in found:
            if cont is contour or area > 0.7 * winner_area:
                continue
            qm = _mask(quad)
            qarea = float(qm.sum())
            if qarea <= 0 or float((qm & winner_mask).sum()) / qarea < 0.75:
                continue
            siblings.append((quad, cont))

        distinct: list[tuple[np.ndarray, np.ndarray]] = []
        distinct_masks: list[np.ndarray] = []
        for quad, cont in siblings:
            qm = _mask(quad)
            if all(
                float((qm & dm).sum()) / float((qm | dm).sum() or 1) < 0.2
                for dm in distinct_masks
            ):
                distinct.append((quad, cont))
                distinct_masks.append(qm)

        # CELLS TILE A TRAY; ARTWORK MERELY DOTS A CARD.
        #
        # "Two or more non-overlapping card-shaped regions inside the winner"
        # is true of a display case AND of an ordinary single card, whose
        # artwork panel, text box and printed frame are all card-ish
        # rectangles sitting inside the cut edge. Firing on the second case
        # made the tool abandon a correctly detected card and drill into a
        # patch of its own artwork -- measured here on real bulk-bin photos,
        # where a card was repeatedly replaced by a fragment of the Pokemon.
        #
        # What separates them is coverage, not count. A tray's cells tile it,
        # so together they account for most of its area; a card's internal
        # rectangles account for a few percent of the card.
        covered = np.zeros((mask_h, mask_w), dtype=np.uint8)
        for dm in distinct_masks:
            covered |= dm
        coverage = (
            float((covered & winner_mask).sum()) / float(winner_mask.sum() or 1)
        )

        if len(distinct) >= 2 and coverage >= 0.5:
            if prefer_point is None:
                raise DetectionError(
                    f"found {len(distinct) + 1} card-shaped regions nested inside "
                    "one larger boundary -- this looks like a display case, "
                    "binder page, or several cards pushed edge to edge, not one "
                    "card. Point the camera at a single card, filling more of "
                    "the frame, or use the multi-card scan mode to measure them "
                    "all at once."
                )
            pp = np.asarray(prefer_point, dtype=np.float64)
            pick_quad, pick_cont = min(
                distinct, key=lambda qc: float(np.linalg.norm(qc[0].mean(axis=0) - pp))
            )
            try:
                r2, res2 = refine_quad(pick_cont, pick_quad)
                r2 = enforce_portrait(order_quad(r2))
                e01b = float(np.linalg.norm(r2[1] - r2[0]))
                e12b = float(np.linalg.norm(r2[2] - r2[1]))
                if (
                    min(e01b, e12b) > 1e-6
                    and 0.92 < max(e01b, e12b) / min(e01b, e12b) < 1.88
                ):
                    refined, contour, residual = r2, pick_cont, res2
            except DetectionError:
                pass  # keep the container quad if the sibling won't refine cleanly

    return refined, contour, residual


def card_plane_corners_mm() -> np.ndarray:
    """Canonical card corners in mm, TL/TR/BR/BL, matching enforce_portrait."""
    return np.array(
        [
            [0.0, 0.0],
            [STANDARD_CARD_W_MM, 0.0],
            [STANDARD_CARD_W_MM, STANDARD_CARD_H_MM],
            [0.0, STANDARD_CARD_H_MM],
        ]
    )


def homography_card_to_image(image_corners: np.ndarray) -> np.ndarray:
    """H mapping card-plane mm -> image px."""
    src = card_plane_corners_mm().astype(np.float32)
    dst = np.asarray(image_corners, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return H.astype(np.float64)


def rectify(
    image: np.ndarray, image_corners: np.ndarray, px_per_mm: float = 12.0
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the card to a fronto-parallel image at a known scale.

    Returns (rectified_image, H_cardmm_to_rectpx).
    """
    w_px = int(round(STANDARD_CARD_W_MM * px_per_mm))
    h_px = int(round(STANDARD_CARD_H_MM * px_per_mm))
    dst = np.array(
        [[0, 0], [w_px - 1, 0], [w_px - 1, h_px - 1], [0, h_px - 1]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(
        np.asarray(image_corners, dtype=np.float32), dst
    )
    out = cv2.warpPerspective(
        image, M, (w_px, h_px), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    H_mm_to_rect = np.array(
        [[px_per_mm, 0, 0], [0, px_per_mm, 0], [0, 0, 1]], dtype=np.float64
    )
    return out, H_mm_to_rect


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homog = np.column_stack([pts, np.ones(len(pts))])
    out = (H @ homog.T).T
    w = out[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return out[:, :2] / w
