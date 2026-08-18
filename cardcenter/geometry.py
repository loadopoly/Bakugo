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
        line, res = fit_line_tls(chosen)
        lines.append(line)
        residuals.append(res)

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

    gray_full = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    )
    gx = cv2.Sobel(gray_full, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_full, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

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

        # The quad must actually describe the shape, not just bound it.
        if cv2.contourArea(cv2.convexHull(pts)) < 0.80 * quad_area:
            continue

        e01 = np.linalg.norm(quad[1] - quad[0])
        e12 = np.linalg.norm(quad[2] - quad[1])
        if min(e01, e12) < 1e-6:
            continue
        aspect = max(e01, e12) / min(e01, e12)
        # A card face is 1.400. Perspective stretches this, but only so far
        # before the quad stops being a card at all.
        #
        # The old window ran to expected*1.45 = 2.03, which admitted the
        # CALIPER'S STEEL SCALE. Measured on 75 real photographs: 73% of
        # accepted quads had an aspect implausible for a card face, including
        # 9.85, 9.33 and 5.67 -- those are the caliper beam, a card seen
        # edge-on, and partial detections. They then routed to geometry_only,
        # so a misdetection arrived wearing a respectable label rather than
        # being refused.
        #
        # 1.75 still allows substantial perspective on a genuine card while
        # excluding anything long and thin.
        if not (1.15 < aspect < 1.75):
            continue

        if edge_support(gradient, quad) < min_edge_support:
            continue

        found.append((quad_area, quad, solid))
    return found


def find_card_quad(
    image: np.ndarray, min_area_frac: float = 0.008
) -> tuple[np.ndarray, np.ndarray, float]:
    """Locate the single largest card in the image.

    ``min_area_frac`` defaulted to 0.03 for synthetic renders, where the card
    fills the frame by construction. On a real capture session the median card
    occupies 3.6% of the frame and many good shots are below 3% -- so the
    default was rejecting cards the detector had already found. Measured on 200
    real photographs: at 0.03 the pipeline measured 0 of 60; the candidate stage
    was finding valid 1.44-aspect quads and the area gate was discarding them.

    Returns (refined_corners, contour, mean_line_residual_px).
    """
    found = quad_candidates(image, min_area_frac=min_area_frac)
    if not found:
        raise DetectionError(
            "could not locate a card-shaped quadrilateral. Shoot the card "
            "against a plain contrasting background with all four edges visible."
        )

    _g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gradient_full = cv2.magnitude(
        cv2.Sobel(_g, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(_g, cv2.CV_32F, 0, 1, ksize=3),
    )

    # Largest-wins is not enough: a slightly larger candidate that fits its own
    # edges badly is a blob or a halo, not a card, and picking it silently
    # displaces every border measurement. But a product of exponential penalties
    # is worse -- it lets a small, crisp inner rectangle outscore the real card.
    # Use explicit gates instead, then take the largest survivor, relaxing the
    # gates only if nothing passes.
    found.sort(key=lambda x: -x[0])
    evaluated: list[tuple[float, np.ndarray, np.ndarray, float, float]] = []
    for area, quad, contour in found[:14]:
        try:
            refined, residual = refine_quad(contour, quad)
        except DetectionError:
            continue
        refined = enforce_portrait(order_quad(refined))
        e01 = float(np.linalg.norm(refined[1] - refined[0]))
        e12 = float(np.linalg.norm(refined[2] - refined[1]))
        if min(e01, e12) < 1e-6:
            continue
        support = edge_support(gradient_full, refined)
        evaluated.append((area, refined, contour, residual, support))

    scored = []
    for max_resid, min_support in ((3.0, 0.75), (5.0, 0.55), (8.0, 0.35), (1e9, 0.0)):
        scored = [
            (a, r, c, res)
            for (a, r, c, res, sup) in evaluated
            if res <= max_resid and sup >= min_support
        ]
        if scored:
            break

    if not scored:
        raise DetectionError(
            "found card-shaped regions but none had straight, consistent edges. "
            "The card may be obscured, bent, or overlapping another card."
        )

    _, refined, contour, residual = max(scored, key=lambda x: x[0])

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
    if residual > 8.0:
        for area, quad, cont in found:
            if cont is contour:
                fallback = enforce_portrait(order_quad(quad))
                e01 = float(np.linalg.norm(fallback[1] - fallback[0]))
                e12 = float(np.linalg.norm(fallback[2] - fallback[1]))
                if min(e01, e12) > 1e-6 and 1.15 < max(e01, e12) / min(e01, e12) < 1.75:
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
    if not (1.15 < final_aspect < 1.75):
        raise DetectionError(
            f"best candidate has aspect {final_aspect:.2f}, which is not a card "
            "face (a card is 1.40). This is usually the caliper beam, a card "
            "seen edge-on, or a partial detection."
        )
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
