"""Finding a binder page in a photo, and dividing it into a pocket grid.

WHY THIS DOES NOT REUSE cardcenter.multicard.detect_cards_in_frame DIRECTLY
-----------------------------------------------------------------------------
detect_cards_in_frame finds each CARD's own quad by contrast against its
surroundings, and it works well when a card's surroundings are background
(a desk, a display case). A binder pocket page is the opposite case: nine
cards sit edge to edge inside one continuous plastic sheet, so the boundary
between pocket 1 and pocket 2 is a faint seam, not a contrast step. Tested
against 11 real photos from a physical Pokemon/Yugioh binder (2026-09-14):
a card-vs-background quad finder run directly on a whole page reliably finds
exactly one quad -- the OUTER edge of the whole page against the carpet/desk
behind it, not nine inner quads. That outer edge, it turns out, is exactly
what this module needs.

So the approach here is different on purpose: find the page's own outer
quad (still card_geometry-style quad-candidate logic: Otsu + adaptive
threshold, largest 4-sided contour), rectify JUST that to a fronto-parallel
image, and then divide the rectified page evenly into a configured pocket
grid rather than trying to detect each pocket's own edges. Two things this
buys for free:

  ADJACENT-PAGE BLEED (spec Work Instruction v2.0 S4/S8) is excluded
  automatically: a sliver of a neighbouring page peeking in at the frame
  edge is outside the CURRENT page's own quad, so it is never part of the
  rectified image at all. No separate bleed-detection step is needed.

  A card's OWN quad-candidate logic (aspect gate ~1.40, the ratio of a
  single card) turns out to also accept a 3x3 grid of cards: three cards
  wide by three tall scales both dimensions by 3x and preserves the 1.40
  aspect. That is a coincidence of a square-ish NxN grid, not a general
  fact (a 3x4 page will not have the same aspect as a card), so this module
  uses its own quad finder with a wide area gate rather than depend on that.

GRID DIMENSIONS ARE NOT AUTO-DETECTED PER PAGE, AND THAT IS A KNOWN GAP
-------------------------------------------------------------------------
The Work Instruction explicitly asks for the actual grid to be detected per
image rather than hard-coded. That was attempted here two ways and both
failed on real photos, honestly reported rather than shipped anyway:

  1. Edge-projection profiling (sum |gradient| down each column/row of the
     rectified page, look for periodic peaks at the seams). On a back page
     the profile is dominated by sticker and barcode edges, not seams,
     because a sticker is a much stronger edge than a faint plastic seam.
     On a front page it is cleaner but still noisy: tested against the
     Riolu/Weavile/Amoonguss page, valley-finding on the column profile
     returned 4 candidate seams for a page that has 2 real ones, with only
     one of the four close to a true third-boundary.

  2. Per-pocket quad detection (treat each pocket like detect_cards_in_frame
     treats a single card). Fails for the reason in the section above: a
     pocket's boundary against its neighbour is not a contrast edge.

So MIN_AREA_FRAC-based page detection plus a CONFIGURED (rows, cols) grid,
evenly subdividing the rectified page, is what this module actually does.
The default is 3 columns because that is what the Work Instruction v2.0
describes as the common case and what all 11 real sample photos showed
(rows varied 3, some pages showed fewer occupied rows near the end of a
binder). ``GridConfig`` exposes rows/cols so a caller can override it for a
binder whose pages are laid out differently; ``build_grid`` also computes a
soft ``seam_confidence`` from the projection profile (S1 above) as a hint,
not a decision -- when the profile's strongest valleys do land close to the
assumed even-thirds boundaries it is reported as "supported", otherwise
"assumed", so a human reviewing flagged output knows which pages' grid
assignment has independent visual support and which is pure geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


class GridDetectionError(RuntimeError):
    """The page (or its pocket grid) could not be located."""


@dataclass(frozen=True)
class GridConfig:
    rows: int = 3
    cols: int = 3
    # Fraction of the frame the page must occupy to be considered "the page"
    # rather than a smaller object in shot (a loose card, a price tag).
    min_area_frac: float = 0.25
    # Inward margin subtracted from each pocket cell before cropping, as a
    # fraction of the cell's own size. Pockets are divided evenly by
    # geometry, so a card sitting slightly off-centre in its pocket, or a
    # touch of perspective residual, can otherwise leak a sliver of the
    # neighbouring pocket into the crop. This trims that sliver at the cost
    # of a small amount of margin around the true card -- a deliberate
    # trade documented here rather than silently tuned.
    pocket_margin_frac: float = 0.045


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points clockwise starting from the one nearest the origin.

    Same algorithm as cardcenter.geometry.order_quad (angle-about-centroid,
    not the x+y/x-y trick, so it survives more than ~30 degrees of roll).
    Duplicated rather than imported: this module has no dependency on the
    cardcenter package at all, by design (see the CLI's --no-cardcenter
    note in binder_ingest.py) so it keeps working even if cardcenter's own
    geometry module changes shape.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    ordered = pts[np.argsort(ang)]
    area = sum(
        ordered[i][0] * ordered[(i + 1) % 4][1] - ordered[(i + 1) % 4][0] * ordered[i][1]
        for i in range(4)
    )
    if area < 0:
        ordered = ordered[::-1]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def find_page_quad(image: np.ndarray, min_area_frac: float = 0.25) -> np.ndarray:
    """Locate the binder page's own outer boundary in a photo.

    Raises GridDetectionError rather than guessing when no sufficiently
    large 4-sided region is found -- a caller (binder_ingest.py) turns that
    into a per-page ``no_card_detected``-style skip with a logged reason,
    not a silent wrong crop.
    """
    h, w = image.shape[:2]
    img_area = float(h * w)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 60, 60)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    block = max(31, (min(gray.shape) // 12) | 1)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 5
    )

    best: Optional[np.ndarray] = None
    best_area = 0.0
    for binary in (otsu, cv2.bitwise_not(otsu), adaptive, cv2.bitwise_not(adaptive)):
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area_frac * img_area or area > 0.98 * img_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = None
            for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
                a = cv2.approxPolyDP(c, eps * peri, True)
                if len(a) == 4:
                    approx = a
                    break
            if approx is None:
                continue
            if area > best_area:
                best_area = area
                best = approx.reshape(4, 2).astype(np.float64)

    if best is None:
        raise GridDetectionError(
            "could not find a page-sized quadrilateral in this photo. Shoot "
            "the binder page flat and square-on, filling most of the frame, "
            "against a background that contrasts with the page edge."
        )
    return order_quad(best)


def rectify_page(image: np.ndarray, quad: np.ndarray, target_px_per_source_px: float = 1.0):
    """Warp the page to fronto-parallel, at close to its own native resolution.

    ``target_px_per_source_px`` scales the output canvas relative to the
    quad's own average side length in the SOURCE image, rather than a fixed
    output width. This matters a lot in practice: an earlier version of this
    pipeline rectified everything to a fixed 900px-wide canvas and then
    cropped a ~180px sticker sub-region out of that for OCR, and Tesseract
    could not read any of it. Re-rectifying the same photo directly from its
    original ~4000px source at native resolution and re-cropping the same
    sticker produced clean, mostly-correct OCR text (this module's own
    ``binder_sticker.read_best_orientation`` is what recovered from there).
    Chaining a downscale-then-crop loses real pixels a sticker's small print
    needs; rectifying once, close to source resolution, does not.
    """
    q = order_quad(quad)
    w_top = float(np.linalg.norm(q[1] - q[0]))
    w_bot = float(np.linalg.norm(q[2] - q[3]))
    h_left = float(np.linalg.norm(q[3] - q[0]))
    h_right = float(np.linalg.norm(q[2] - q[1]))
    out_w = max(200, int(round(0.5 * (w_top + w_bot) * target_px_per_source_px)))
    aspect = ((h_left + h_right) / 2.0) / max((w_top + w_bot) / 2.0, 1e-6)
    out_h = max(200, int(round(out_w * aspect)))
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    rect = cv2.warpPerspective(
        image, M, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rect, M


@dataclass(frozen=True)
class Pocket:
    row: int  # 1-indexed, top to bottom
    col: int  # 1-indexed, left to right
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class PageGrid:
    rows: int
    cols: int
    pockets: dict  # (row, col) -> Pocket, every cell present regardless of occupancy
    rect_shape: tuple  # (h, w) of the rectified page image this grid is over
    seam_confidence: str  # "supported" | "assumed"


def _seam_confidence(rect: np.ndarray, rows: int, cols: int) -> str:
    """Soft cross-check: do the strongest edge-projection valleys land near
    the assumed even grid boundaries? A HINT surfaced in the output for a
    human reviewer, never something this module gates on -- see the module
    docstring for why full seam auto-detection was tried and abandoned.
    """
    try:
        gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        h, w = gray.shape[:2]

        def near_any(profile, positions, span, tol_frac=0.06):
            k = max(9, int(span * 0.02) | 1)
            prof = np.convolve(np.abs(profile), np.ones(k) / k, mode="same")
            tol = span * tol_frac
            hits = 0
            for p in positions:
                lo, hi = int(max(0, p - tol)), int(min(len(prof), p + tol))
                if hi <= lo:
                    continue
                window = prof[lo:hi]
                if window.max() > 0.7 * prof.max():
                    hits += 1
            return hits, len(positions)

        col_positions = [w * i / cols for i in range(1, cols)]
        row_positions = [h * i / rows for i in range(1, rows)]
        col_hits, col_n = near_any(gx.sum(axis=0), col_positions, w)
        row_hits, row_n = near_any(gy.sum(axis=1), row_positions, h)
        total_n = max(1, col_n + row_n)
        if (col_hits + row_hits) / total_n >= 0.75:
            return "supported"
    except Exception:
        pass
    return "assumed"


def build_grid(rect: np.ndarray, cfg: GridConfig = GridConfig()) -> PageGrid:
    h, w = rect.shape[:2]
    pockets = {}
    ch, cw = h / cfg.rows, w / cfg.cols
    for r in range(1, cfg.rows + 1):
        for c in range(1, cfg.cols + 1):
            y0, y1 = int((r - 1) * ch), int(r * ch)
            x0, x1 = int((c - 1) * cw), int(c * cw)
            pockets[(r, c)] = Pocket(r, c, x0, y0, x1, y1)
    conf = _seam_confidence(rect, cfg.rows, cfg.cols)
    return PageGrid(cfg.rows, cfg.cols, pockets, (h, w), conf)


def crop_pocket(rect: np.ndarray, pocket: Pocket, margin_frac: float) -> np.ndarray:
    w = pocket.x1 - pocket.x0
    h = pocket.y1 - pocket.y0
    mx, my = int(w * margin_frac), int(h * margin_frac)
    x0, y0 = pocket.x0 + mx, pocket.y0 + my
    x1, y1 = pocket.x1 - mx, pocket.y1 - my
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(rect.shape[1], x1), min(rect.shape[0], y1)
    return rect[y0:y1, x0:x1]


def mirror_column(col: int, cols: int) -> int:
    """Work Instruction v2.0 S5: back_column = (C + 1) - front_column.

    The page flips about its LEFT (spine) edge, a vertical axis, so columns
    reverse and rows are unchanged. This is pure arithmetic and is kept as
    its own function because it is the one piece of this whole module that
    the spec calls out as the thing most likely to be gotten backwards.
    """
    return (cols + 1) - col


def has_content(pocket_crop: np.ndarray, dark_frac_floor: float = 0.02) -> bool:
    """Cheap occupancy check: an empty plastic pocket over the page backing
    is close to uniformly lit; a card underneath introduces real texture and
    tonal range. Not a card detector -- just enough to tell "there is
    plainly something here" from "this is empty sleeve", which is all
    no_card_detected (Work Instruction S8) needs.
    """
    if pocket_crop.size == 0:
        return False
    gray = cv2.cvtColor(pocket_crop, cv2.COLOR_BGR2GRAY) if pocket_crop.ndim == 3 else pocket_crop
    std = float(gray.std())
    edges = cv2.Canny(gray, 40, 120)
    edge_frac = float((edges > 0).mean())
    return std > 18.0 or edge_frac > dark_frac_floor
