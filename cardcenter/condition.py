"""Corners, edges and surface, measured from the pixels -- one view at a time.

Until 2.24 the grade estimate had a centering measurement and nothing else.
Corners, edges and surface started at 10 and were docked by how cleanly the
outline was found (a line-fit residual, the border detector's confidence),
with each penalty capped at 2 points. So a soft or tilted live frame landed at
exactly 10 - 2 = 8 on every card, which is what the 2.23 screenshots showed: a
crisp 51.6/48.4 Probopass and a soft one both "~PSA 8". Detection quality says
how well the photo shows the card. It says nothing about the card.

This module looks at the card instead, in a canonical frame: the card warped
flat at up to ``MAX_PX_PER_MM`` with ``MARGIN_MM`` of whatever it lies on kept
round it, so its own edge and corners can be compared with the background.

CORNERS. Each corner window is turned to the top-left orientation. Rays from
the centre of the nominal rounded corner (radius ``NOMINAL_RADIUS_MM``) find
where card material stops. ``loss_mm`` is how far inside the nominal outline
that happens (worn, rounded or chipped corners lose material; a fresh die cut
loses none), and ``whitening_mm`` is how deep the run of white fibre goes
inward from that boundary. The card/background decision uses colour models
sampled from the corner's own two border bands and background bands, so a
silver border on wood, a yellow one on a black mat and a blue back on a desk
all work the same way.

EDGES. Each side is sampled as normal profiles every 0.5 mm. The boundary is
found per profile; a robust line through those gives the straight cut, and
departures inward from it are nicks. White fibre at the boundary, measured
inward, is edge wear. Positions where the border well inside the edge is also
white are glare on the border, not wear, and are excluded rather than scored.

SURFACE. Scratches, scuffs, dents and creases are seen in the reflection of a
light, not in the print: a grader tilts the card under a lamp. A single
photograph can't tell a scratch from a line in the artwork, so this module
only records, per view, which parts of the card were under a highlight and
where thin dark structures sit inside it. Telling them from the print needs
the same place seen without the highlight, which is multi-view work done in
``condition_evidence``.

WHAT IS NOT CLAIMED. A reading is only made where the pixels support it:
enough px/mm, a sharp enough transition, a background distinguishable from
both the border and from white fibre. Anything else is reported as not
resolvable, with the reason, and is never scored as a clean card. The
thresholds that turn these millimetres into grades live in
``data/condition_standards.json`` and are this project's reading of the
graders' published wording, not graders' numbers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .types import STANDARD_CARD_H_MM, STANDARD_CARD_W_MM, CaptureSpec

CORNERS = ("top_left", "top_right", "bottom_right", "bottom_left")
SIDES = ("top", "right", "bottom", "left")

MAX_PX_PER_MM = 16.0
MARGIN_MM = 2.5
CORNER_WINDOW_MM = 10.0
CORNER_LINE_FROM_MM = 6.0   # edges are fitted past any plausible worn arc
EDGE_END_MM = 4.0          # the first and last 4 mm of a side belong to its corners
EDGE_STEP_MM = 0.5
CARD_THICKNESS_MM = 0.32   # a standard TCG card; the side face seen at a tilt
SURFACE_CELL_MM = 3.0

_DATA = Path(__file__).parent / "data" / "condition_standards.json"


def plain(obj):
    """numpy scalars -> Python, recursively: the readings go out as JSON."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


@lru_cache(maxsize=1)
def load_condition_standards() -> dict:
    with open(_DATA, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _floors() -> dict:
    return load_condition_standards()["resolution"]


def nominal_radius_mm() -> float:
    return float(load_condition_standards()["nominal_corner_radius_mm"])


# ---------------------------------------------------------------------------
# Per-view readings
# ---------------------------------------------------------------------------


@dataclass
class CornerReading:
    name: str
    resolvable: bool
    reason: str = ""
    px_per_mm: float = 0.0
    loss_mm: float = 0.0           # material missing inside the nominal rounded corner
    whitening_mm: float = 0.0      # depth of white fibre inward from the boundary
    whitening_len_mm: float = 0.0  # contour length carrying it
    whitening_resolvable: bool = False
    glare: bool = False
    blur_mm: float = 0.0
    sigma_mm: float = 0.0

    @property
    def severity_mm(self) -> float:
        """The larger of the two kinds of corner wear, in mm."""
        w = self.whitening_mm if self.whitening_resolvable else 0.0
        return max(self.loss_mm, w)

    def to_dict(self) -> dict:
        return {
            "resolvable": self.resolvable,
            "reason": self.reason,
            "px_per_mm": round(self.px_per_mm, 2),
            "loss_mm": round(self.loss_mm, 3),
            "whitening_mm": round(self.whitening_mm, 3),
            "whitening_len_mm": round(self.whitening_len_mm, 2),
            "whitening_resolvable": self.whitening_resolvable,
            "glare": self.glare,
            "blur_mm": round(self.blur_mm, 3),
            "sigma_mm": round(self.sigma_mm, 3),
        }


@dataclass
class EdgeReading:
    side: str
    resolvable: bool
    reason: str = ""
    px_per_mm: float = 0.0
    whitening_frac: float = 0.0    # share of the usable length carrying white fibre
    whitening_resolvable: bool = False
    usable_frac: float = 0.0       # share of the side not lost to glare or the frame
    nick_count: int = 0
    nick_max_mm: float = 0.0       # deepest departure inward from the straight cut
    noise_mm: float = 0.0
    blur_mm: float = 0.0
    side_face_mm: float = 0.0

    def to_dict(self) -> dict:
        return {
            "resolvable": self.resolvable,
            "reason": self.reason,
            "px_per_mm": round(self.px_per_mm, 2),
            "whitening_frac": round(self.whitening_frac, 3),
            "whitening_resolvable": self.whitening_resolvable,
            "usable_frac": round(self.usable_frac, 3),
            "nick_count": self.nick_count,
            "nick_max_mm": round(self.nick_max_mm, 3),
            "noise_mm": round(self.noise_mm, 3),
            "blur_mm": round(self.blur_mm, 3),
            "side_face_mm": round(self.side_face_mm, 3),
        }


@dataclass
class SurfaceReading:
    resolvable: bool
    reason: str = ""
    px_per_mm: float = 0.0
    # cells (SURFACE_CELL_MM square) that sat under a highlight in this view
    inspected: Optional[np.ndarray] = None
    # thin dark structures inside a highlight: (x_mm, y_mm, length_mm, dx, dy)
    candidates: list = field(default_factory=list)
    glare_frac: float = 0.0
    texture: float = 0.0           # share of highlight pixels that are structured (foil)
    # the card, flat and grey, at the surface resolution, with the highlight
    # masked out: what the print looks like without a reflection on it
    diffuse: Optional[np.ndarray] = None
    diffuse_mask: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "resolvable": self.resolvable,
            "reason": self.reason,
            "px_per_mm": round(self.px_per_mm, 2),
            "glare_frac": round(self.glare_frac, 3),
            "texture": round(self.texture, 3),
            "candidates": len(self.candidates),
            "inspected_frac": (round(float(self.inspected.mean()), 3)
                               if self.inspected is not None else 0.0),
        }


@dataclass
class ConditionView:
    """Everything one view says about the card's physical condition."""

    face: str
    face_confidence: float
    px_per_mm: float
    corners: dict
    edges: dict
    surface: SurfaceReading
    tilt_deg: Optional[float] = None

    def to_dict(self) -> dict:
        return plain({
            "face": self.face,
            "face_confidence": round(self.face_confidence, 2),
            "px_per_mm": round(self.px_per_mm, 2),
            "tilt_deg": None if self.tilt_deg is None else round(self.tilt_deg, 1),
            "corners": {k: v.to_dict() for k, v in self.corners.items()},
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
            "surface": self.surface.to_dict(),
        })


# ---------------------------------------------------------------------------
# Canonical frame
# ---------------------------------------------------------------------------


@dataclass
class _Canon:
    bgr: np.ndarray
    lab: np.ndarray
    R: float                 # canonical px per mm
    M: float                 # margin, mm
    detail: np.ndarray       # bool: pixels that carry real detail
    detail_px_per_mm: float  # true resolution where detail is set
    other_px_per_mm: float   # true resolution elsewhere
    to_image: Optional[np.ndarray] = None   # card mm -> image px homography

    def px(self, mm: float) -> float:
        return (self.M + mm) * self.R

    def local_scale(self, x: float, y: float) -> float:
        """Image px per card mm at (x, y), in the worse direction. On a tilted
        card the far edge has a fraction of the near edge's pixels; the
        card-average px/mm says nothing about which."""
        if self.to_image is None:
            return 1.0
        Hm = self.to_image
        p = np.array([x, y, 1.0])
        w = float(Hm[2] @ p)
        J = (Hm[:2, :2] * w - np.outer(Hm[:2] @ p, Hm[2, :2])) / (w * w)
        return float(np.linalg.svd(J, compute_uv=False).min())

    def res_at(self, x0: float, y0: float, x1: float, y1: float) -> float:
        """True px/mm over a card-mm rectangle: the worst point of it, and
        the upsampled resolution when it straddles detail and upsampled
        pixels."""
        base = self.detail_px_per_mm
        if self.detail_px_per_mm != self.other_px_per_mm:
            h, w = self.detail.shape
            a = self.detail[max(0, int(self.px(y0))):min(h, int(self.px(y1)) + 1),
                            max(0, int(self.px(x0))):min(w, int(self.px(x1)) + 1)]
            if not (a.size and a.mean() >= 0.98):
                base = self.other_px_per_mm
        if self.to_image is None:
            return base
        W, H = STANDARD_CARD_W_MM, STANDARD_CARD_H_MM
        pts = [(min(max(x, 0.0), W), min(max(y, 0.0), H)) for x in (x0, x1) for y in (y0, y1)]
        # relative to the card's mean scale, which is what px_per_mm states
        mean = self.local_scale(W / 2.0, H / 2.0)
        worst = min(self.local_scale(x, y) for x, y in pts)
        return float(base * min(1.0, worst / max(mean, 1e-9)))


def _card_quad_mm(M: float, R: float) -> np.ndarray:
    W, H = STANDARD_CARD_W_MM, STANDARD_CARD_H_MM
    return np.array([[M * R, M * R], [(W + M) * R, M * R],
                     [(W + M) * R, (H + M) * R], [M * R, (H + M) * R]], np.float32)


def _canonical(image, corners_px, px_per_mm, detail_mask=None,
               other_px_per_mm=None) -> _Canon:
    R = float(min(MAX_PX_PER_MM, max(2.0, px_per_mm)))
    M = MARGIN_MM
    W, H = STANDARD_CARD_W_MM, STANDARD_CARD_H_MM
    size = (int(round((W + 2 * M) * R)), int(round((H + 2 * M) * R)))
    Hm = cv2.getPerspectiveTransform(np.asarray(corners_px, np.float32).reshape(4, 2),
                                     _card_quad_mm(M, R))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bgr = cv2.warpPerspective(image, Hm, size, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    if detail_mask is not None and other_px_per_mm is not None:
        dm = cv2.warpPerspective(detail_mask.astype(np.uint8), Hm, size,
                                 flags=cv2.INTER_NEAREST, borderValue=0) > 0
        other = float(min(px_per_mm, other_px_per_mm))
    else:
        dm = np.ones((size[1], size[0]), bool)
        other = float(px_per_mm)
    src = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    to_image = cv2.getPerspectiveTransform(src, np.asarray(corners_px, np.float32).reshape(4, 2)).astype(np.float64)
    return _Canon(bgr, lab, R, M, dm, float(px_per_mm), other, to_image)


def _rot_tl(a: np.ndarray, k: int) -> np.ndarray:
    """Flip a canonical array so corner ``k`` (TL, TR, BR, BL) sits top-left."""
    if k == 1:
        a = a[:, ::-1]
    elif k == 2:
        a = a[::-1, ::-1]
    elif k == 3:
        a = a[::-1, :]
    return np.ascontiguousarray(a)


# ---------------------------------------------------------------------------
# Colour models
# ---------------------------------------------------------------------------


@dataclass
class _Models:
    border: np.ndarray
    border_s: np.ndarray
    # the background as up to three colour clusters (mean, spread): a card
    # on a black bag with one grey fold of reflection is two backgrounds,
    # and a single Gaussian between them calls the fold "card"
    bg_clusters: list
    separable: bool
    white_vs_bg: bool          # can white fibre be told from the background?
    white_L: float             # L at or above which a pixel reads as fibre
    white_C: float             # chroma at or below which it does

    @property
    def bg(self) -> np.ndarray:
        return self.bg_clusters[0][0]

    def _d_bg(self, lab: np.ndarray) -> np.ndarray:
        out = None
        for m, s in self.bg_clusters:
            d = np.sqrt((((lab - m) / s) ** 2).sum(-1))
            out = d if out is None else np.minimum(out, d)
        return out

    def p_card(self, lab: np.ndarray, white_is_card: bool = True) -> np.ndarray:
        """Probability-like score that each pixel is card material (border or
        white fibre) rather than background."""
        db = np.sqrt((((lab - self.border) / self.border_s) ** 2).sum(-1))
        dg = self._d_bg(lab)
        if self.white_vs_bg and white_is_card:
            white = self.is_white(lab)
            db = np.where(white, np.minimum(db, 1.0), db)
        return 1.0 / (1.0 + np.exp(np.clip(db - dg, -30.0, 30.0)))

    def is_white(self, lab: np.ndarray) -> np.ndarray:
        C = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
        return (lab[..., 0] >= self.white_L) & (C <= self.white_C)


def _robust(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.median(samples, axis=0)
    s = 1.4826 * np.median(np.abs(samples - m), axis=0)
    return m, np.maximum(s, 2.5)


def _bg_clusters(bg: np.ndarray, k_max: int = 3) -> list:
    """(mean, spread, weight) per background cluster, largest first."""
    bg = bg.reshape(-1, 3).astype(np.float32)
    if len(bg) < 60:
        m, s = _robust(bg)
        return [(m, s, 1.0)]
    m0, s0 = _robust(bg)
    if float(np.linalg.norm(s0)) < 12.0:
        return [(m0, s0, 1.0)]
    k = min(k_max, max(1, len(bg) // 40))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, _ = cv2.kmeans(bg, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    out = []
    for c in range(k):
        pts = bg[labels == c]
        if len(pts) < max(12, 0.05 * len(bg)):
            continue
        m, s = _robust(pts)
        out.append((m, s, len(pts) / len(bg)))
    out.sort(key=lambda t: -t[2])
    return out or [(m0, s0, 1.0)]


def _models(border_px: np.ndarray, bg_px: np.ndarray) -> Optional[_Models]:
    if len(border_px) < 20 or len(bg_px) < 20:
        return None
    mb, sb = _robust(border_px.reshape(-1, 3))
    clusters = _bg_clusters(bg_px)
    # Separable when every sizeable background cluster sits further from the
    # border than the pixels scatter. The spreads are capped: a metallic
    # border shades across its own width, which widens L without making the
    # border look any more like the table.
    separable = True
    for mg, sg, wgt in clusters:
        gap = float(np.linalg.norm((mb - mg) / np.sqrt(np.minimum(sb, 20.0) ** 2
                                                       + np.minimum(sg, 20.0) ** 2)))
        if (float(np.linalg.norm(mb - mg)) < 12.0 or gap < 1.6) and wgt >= 0.15:
            separable = False
    Cb = float(math.hypot(mb[1] - 128.0, mb[2] - 128.0))
    # White fibre is brighter than the border it shows through. Lightness is
    # the cue that survives a phone's JPEG: chroma is stored at half
    # resolution and smeared another ~0.5 mm by demosaicing, so the outer
    # half-millimetre of ANY coloured border on a dark background reads as
    # desaturated -- a yellow border over a black bag measured b* 149 at the
    # cut rising to 186 a millimetre in, with L* flat. Colour loss is only
    # required on top, never alone.
    white_L = float(mb[0] + min(45.0, max(18.0, 3.0 * sb[0])))
    if Cb >= 25.0:
        white_C = float(max(14.0, 0.65 * Cb))
    else:
        white_C = float(max(14.0, 0.6 * Cb + 4.0))
    white_L = max(white_L, 150.0)
    bg_white = False
    for mg, sg, wgt in clusters:
        Cg = float(math.hypot(mg[1] - 128.0, mg[2] - 128.0))
        if wgt >= 0.1 and (mg[0] + 1.5 * min(sg[0], 20.0) >= white_L - 10.0) and Cg <= white_C + 6.0:
            bg_white = True
    # a border already near white leaves no room to see fibre on it
    resolvable_white = (not bg_white) and white_L <= 248.0
    return _Models(mb, sb, [(m, s) for m, s, _ in clusters], separable,
                   resolvable_white, white_L, white_C)


# ---------------------------------------------------------------------------
# Face
# ---------------------------------------------------------------------------


def classify_face(canon_bgr: np.ndarray, R: float, M: float) -> tuple[str, float]:
    """``("back", c)`` for the Pokemon back, else ``("front", c)``.

    The back is a saturated blue (hue ~108 in OpenCV's 0-180, measured on
    real backs in ingest/) right out to the cut, with the red half of a
    Poke Ball at its centre. All three are required: blue artwork on a front
    (water types, full arts) fills the face but not the border ring, and a
    blue-bordered front has no red disc in the middle. Other games' backs
    read as front: no claim is made about them."""
    W, H = STANDARD_CARD_W_MM, STANDARD_CARD_H_MM
    hsv = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    blue = (h >= 96) & (h <= 126) & (s >= 80) & (v >= 35)
    red = ((h <= 8) | (h >= 172)) & (s >= 90) & (v >= 60)

    def box(x0, y0, x1, y1):
        return (slice(int((M + y0) * R), int((M + y1) * R)),
                slice(int((M + x0) * R), int((M + x1) * R)))

    inner = blue[box(1.0, 1.0, W - 1.0, H - 1.0)].mean()
    ring = np.concatenate([
        blue[box(0.6, 0.6, W - 0.6, 2.2)].ravel(), blue[box(0.6, H - 2.2, W - 0.6, H - 0.6)].ravel(),
        blue[box(0.6, 2.2, 2.2, H - 2.2)].ravel(), blue[box(W - 2.2, 2.2, W - 0.6, H - 2.2)].ravel(),
    ]).mean()
    centre_red = red[box(W / 2 - 12, H / 2 - 14, W / 2 + 12, H / 2 + 2)].mean()
    score = min(inner / 0.42, ring / 0.6, centre_red / 0.04)
    if score >= 1.0:
        return "back", float(min(1.0, 0.5 + 0.5 * min(score, 2.0) / 2.0))
    return "front", float(min(1.0, 0.5 + 0.5 * (1.0 - score)))


# ---------------------------------------------------------------------------
# Pose: how much of the card's side face shows past each edge
# ---------------------------------------------------------------------------


def _focal_from_rectangle(Hm: np.ndarray, cx: float, cy: float) -> Optional[float]:
    """Focal length (px) from the homography of a rectangle of known aspect
    with the principal point at (cx, cy), or None when the view is too close
    to square-on for the constraints to say anything."""
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
    G = T @ Hm
    G = G / np.linalg.norm(G[:, 0])
    a1, a2 = G[:, 0], G[:, 1]
    rows, rhs = [], []
    # r1.r2 = 0 :  (x1 x2 + y1 y2)/f^2 + z1 z2 = 0
    rows.append(a1[0] * a2[0] + a1[1] * a2[1])
    rhs.append(-a1[2] * a2[2])
    # |r1| = |r2|
    rows.append(a1[0] ** 2 + a1[1] ** 2 - a2[0] ** 2 - a2[1] ** 2)
    rhs.append(-(a1[2] ** 2 - a2[2] ** 2))
    A = np.asarray(rows)
    b = np.asarray(rhs)
    if float(np.abs(b).max()) < 1e-9:
        return None
    inv_f2 = float((A @ b) / max(float(A @ A), 1e-18))
    if not np.isfinite(inv_f2) or inv_f2 <= 0:
        return None
    return float(1.0 / math.sqrt(inv_f2))


def side_face_mm(corners_px, capture: Optional[CaptureSpec], image_shape) -> tuple[dict, Optional[float]]:
    """Visible width of the card's side face past each edge, in card mm, and
    the camera's tilt from the card normal. Zero everywhere when the camera
    is unknown.

    A card is ~0.32 mm thick. Seen from a tilt the side face facing the
    camera shows outside the top surface's edge, and it is the white or grey
    core of the card stock -- it reads exactly like edge wear. The bottom of
    that face projects onto the top plane at P + T/(h+T) (C - P), so it shows
    outside an edge whose outward normal n has (C - P).n > 0, by that much."""
    zero = {s: 0.0 for s in SIDES}
    if capture is None:
        return zero, None
    K = capture.intrinsics(image_shape)
    if K is None:
        return zero, None
    W, H = STANDARD_CARD_W_MM, STANDARD_CARD_H_MM
    src = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, np.asarray(corners_px, np.float32).reshape(4, 2))
    # The focal length a caller passes is usually a guess (a nominal field of
    # view, applied to whichever side of a portrait frame is "width"). The
    # card's own outline fixes it when the view is oblique enough: a
    # rectangle's sides are orthogonal and its axes equally scaled (Zhang).
    f_self = _focal_from_rectangle(Hm, K[0, 2], K[1, 2])
    if f_self is not None and 0.3 * K[0, 0] <= f_self <= 3.0 * K[0, 0]:
        K = K.copy()
        K[0, 0] = K[1, 1] = f_self
    B = np.linalg.inv(K) @ Hm
    n1 = np.linalg.norm(B[:, 0])
    if n1 < 1e-12:
        return zero, None
    lam = 1.0 / n1
    r1, r2, t = B[:, 0] * lam, B[:, 1] * lam, B[:, 2] * lam
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    Rm = np.column_stack([r1, r2, r3])
    C = -Rm.T @ t
    h = abs(float(C[2]))
    if h < 1e-6:
        return zero, None
    tilt = math.degrees(math.acos(min(1.0, abs(float(r3[2])))))
    mids = {"top": (W / 2, 0.0, (0, -1)), "bottom": (W / 2, H, (0, 1)),
            "left": (0.0, H / 2, (-1, 0)), "right": (W, H / 2, (1, 0))}
    out = {}
    for s, (px, py, n) in mids.items():
        reach = (C[0] - px) * n[0] + (C[1] - py) * n[1]
        out[s] = float(max(0.0, CARD_THICKNESS_MM / (h + CARD_THICKNESS_MM) * reach))
    return out, tilt


# ---------------------------------------------------------------------------
# Corners
# ---------------------------------------------------------------------------


def _step_fit(p: np.ndarray) -> int:
    """Index splitting a profile into card (high p) before it and background
    after it, at least cost. ``p`` runs from inside the card outward."""
    n = len(p)
    if n < 3:
        return 0
    # cost(k) = sum_{i<k}(1-p_i) + sum_{i>=k} p_i
    c1 = np.concatenate([[0.0], np.cumsum(1.0 - p)])
    c2 = np.concatenate([np.cumsum(p[::-1])[::-1], [0.0]])
    return int(np.argmin(c1 + c2))


def _transition_width(t: np.ndarray, step_mm: float) -> float:
    """20-80% fall of a card->background profile, in mm. ``t`` is the
    pixel's position between the background (0) and border (1) colours --
    linear in the pixel values, so it shows the optics' blur (the card/
    background DECISION is a sigmoid and would hide it)."""
    if len(t) < 4:
        return 0.0
    t = np.clip(t, -0.5, 1.5)
    hi_i = np.where(t >= 0.8)[0]
    lo_i = np.where(t <= 0.2)[0]
    if not len(hi_i) or not len(lo_i):
        return float("inf")
    # first fall past 0.2 after the last 0.8 before it
    first_lo = lo_i.min()
    before = hi_i[hi_i < first_lo]
    if not len(before):
        return float("inf")
    last_hi = before.max()
    # sub-sample: interpolate both crossings
    def cross(i0, i1, level):
        a, b = t[i0], t[i1]
        return i0 + (a - level) / (a - b) if a != b else i0
    x8 = cross(last_hi, last_hi + 1, 0.8)
    x2 = cross(first_lo - 1, first_lo, 0.2)
    return float(max(0.0, x2 - x8)) * step_mm


def _sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return cv2.remap(img, xs.astype(np.float32), ys.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _linear_t(lab: np.ndarray, m: "_Models") -> np.ndarray:
    """Position of each pixel along the background->border colour axis."""
    bg = m.bg.astype(np.float32)
    d = (m.border - bg).astype(np.float32)
    return ((lab - bg) @ d) / float(max(d @ d, 1e-6))


def _fit_line(pos: np.ndarray, off: np.ndarray) -> Optional[tuple]:
    """Robust off = a + b * pos, or None."""
    ok = np.isfinite(off)
    if ok.sum() < 4:
        return None
    P_, O_ = pos[ok], off[ok]
    A = np.column_stack([np.ones_like(P_), P_])
    w = np.ones_like(P_)
    coef = np.zeros(2)
    for _ in range(4):
        coef, *_ = np.linalg.lstsq(A * w[:, None], O_ * w, rcond=None)
        r = O_ - A @ coef
        sd = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        w = 1.0 / np.maximum(1.0, np.abs(r) / (2.5 * sd))
    return float(coef[0]), float(coef[1])


def _analyse_corner(canon: _Canon, k: int, name: str, inner_mm: dict,
                    debug: Optional[dict] = None) -> CornerReading:
    fl = _floors()
    R, M = canon.R, canon.M
    lab = _rot_tl(canon.lab, k)
    n = int(round((M + CORNER_WINDOW_MM) * R))
    win = lab[:n, :n]
    # true resolution at this corner, in canonical TL orientation
    xs = (0.0, CORNER_WINDOW_MM) if k in (0, 3) else (STANDARD_CARD_W_MM - CORNER_WINDOW_MM, STANDARD_CARD_W_MM)
    ys = (0.0, CORNER_WINDOW_MM) if k in (0, 1) else (STANDARD_CARD_H_MM - CORNER_WINDOW_MM, STANDARD_CARD_H_MM)
    res = canon.res_at(xs[0] - 1.0, ys[0] - 1.0, xs[1] + 1.0, ys[1] + 1.0)
    out = CornerReading(name=name, resolvable=False, px_per_mm=res)
    if res < fl["corner_min_px_per_mm"]:
        out.reason = f"{res:.1f} px/mm here; corners need {fl['corner_min_px_per_mm']:.0f}"
        return out

    def P(mm):
        return (M + mm) * R

    # The border width on the two sides that meet here (TL orientation).
    bw_a = inner_mm.get(("top", "top", "bottom", "bottom")[k], 2.5)   # along v (rows)
    bw_b = inner_mm.get(("left", "right", "right", "left")[k], 2.5)   # along u (cols)
    deep = max(0.9, min(1.5, min(bw_a, bw_b) - 0.4))
    b_in = 0.45
    if deep <= b_in + 0.2:
        out.reason = "border too narrow to sample"
        return out
    i = lambda mm: int(round(P(mm)))
    border = np.concatenate([
        win[i(b_in):i(deep), i(CORNER_LINE_FROM_MM):i(CORNER_WINDOW_MM)].reshape(-1, 3),
        win[i(CORNER_LINE_FROM_MM):i(CORNER_WINDOW_MM), i(b_in):i(deep)].reshape(-1, 3)])
    bg = np.concatenate([
        win[i(-MARGIN_MM + 0.25):i(-0.7), i(1.0):i(CORNER_WINDOW_MM)].reshape(-1, 3),
        win[i(1.0):i(CORNER_WINDOW_MM), i(-MARGIN_MM + 0.25):i(-0.7)].reshape(-1, 3)])
    models = _models(border, bg)
    if models is None:
        out.reason = "too few pixels to model the border"
        return out
    if not models.separable:
        out.reason = "the background here looks like the border"
        return out
    pc = models.p_card(win).astype(np.float32)
    # the straight edges are followed on border-vs-background alone: a
    # sleeve's edge or a reflection beside the card is white too
    pc_plain = models.p_card(win, white_is_card=False).astype(np.float32)
    lt = _linear_t(win, models).astype(np.float32)
    white = models.is_white(win).astype(np.float32)
    step = 0.5 / R

    # Where the two straight edges actually run near this corner. The
    # outline that placed the canonical frame is good to a few tenths of a
    # millimetre on a tilted phone frame; a corner judged against it reads
    # every such offset as wear (the 2025-09-25 Machamp's bottom-left read
    # 0.67 mm "lost" for a corner whose edges simply sat 0.5 mm in).
    # The search stays within a millimetre of where the outline put the
    # edge: a penny sleeve's own edge runs 1-3 mm outside the card.
    depth = np.arange(-1.0, 1.0, step)
    along = np.arange(CORNER_LINE_FROM_MM, CORNER_WINDOW_MM - 0.3, 0.25)
    top_off = np.full(len(along), np.nan)
    left_off = np.full(len(along), np.nan)
    for j, a_mm in enumerate(along):
        # top edge: walk up column u = a_mm, from inside outward
        prof = _sample(pc_plain, np.full((1, len(depth)), P(a_mm)), P(depth[::-1])[None, :])[0]
        kk = _step_fit(prof)
        if 0 < kk < len(depth) and prof[:kk].mean() - prof[kk:].mean() >= 0.5:
            top_off[j] = depth[::-1][kk]
        prof = _sample(pc_plain, P(depth[::-1])[None, :], np.full((1, len(depth)), P(a_mm)))[0]
        kk = _step_fit(prof)
        if 0 < kk < len(depth) and prof[:kk].mean() - prof[kk:].mean() >= 0.5:
            left_off[j] = depth[::-1][kk]
    lt_top = _fit_line(along, top_off)       # v = a + b u
    lt_left = _fit_line(along, left_off)     # u = c + d v
    if lt_top is None or lt_left is None:
        out.reason = "could not follow the edges into the corner"
        return out
    a_, b_ = lt_top
    c_, d_ = lt_left
    if debug is not None:
        debug["top_pts"] = list(zip(along, top_off))
        debug["left_pts"] = list(zip(left_off, along))
    r0 = nominal_radius_mm()
    # centre of the nominal arc: r0 inside both fitted edges
    s1, s2 = math.sqrt(1 + b_ * b_), math.sqrt(1 + d_ * d_)
    A = np.array([[-b_, 1.0], [1.0, -d_]])
    rhs = np.array([a_ + r0 * s1, c_ + r0 * s2])
    try:
        cx, cy = (float(v) for v in np.linalg.solve(A, rhs))
    except np.linalg.LinAlgError:
        out.reason = "could not follow the edges into the corner"
        return out

    rho = np.arange(max(0.0, r0 - 2.2), r0 + 1.0, step)
    thetas = np.linspace(math.pi * 1.06, math.pi * 1.44, 23)
    loss, wdepth, aligned, glare_hits = [], [], [], 0
    blend_px = max(1, int(round(0.12 / step)))
    span = int(round(0.7 / step))
    for th in thetas:
        ux = cx + rho * math.cos(th)
        vy = cy + rho * math.sin(th)
        px_ = P(ux)[None, :]
        py_ = P(vy)[None, :]
        prof = _sample(pc, px_, py_)[0]
        tprof = _sample(lt, px_, py_)[0]
        wprof = _sample(white, px_, py_)[0]
        kcut = _step_fit(prof)
        edge_rho = rho[min(kcut, len(rho) - 1)] if kcut < len(rho) else rho[-1] + step
        loss.append(r0 - edge_rho)
        if debug is not None:
            debug.setdefault("arc", []).append((cx + edge_rho * math.cos(th), cy + edge_rho * math.sin(th)))
            debug.setdefault("nominal", []).append((cx + r0 * math.cos(th), cy + r0 * math.sin(th)))
        if span <= kcut <= len(tprof) - span:
            aligned.append(tprof[kcut - span:kcut + span])
        # white run inward from the boundary
        run = 0
        for j in range(kcut - 1, -1, -1):
            if wprof[j] >= 0.5:
                run += 1
            elif run == 0 and kcut - 1 - j < blend_px:
                continue           # the boundary's blended pixels
            else:
                break
        wdepth.append(run * step)
        # glare: border well inside the boundary also white
        inner = wprof[max(0, kcut - int(1.6 / step)):max(0, kcut - int(0.9 / step))]
        if len(inner) and inner.mean() >= 0.4:
            glare_hits += 1
    loss = np.asarray(loss)
    wdepth = np.asarray(wdepth)
    # the blur of the corner's outline: the edge-spread of the profiles,
    # averaged across rays so print and background texture cancel
    blur = (_transition_width(np.mean(aligned, axis=0), step)
            if len(aligned) >= len(thetas) // 2 else float("inf"))
    out.blur_mm = blur if np.isfinite(blur) else 9.9
    if not np.isfinite(blur) or blur > fl["corner_max_blur_mm"]:
        out.reason = "the corner is too soft to place its outline"
        return out
    out.glare = glare_hits >= len(thetas) // 3
    # a chip is a few rays, rounding is the middle ones: the 75th percentile
    # catches both without letting one stray ray decide
    out.loss_mm = float(max(0.0, np.percentile(loss, 75)))
    out.sigma_mm = float(math.sqrt((0.6 / res) ** 2 + (0.3 * blur) ** 2 + 0.03 ** 2))
    out.whitening_resolvable = (models.white_vs_bg and not out.glare
                                and blur <= fl["whitening_max_blur_mm"])
    if out.whitening_resolvable:
        out.whitening_mm = float(np.percentile(wdepth, 75))
        frac = float((wdepth >= max(0.08, 1.0 / res)).mean())
        out.whitening_len_mm = frac * (math.pi / 2.0) * r0
    out.resolvable = True
    return out


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def _side_strip(canon: _Canon, side: str, lab: np.ndarray, out_mm: float, in_mm: float):
    """Normal profiles along one side: (profiles [n_pos, n_depth, ...], depth
    axis in mm (negative = outside), positions in mm along the side)."""
    R, M = canon.R, canon.M
    W, H = STANDARD_CARD_W_MM, STANDARD_CARD_H_MM
    length = W if side in ("top", "bottom") else H
    pos = np.arange(EDGE_END_MM, length - EDGE_END_MM + 1e-6, EDGE_STEP_MM)
    depth = np.arange(-out_mm, in_mm + 1e-9, 0.5 / R)
    if side == "top":
        X = pos[:, None] + 0 * depth[None, :]
        Y = 0 * pos[:, None] + depth[None, :]
    elif side == "bottom":
        X = pos[:, None] + 0 * depth[None, :]
        Y = H - depth[None, :] + 0 * pos[:, None]
    elif side == "left":
        X = 0 * pos[:, None] + depth[None, :]
        Y = pos[:, None] + 0 * depth[None, :]
    else:
        X = W - depth[None, :] + 0 * pos[:, None]
        Y = pos[:, None] + 0 * depth[None, :]
    xs = ((M + X) * R).astype(np.float32)
    ys = ((M + Y) * R).astype(np.float32)
    return _sample(lab, xs, ys), depth, pos


EDGE_SEGMENT_MM = 10.0
NICK_BASE_MM = 3.0


def _analyse_edge(canon: _Canon, side: str, inner_mm: dict, sf_mm: float) -> EdgeReading:
    fl = _floors()
    W, H = STANDARD_CARD_W_MM, STANDARD_CARD_H_MM
    box = {"top": (0, -1, W, 2), "bottom": (0, H - 2, W, H + 1),
           "left": (-1, 0, 2, H), "right": (W - 2, 0, W + 1, H)}[side]
    res = canon.res_at(*box)
    out = EdgeReading(side=side, resolvable=False, px_per_mm=res, side_face_mm=sf_mm)
    if res < fl["edge_min_px_per_mm"]:
        out.reason = f"{res:.1f} px/mm here; edges need {fl['edge_min_px_per_mm']:.0f}"
        return out
    bw = float(inner_mm.get(side, 2.5))
    deep = max(0.9, min(1.6, bw - 0.4))
    prof, depth, pos = _side_strip(canon, side, canon.lab, MARGIN_MM - 0.2, deep + 0.3)
    step = depth[1] - depth[0]
    d = lambda mm: int(np.argmin(np.abs(depth - mm)))
    n = len(pos)
    pc = np.zeros(prof.shape[:2], np.float32)
    lt = np.zeros(prof.shape[:2], np.float32)
    white = np.zeros(prof.shape[:2], bool)
    ok = np.zeros(n, bool)
    white_ok = np.zeros(n, bool)
    # Colour models per stretch of the side, not per side: wood grain, a
    # shadow across one end, and a metallic border's shading all change
    # along 88 mm.
    seg = max(4, int(round(EDGE_SEGMENT_MM / EDGE_STEP_MM)))
    for a in range(0, n, seg):
        b = min(n, a + seg)
        if n - b < seg // 2:
            b = n
        border = prof[a:b, d(0.45):d(deep)].reshape(-1, 3)
        bg = prof[a:b, d(-MARGIN_MM + 0.3):d(-0.7)].reshape(-1, 3)
        m = _models(border, bg)
        if m is not None and m.separable:
            pc[a:b] = m.p_card(prof[a:b])
            lt[a:b] = _linear_t(prof[a:b], m)
            white[a:b] = m.is_white(prof[a:b])
            ok[a:b] = True
            white_ok[a:b] = m.white_vs_bg
        if b == n:
            break
    if ok.mean() < 0.3:
        out.reason = "the background here looks like the border"
        return out
    idx = np.nonzero(ok)[0]
    lo, hi = d(-1.0), d(min(1.0, deep))
    cut = np.full(n, np.nan)
    widths = []
    for j in idx:
        segp = pc[j, lo:hi + 1][::-1]               # inside -> outside
        k = _step_fit(segp)
        cut[j] = depth[hi - k + 1] if k > 0 else depth[hi]
        segt = lt[j, lo:hi + 1][::-1]
        span = int(round(0.7 / step))
        if span <= k <= len(segt) - span:
            widths.append(segt[k - span:k + span])
    # edge-spread of the cut, averaged along the side
    blur = (_transition_width(np.mean(widths, axis=0), step)
            if len(widths) >= len(idx) // 2 else float("inf"))
    out.blur_mm = blur if np.isfinite(blur) else 9.9
    if not np.isfinite(blur) or blur > fl["edge_max_blur_mm"]:
        out.reason = "the edge is too soft to follow"
        return out
    # The cut is not a straight line in the flattened card: a lens's
    # distortion and a card that bows a little bend it by a tenth of a
    # millimetre or more over its length (the 2026-10-08 Probopass, a fresh
    # card, bows 0.14 mm along its top edge). A nick is a short departure, so
    # each position is compared with the cut's own course round it: a
    # running median over +-NICK_BASE_MM.
    half = max(2, int(round(NICK_BASE_MM / EDGE_STEP_MM)))
    Cu = cut[idx]
    base = np.array([np.median(Cu[max(0, i - half):i + half + 1]) for i in range(len(Cu))])
    resid = np.full(n, np.nan)
    resid[idx] = Cu - base                         # + = inward of the cut
    rv = resid[idx]
    noise = float(1.4826 * np.median(np.abs(rv - np.median(rv))))
    out.noise_mm = noise
    thr = max(fl["nick_min_mm"], 3.0 * noise, 1.0 / res)
    inward = np.nan_to_num(resid, nan=-1.0) > thr
    runs, j = [], 0
    while j < n:
        if inward[j]:
            k = j
            while k < n and inward[k]:
                k += 1
            runs.append((j, k))
            j = k
        else:
            j += 1
    out.nick_count = len(runs)
    out.nick_max_mm = float(max((np.nanmax(resid[a:b]) for a, b in runs), default=0.0))
    # wear: white fibre starting at the top surface's edge (past the side face)
    glare_pos = np.zeros(n, bool)
    whitened = np.zeros(n, bool)
    wmin = max(0.08, 1.0 / res)
    blend_px = max(1, int(round(max(0.12, blur) / step)))
    for jj in idx:
        start = cut[jj] + sf_mm
        a = d(start + 0.5 * step)
        run, skipped = 0, 0
        for t in range(a, min(len(depth) - 1, d(start + 0.8)) + 1):
            if white[jj, t]:
                run += 1
            elif run == 0 and skipped < blend_px:
                skipped += 1       # the boundary's blended pixels
            else:
                break
        whitened[jj] = run * step >= wmin
        deep_band = white[jj, d(cut[jj] + 0.9):d(cut[jj] + deep) + 1]
        glare_pos[jj] = deep_band.size > 0 and deep_band.mean() >= 0.4
    usable = ok & white_ok & ~glare_pos
    out.usable_frac = float(usable.mean())
    out.whitening_resolvable = out.usable_frac >= 0.3 and blur <= fl["whitening_max_blur_mm"]
    if out.whitening_resolvable:
        out.whitening_frac = float(whitened[usable].mean())
    out.resolvable = True
    return out


# ---------------------------------------------------------------------------
# Surface
# ---------------------------------------------------------------------------


def _analyse_surface(canon: _Canon) -> SurfaceReading:
    fl = _floors()
    W, H = STANDARD_CARD_W_MM, STANDARD_CARD_H_MM
    res = canon.res_at(1.0, 1.0, W - 1.0, H - 1.0)
    Rs = float(min(canon.R, fl["surface_work_px_per_mm"]))
    out = SurfaceReading(resolvable=False, px_per_mm=res)
    # the card itself, flat, at Rs
    x0, y0 = int(round(canon.M * canon.R)), int(round(canon.M * canon.R))
    x1 = int(round((canon.M + W) * canon.R))
    y1 = int(round((canon.M + H) * canon.R))
    card = canon.bgr[y0:y1, x0:x1]
    sw, sh = int(round(W * Rs)), int(round(H * Rs))
    card = cv2.resize(card, (sw, sh), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY).astype(np.float32)
    s, v = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    glare = (v >= 245) | ((s <= 45) & (v >= 228))
    # keep away from the cut: the edge band belongs to edges/corners
    inset = int(round(1.0 * Rs))
    keep = np.zeros_like(glare)
    keep[inset:-inset, inset:-inset] = True
    glare &= keep
    glare = cv2.morphologyEx(glare.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8))
    # A scratch is a thin DARK line through the highlight, so it is a hole
    # in the bright mask; close the mask over holes up to ~1 mm across or
    # the very thing being looked for is cut out of the region searched.
    kc = 2 * int(round(0.6 * Rs)) + 1
    glare = cv2.morphologyEx(glare, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))) > 0
    glare &= keep
    out.glare_frac = float(glare.mean())
    cell = SURFACE_CELL_MM * Rs
    gy, gx = int(math.ceil(sh / cell)), int(math.ceil(sw / cell))
    insp = np.zeros((gy, gx), bool)
    for r in range(gy):
        for c in range(gx):
            blk = glare[int(r * cell):int((r + 1) * cell), int(c * cell):int((c + 1) * cell)]
            insp[r, c] = blk.size > 0 and blk.mean() >= 0.35
    out.inspected = insp
    # what the print looks like without a reflection: glare masked out
    diffuse_mask = ~cv2.dilate(glare.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    out.diffuse = gray.astype(np.uint8)
    out.diffuse_mask = diffuse_mask & keep
    if res < fl["surface_min_px_per_mm"]:
        out.reason = f"{res:.1f} px/mm on the face; surface needs {fl['surface_min_px_per_mm']:.0f}"
        return out
    if out.glare_frac < 0.01:
        out.reason = "no highlight on the card in this view"
        out.resolvable = True
        return out
    # thin dark structures inside the highlight
    bg = cv2.medianBlur(gray.astype(np.uint8), 2 * int(0.6 * Rs) + 1).astype(np.float32)
    resid = gray - bg
    inside = cv2.erode(glare.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if inside.sum() < 50:
        out.resolvable = True
        out.reason = "highlight too small to read"
        return out
    vals = resid[inside]
    noise = float(1.4826 * np.median(np.abs(vals - np.median(vals)))) + 1.0
    dark = (resid < -max(12.0, 4.0 * noise)) & inside
    out.texture = float(dark.mean() / max(inside.mean(), 1e-9))
    if out.texture > fl["surface_texture_max"]:
        out.resolvable = True
        out.reason = "the highlight is full of structure (foil): scratches can't be told from it"
        out.candidates = []
        return out
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    cands = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 3:
            continue
        ys_, xs_ = np.nonzero(lbl == i)
        pts = np.column_stack([xs_, ys_]).astype(np.float64)
        c = pts.mean(axis=0)
        cov = np.cov((pts - c).T) if len(pts) > 2 else np.eye(2)
        evals, evecs = np.linalg.eigh(cov)
        major = evecs[:, int(np.argmax(evals))]
        proj = (pts - c) @ major
        L = float(proj.max() - proj.min() + 1.0)
        Wd = float(max(1.0, 4.0 * math.sqrt(max(float(evals.min()), 0.0))))
        if L / Rs < fl["surface_min_line_mm"] or L / Wd < 3.0:
            continue
        # (x_mm, y_mm, length_mm, dx, dy): centre, length and unit direction
        cands.append((float(c[0] / Rs), float(c[1] / Rs), float(L / Rs),
                      float(major[0]), float(major[1])))
    out.candidates = cands
    out.resolvable = True
    return out


# ---------------------------------------------------------------------------
# One view
# ---------------------------------------------------------------------------


def image_px_per_mm(corners_px) -> float:
    """The card's mean scale in its image, from its outline. Not
    ``CenteringResult.px_per_mm``: that is the RECTIFICATION scale, clipped
    to at least 6 px/mm, and a 4.8 px/mm live frame must not pass a 6 px/mm
    floor on the strength of it."""
    q = np.asarray(corners_px, np.float64).reshape(4, 2)
    w = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    h = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    return float(0.5 * (w / STANDARD_CARD_W_MM + h / STANDARD_CARD_H_MM))


def analyse_view(
    image: np.ndarray,
    corners_px: np.ndarray,
    px_per_mm: Optional[float] = None,
    inner_rect_mm: Optional[tuple] = None,
    capture: Optional[CaptureSpec] = None,
    detail_mask: Optional[np.ndarray] = None,
    detail_upsample: Optional[float] = None,
) -> ConditionView:
    """Read corners, edges and surface off one image of a located card.

    ``corners_px`` is the card's outer outline (TL, TR, BR, BL), as
    ``measure_centering`` returns it in ``corners_px``. The card's scale is
    taken from that outline (``px_per_mm``, if given, can only lower it).
    ``detail_mask`` marks the pixels that came at full resolution when the
    rest of ``image`` was upsampled by ``detail_upsample`` (the live edge
    strips over the tracking frame). ``inner_rect_mm`` (left, top, right,
    bottom of the printed frame, mm) sets how deep the border can be sampled.
    """
    ppm = image_px_per_mm(corners_px)
    if px_per_mm is not None and px_per_mm > 0:
        ppm = min(ppm, float(px_per_mm))
    other = None
    if detail_mask is not None and detail_upsample is not None and detail_upsample > 1.0:
        other = ppm / float(detail_upsample)
    canon = _canonical(image, corners_px, ppm, detail_mask, other)
    W, H = STANDARD_CARD_W_MM, STANDARD_CARD_H_MM
    if inner_rect_mm is not None:
        l, t, r, b = (float(v) for v in inner_rect_mm)
        inner = {"left": l, "top": t, "right": W - r, "bottom": H - b}
    else:
        inner = {s: 2.5 for s in SIDES}
    inner = {k: float(min(6.0, max(0.8, v))) for k, v in inner.items()}
    face, face_c = classify_face(canon.bgr, canon.R, canon.M)
    sf, tilt = side_face_mm(corners_px, capture, image.shape)
    corners = {name: _analyse_corner(canon, k, name, inner) for k, name in enumerate(CORNERS)}
    edges = {s: _analyse_edge(canon, s, inner, sf[s]) for s in SIDES}
    surface = _analyse_surface(canon)
    return ConditionView(face=face, face_confidence=face_c, px_per_mm=float(ppm),
                         corners=corners, edges=edges, surface=surface, tilt_deg=tilt)
