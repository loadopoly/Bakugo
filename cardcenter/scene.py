"""Find cards the way a shop actually shows them.

Field frames from a phone at a card shop (tests/fixtures/field) look nothing
like a card on a plain mat: cards lie on a wood counter at 20-40 degrees of
tilt, in sleeves, two or three to a frame, overlapping, a finger over an edge,
motion-soft. On those frames the threshold-and-contour detector
(geometry.find_card_quad) found the card under the reticle in 0 of 17 checks:
the card's outer edge -- silver or a sleeve against light wood -- is the
WEAKEST edge in the picture, far weaker than the artwork inside it, so no
single threshold gives a closed contour around the card.

This module does not need a closed contour. It works from straight edges:

1. Line segments (LSD) on the L, a and b channels. A silver border on tan wood
   is faint in brightness and clear in colour, so colour channels matter.
2. Around an aim point (the reticle, or where the card was last seen), every
   pair of roughly parallel lines with the aim point between them is one axis
   of a possible card, and two such pairs at a large angle make a quad.
3. Each quad is judged on what the picture says about it, not on how it was
   found:
   * edge support: along each side, the fraction of the side where there is an
     edge ACROSS the side with a consistent polarity (card brighter than the
     table, or darker, all the way along). Artwork texture crossing a line has
     random polarity; a real boundary does not. Three well-supported sides are
     enough, so a finger or an overlapping card over one side is survivable.
   * shape through the perspective: the quad is un-projected with a plausible
     focal length and must be a 63.5 x 88.9 mm rectangle (aspect 1.40). A card
     at 40 degrees of tilt looks 1.1 in the image and is still accepted; a
     1.1 rectangle seen square-on is not.
4. Nested candidates (the printed frame inside the card, the card inside its
   sleeve) all pass; the outermost one that is nearly as well supported as the
   best wins, which is the rule find_card_quad already uses.

The result says how much of each side was seen (``complete`` needs all four),
the tilt of the card, and the aspect recovered, so the caller can decide what
to do with a card it can identify but should not measure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from .geometry import enforce_portrait, order_quad
from .types import STANDARD_CARD_H_MM, STANDARD_CARD_W_MM, DetectionError

CARD_ASPECT = STANDARD_CARD_H_MM / STANDARD_CARD_W_MM      # 1.40
ASPECT_TOL = 0.16                                          # sleeves, focal doubt
WORK_LONG_SIDE = 720
NOMINAL_FOV_DEG = 55.0          # across the frame width; phones sit ~45-70
FOCAL_FACTORS = (0.7, 1.0, 1.45)
MAX_TILT_DEG = 62.0
MIN_SIDE_FRAC = 0.045           # of the frame's short side
SIDE_OK = 0.5                   # a side counts as seen at this support
PAIR_ANGLE_MAX = math.radians(28.0)
CROSS_ANGLE_MIN = math.radians(48.0)
# How far a card's tilt may sit from what the phone's own tilt says before it
# costs a card candidate its place (see SceneSearch.expected_tilt_deg).
TILT_PRIOR_DEG = 25.0
CHROMA_SIGMA = 1.0
RUNS_OFF = "whole card not in view -- lift the phone until all four edges of the card show"


@dataclass(frozen=True)
class CardHit:
    """One card found in a frame (source pixels)."""

    quad: np.ndarray                 # TL, TR, BR, BL; edge 0-1 is the short side
    side_support: tuple              # per side, 0..1
    support: float                   # mean of the three best-seen sides
    aspect: float                    # long/short, recovered through the perspective
    tilt_deg: float                  # of the card plane to the image plane
    complete: bool                   # every side seen (no occlusion, not cut off)
    score: float
    residual_px: float = 1.0         # rms scatter of the edge points about the sides

    @property
    def centre(self) -> np.ndarray:
        return self.quad.mean(axis=0)


# --------------------------------------------------------------------------
# edges and lines


def _gradients(img: np.ndarray):
    """Colour gradient: per pixel, the Lab channel with the strongest edge
    (chroma weighted up -- its range is a third of L's)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.0)
    # Chroma is stored at half resolution in a phone JPEG (4:2:0). More
    # smoothing than luma was tried (1.6, 2.2): no change on the live path
    # (540 px, quality 0.75 and 0.92), worse at full frame size.
    lab[..., 1:] = cv2.GaussianBlur(lab[..., 1:], (0, 0), CHROMA_SIGMA)
    best_gx = best_gy = best_m = None
    for c, wgt in ((0, 1.0), (1, 2.2), (2, 2.2)):
        gx = cv2.Sobel(lab[..., c], cv2.CV_32F, 1, 0, ksize=3) * wgt
        gy = cv2.Sobel(lab[..., c], cv2.CV_32F, 0, 1, ksize=3) * wgt
        m = gx * gx + gy * gy
        if best_m is None:
            best_gx, best_gy, best_m = gx, gy, m
        else:
            take = m > best_m
            best_gx = np.where(take, gx, best_gx)
            best_gy = np.where(take, gy, best_gy)
            best_m = np.where(take, m, best_m)
    return best_gx, best_gy, np.sqrt(best_m), lab


def _segments(lab: np.ndarray, min_len: float) -> np.ndarray:
    """LSD on each Lab channel; rows (x1, y1, x2, y2)."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    out = []
    for c in range(3):
        ch = lab[..., c]
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        u8 = np.clip((ch - lo) * (255.0 / max(hi - lo, 1e-3)), 0, 255).astype(np.uint8)
        lines = lsd.detect(u8)[0]
        if lines is not None:
            out.append(lines.reshape(-1, 4))
    if not out:
        return np.zeros((0, 4), np.float32)
    seg = np.concatenate(out)
    ln = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    return seg[ln >= min_len]


class _LineSet:
    """Merged straight lines with the stretches of each that carry segments.

    Line i: x cos(theta_i) + y sin(theta_i) = rho_i, theta in [0, pi).
    Position along it is t = -x sin(theta) + y cos(theta). ``cum[i]`` is the
    cumulative segment coverage over t in 1 px bins from -D to D, so the part
    of any stretch [t1, t2] that has evidence is two lookups.
    """

    def __init__(self, seg: np.ndarray, diag: float, ang_tol=math.radians(2.5), off_tol=2.5):
        self.D = int(math.ceil(diag)) + 2
        if len(seg) == 0:
            self.theta = np.zeros(0)
            self.rho = np.zeros(0)
            self.length = np.zeros(0)
            self.cum = np.zeros((0, 2 * self.D + 2), np.float32)
            return
        dx, dy = seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]
        length = np.hypot(dx, dy)
        theta = (np.arctan2(dy, dx) + math.pi / 2) % math.pi
        mx, my = (seg[:, 0] + seg[:, 2]) / 2, (seg[:, 1] + seg[:, 3]) / 2
        rho = mx * np.cos(theta) + my * np.sin(theta)
        order = np.argsort(-length)
        groups: list[list] = []          # [theta, rho, [segment indices]]
        for i in order:
            t, r = float(theta[i]), float(rho[i])
            placed = False
            for g in groups:
                dt = abs(t - g[0])
                if dt > math.pi / 2:
                    dt, r_cmp = math.pi - dt, -r
                else:
                    r_cmp = r
                if dt < ang_tol and abs(r_cmp - g[1]) < off_tol:
                    g[2].append(i)
                    placed = True
                    break
            if not placed:
                groups.append([t, r, [i]])
        n = len(groups)
        self.theta = np.array([g[0] for g in groups])
        self.rho = np.array([g[1] for g in groups])
        self.cum = np.zeros((n, 2 * self.D + 2), np.float32)
        self.length = np.zeros(n)
        for k, g in enumerate(groups):
            th = g[0]
            cov = np.zeros(2 * self.D + 1, np.float32)
            sn, cs = math.sin(th), math.cos(th)
            for i in g[2]:
                t1 = -seg[i, 0] * sn + seg[i, 1] * cs
                t2 = -seg[i, 2] * sn + seg[i, 3] * cs
                a, b = sorted((t1, t2))
                ia = int(np.clip(math.floor(a) + self.D, 0, 2 * self.D))
                ib = int(np.clip(math.ceil(b) + self.D, 0, 2 * self.D))
                cov[ia:ib + 1] = 1.0
            self.cum[k, 1:] = np.cumsum(cov)
            self.length[k] = float(cov.sum())

    def __len__(self):
        return len(self.theta)

    def along(self, idx, pts):
        """t of points (N,2) on lines idx (N,)."""
        th = self.theta[idx]
        return -pts[:, 0] * np.sin(th) + pts[:, 1] * np.cos(th)

    def covered(self, idx, t1, t2):
        """Covered length of lines idx over [t1, t2] (arrays)."""
        a = np.clip(np.floor(np.minimum(t1, t2)).astype(int) + self.D, 0, 2 * self.D + 1)
        b = np.clip(np.ceil(np.maximum(t1, t2)).astype(int) + self.D, 0, 2 * self.D + 1)
        return self.cum[idx, b] - self.cum[idx, a]


# --------------------------------------------------------------------------
# geometry


def _intersect(l1, l2):
    t1, r1 = l1[0], l1[1]
    t2, r2 = l2[0], l2[1]
    a = np.array([[math.cos(t1), math.sin(t1)], [math.cos(t2), math.sin(t2)]])
    det = np.linalg.det(a)
    if abs(det) < 1e-6:
        return None
    return np.linalg.solve(a, np.array([r1, r2]))


def card_pose(quad: np.ndarray, shape, fov_deg: float = NOMINAL_FOV_DEG):
    """(aspect long/short, tilt_deg) of the rectangle behind ``quad``.

    Un-projects with K from the field of view: for the homography H taking the
    unit square to the quad, K^-1 H = [s w r1, s h r2, s t], so w/h is the
    ratio of the first two column norms and the plane normal is r1 x r2. The
    focal length is not known well on a phone (the frame is a crop of a crop),
    so a small set of focal lengths is tried and the aspect closest to a card
    is kept -- at small tilt they all agree, and at large tilt the right one
    is the one that makes a card.
    """
    h, w = shape[:2]
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    try:
        H = cv2.getPerspectiveTransform(src.astype(np.float32), q.astype(np.float32))
    except cv2.error:
        return None
    f0 = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    best = None
    for k in FOCAL_FACTORS:
        f = f0 * k
        Kinv = np.array([[1 / f, 0, -w / (2 * f)], [0, 1 / f, -h / (2 * f)], [0, 0, 1]])
        M = Kinv @ H
        a1, a2 = M[:, 0], M[:, 1]
        n1, n2 = np.linalg.norm(a1), np.linalg.norm(a2)
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        aspect = max(n1, n2) / min(n1, n2)
        normal = np.cross(a1 / n1, a2 / n2)
        nn = np.linalg.norm(normal)
        if nn < 1e-9:
            continue
        tilt = math.degrees(math.acos(min(1.0, abs(normal[2]) / nn)))
        err = abs(aspect - CARD_ASPECT)
        if best is None or err < best[0]:
            best = (err, aspect, tilt)
    if best is None:
        return None
    return best[1], best[2]


def card_pose_batch(Q: np.ndarray, shape, fov_deg: float = NOMINAL_FOV_DEG,
                    expected_tilt_deg: Optional[float] = None):
    """card_pose for quads Q (N,4,2): arrays (aspect, tilt_deg); nan where
    degenerate. Square-to-quad homography in closed form (Heckbert).

    With ``expected_tilt_deg`` the focal length is chosen to agree with both
    the card's aspect and the expected tilt, not the aspect alone."""
    h, w = shape[:2]
    x, y = Q[..., 0], Q[..., 1]
    dx1, dx2, dx3 = x[:, 1] - x[:, 2], x[:, 3] - x[:, 2], x[:, 0] - x[:, 1] + x[:, 2] - x[:, 3]
    dy1, dy2, dy3 = y[:, 1] - y[:, 2], y[:, 3] - y[:, 2], y[:, 0] - y[:, 1] + y[:, 2] - y[:, 3]
    det = dx1 * dy2 - dx2 * dy1
    det = np.where(np.abs(det) < 1e-9, 1e-9, det)
    g = (dx3 * dy2 - dx2 * dy3) / det
    hh = (dx1 * dy3 - dx3 * dy1) / det
    H = np.zeros((len(Q), 3, 3))
    H[:, 0, 0] = x[:, 1] - x[:, 0] + g * x[:, 1]
    H[:, 0, 1] = x[:, 3] - x[:, 0] + hh * x[:, 3]
    H[:, 0, 2] = x[:, 0]
    H[:, 1, 0] = y[:, 1] - y[:, 0] + g * y[:, 1]
    H[:, 1, 1] = y[:, 3] - y[:, 0] + hh * y[:, 3]
    H[:, 1, 2] = y[:, 0]
    H[:, 2, 0] = g
    H[:, 2, 1] = hh
    H[:, 2, 2] = 1.0
    f0 = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    best_err = np.full(len(Q), np.inf)
    best_asp = np.full(len(Q), np.nan)
    best_tilt = np.full(len(Q), np.nan)
    for k in FOCAL_FACTORS:
        f = f0 * k
        Kinv = np.array([[1 / f, 0, -w / (2 * f)], [0, 1 / f, -h / (2 * f)], [0, 0, 1.0]])
        M = Kinv[None] @ H
        a1, a2 = M[:, :, 0], M[:, :, 1]
        n1 = np.linalg.norm(a1, axis=1)
        n2 = np.linalg.norm(a2, axis=1)
        good = (n1 > 1e-12) & (n2 > 1e-12)
        asp = np.maximum(n1, n2) / np.maximum(np.minimum(n1, n2), 1e-12)
        nrm = np.cross(a1 / np.maximum(n1, 1e-12)[:, None], a2 / np.maximum(n2, 1e-12)[:, None])
        nn = np.linalg.norm(nrm, axis=1)
        good &= nn > 1e-9
        tl = np.degrees(np.arccos(np.clip(np.abs(nrm[:, 2]) / np.maximum(nn, 1e-12), 0, 1)))
        err = np.abs(asp - CARD_ASPECT) / ASPECT_TOL
        if expected_tilt_deg is not None:
            err = err + np.abs(tl - expected_tilt_deg) / TILT_PRIOR_DEG
        err = np.where(good, err, np.inf)
        take = err < best_err
        best_err = np.where(take, err, best_err)
        best_asp = np.where(take, asp, best_asp)
        best_tilt = np.where(take, tl, best_tilt)
    return best_asp, best_tilt


def _is_window_of(inner: np.ndarray, outer: np.ndarray) -> bool:
    """Does ``inner`` sit inside ``outer`` the way the art of a card sits in
    the card?

    Aimed at the artwork, the best-supported quad is often the art window
    (clean printed lines, and on Pokemon and Magic cards itself a 1.4
    rectangle turned sideways), or the window plus the card's own top edge.
    The card around it is the answer. The layout is what identifies it: the
    art spans the card's width inside the border, starts near one end (under
    the name bar) and stops well short of the other (the text box). Measured
    on the field frames the art-side quads sat at 0.0-0.08 from the long
    sides, 0.0-0.05 from the near end and 0.50-0.53 from the far end.

    Two cards stacked with aligned sides also make a bigger card-shaped quad
    around the top one, but the top card reaches the far end of that quad
    (it IS its lower part): its far-end inset is ~0.1 where the art's is 0.5,
    so this says no. Measured on desk_spread: 0.07-0.11 near, 0.25-0.36 far.
    """
    src = np.asarray(outer, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    try:
        M = cv2.getPerspectiveTransform(src, dst)
    except cv2.error:
        return False
    p = cv2.perspectiveTransform(np.asarray(inner, np.float32).reshape(-1, 1, 2), M).reshape(-1, 2)
    left, right = float(p[:, 0].min()), 1.0 - float(p[:, 0].max())
    top, bottom = float(p[:, 1].min()), 1.0 - float(p[:, 1].max())
    near, far = min(top, bottom), max(top, bottom)
    return (-0.08 <= left <= 0.15 and -0.08 <= right <= 0.15
            and -0.08 <= near <= 0.2 and 0.38 <= far <= 0.7)


def _convex(q: np.ndarray) -> bool:
    s = 0
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cr) < 1e-9:
            return False
        sg = 1 if cr > 0 else -1
        if s == 0:
            s = sg
        elif sg != s:
            return False
    return True


# --------------------------------------------------------------------------
# evidence


class _Evidence:
    """Signed edge strength across a line, sampled quickly."""

    def __init__(self, gx, gy, mag):
        self.gx, self.gy = gx, gy
        self.h, self.w = gx.shape
        # an edge worth the name: a fraction of the strong edges in the frame,
        # with a floor so a flat frame does not turn noise into support
        p95 = float(np.percentile(mag, 95))
        self.thresh = max(0.22 * p95, 6.0)

    def side_profile(self, a, b, n_samples=28, t0=0.12, t1=0.88, reach=(-1.5, -0.75, 0.0, 0.75, 1.5)):
        """Signed normal gradient at samples along a->b, max |.| over the
        small normal offsets. Returns (values, inside_frame_mask)."""
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        d = b - a
        L = float(np.hypot(*d))
        if L < 1e-6:
            return np.zeros(0), np.zeros(0, bool)
        u = d / L
        nrm = np.array([-u[1], u[0]])
        ts = np.linspace(t0, t1, n_samples)
        base = a[None, :] + ts[:, None] * d[None, :]
        best = np.zeros(n_samples)
        inside = np.ones(n_samples, bool)
        for off in reach:
            p = base + off * nrm[None, :]
            xi = np.round(p[:, 0]).astype(int)
            yi = np.round(p[:, 1]).astype(int)
            ok = (xi >= 0) & (xi < self.w) & (yi >= 0) & (yi < self.h)
            if off == 0.0:
                inside = ok
            xi = np.clip(xi, 0, self.w - 1)
            yi = np.clip(yi, 0, self.h - 1)
            v = self.gx[yi, xi] * nrm[0] + self.gy[yi, xi] * nrm[1]
            v = np.where(ok, v, 0.0)
            best = np.where(np.abs(v) > np.abs(best), v, best)
        return best, inside

    def side_support(self, a, b, whole: bool = False) -> float:
        v, inside = (self.side_profile(a, b, n_samples=8, t0=0.0, t1=1.0) if whole
                     else self.side_profile(a, b))
        if len(v) == 0 or inside.mean() < 0.6:
            return 0.0
        strong = np.abs(v) >= self.thresh
        if not strong.any():
            return 0.0
        pos = int((strong & (v > 0)).sum())
        neg = int((strong & (v < 0)).sum())
        # the boundary keeps one polarity all the way along
        return max(pos, neg) / float(len(v))


# --------------------------------------------------------------------------
# search


def _resize(img, long_side):
    h, w = img.shape[:2]
    s = min(1.0, long_side / float(max(h, w)))
    if s < 1.0:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return img, s


class SceneSearch:
    """Precomputed edges and lines for one frame; ask it for cards."""

    MAX_LINES = 80
    MAX_PAIRS = 160
    MAX_JUDGED = 400

    def __init__(self, image: np.ndarray, fov_deg: float = NOMINAL_FOV_DEG,
                 long_side: int = WORK_LONG_SIDE, expected_tilt_deg: Optional[float] = None):
        """``expected_tilt_deg``: the tilt a card lying on the counter would
        have, from the phone's gyroscope (how far the camera looks away from
        straight down). When the phone is looking down at a counter, the
        cards on it are at that tilt -- a candidate that is only card-shaped
        at some other tilt is less likely to be a card. Two cards stacked
        with aligned sides make a quad that is card-shaped face-on; the top
        card alone is card-shaped at the counter's tilt, and this is what
        tells them apart. None when the phone is held up (a card held facing
        it, or a wall display): no prior. Measured on the field frames with
        the tilts the phone showed, it did not help (153 of 168 aim points
        against 156 without), so the app does not send it yet; it is here
        for when there is a larger set to decide on."""
        self.expected_tilt = expected_tilt_deg
        # why the last card_at() said no, when it knows better than "nothing"
        self.refusal = ""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.src_shape = image.shape[:2]
        self.img, self.scale = _resize(image, long_side)
        self.h, self.w = self.img.shape[:2]
        self.fov = fov_deg
        gx, gy, mag, lab = _gradients(self.img)
        self.lab = lab
        self.ev = _Evidence(gx, gy, mag)
        short = min(self.h, self.w)
        self.min_side = MIN_SIDE_FRAC * short
        self._lines = None

    @property
    def lines(self) -> "_LineSet":
        """Built on first use: judging a given outline (the live tracker, every
        frame) needs only the gradients, and the lines are two thirds of the
        cost of a search."""
        if self._lines is None:
            self._lines = _LineSet(_segments(self.lab, max(8.0, 0.4 * self.min_side)),
                                   math.hypot(self.h, self.w))
        return self._lines

    def _candidates(self, p):
        """Quads around p, pre-ranked by how much of each side has segments."""
        L = self.lines
        if len(L) < 4:
            return []
        px, py = float(p[0]), float(p[1])
        ct, st = np.cos(L.theta), np.sin(L.theta)
        d = ct * px + st * py - L.rho
        tp = -px * st + py * ct
        R = 2.2 * np.abs(d) + self.min_side
        cov = L.covered(np.arange(len(L)), tp - R, tp + R)
        keep = (np.abs(d) >= 0.25 * self.min_side) & (np.abs(d) <= 1.15 * max(self.h, self.w))
        keep &= cov >= 0.6 * self.min_side
        idx = np.nonzero(keep)[0]
        if len(idx) < 4:
            return []
        idx = idx[np.argsort(-cov[idx])][: self.MAX_LINES]
        # pairs: nearly parallel, p between them, far enough apart
        I, J = np.triu_indices(len(idx), 1)
        li, lj = idx[I], idx[J]
        dt = np.abs(L.theta[li] - L.theta[lj])
        flip = dt > math.pi / 2
        dt = np.where(flip, math.pi - dt, dt)
        dj = np.where(flip, -d[lj], d[lj])
        ok = (dt <= PAIR_ANGLE_MAX) & (d[li] * dj < 0)
        ok &= (np.abs(d[li]) + np.abs(d[lj])) >= self.min_side
        li, lj = li[ok], lj[ok]
        if len(li) < 2:
            return []
        wgt = cov[li] + cov[lj]
        order = np.argsort(-wgt)[: self.MAX_PAIRS]
        li, lj = li[order], lj[order]
        # pair direction: theta of the first line (the two are within 28 deg)
        pt = L.theta[li]
        A, B = np.triu_indices(len(li), 1)
        cross = np.abs(pt[A] - pt[B]) % math.pi
        cross = np.minimum(cross, math.pi - cross)
        ok = cross >= CROSS_ANGLE_MIN
        A, B = A[ok], B[ok]
        if len(A) == 0:
            return []
        a1, a2, b1, b2 = li[A], lj[A], li[B], lj[B]

        def meet(u, v):
            tu, tv = L.theta[u], L.theta[v]
            det = np.sin(tv - tu)
            det = np.where(np.abs(det) < 1e-9, 1e-9, det)
            x = (L.rho[u] * np.sin(tv) - L.rho[v] * np.sin(tu)) / det
            y = (L.rho[v] * np.cos(tu) - L.rho[u] * np.cos(tv)) / det
            return np.stack([x, y], axis=1)

        C = np.stack([meet(a1, b1), meet(a1, b2), meet(a2, b2), meet(a2, b1)], axis=1)  # N,4,2
        m = 0.02 * max(self.h, self.w)
        inb = ((C[..., 0] >= -m) & (C[..., 0] <= self.w + m)
               & (C[..., 1] >= -m) & (C[..., 1] <= self.h + m)).all(axis=1)
        sides = np.linalg.norm(np.roll(C, -1, axis=1) - C, axis=2)
        inb &= sides.min(axis=1) >= self.min_side
        C, a1, a2, b1, b2, sides = C[inb], a1[inb], a2[inb], b1[inb], b2[inb], sides[inb]
        if len(C) == 0:
            return []
        # segment coverage of each side, ends trimmed for the rounded corners
        side_lines = [a1, b2, a2, b1]
        covs = []
        for k in range(4):
            c1, c2 = C[:, k], C[:, (k + 1) % 4]
            s1 = c1 + 0.12 * (c2 - c1)
            s2 = c1 + 0.88 * (c2 - c1)
            ln = side_lines[k]
            t1, t2 = L.along(ln, s1), L.along(ln, s2)
            covs.append(L.covered(ln, t1, t2) / np.maximum(np.abs(t2 - t1), 1.0))
        covs = np.sort(np.stack(covs, axis=1), axis=1)[:, ::-1]
        good = covs[:, 2] >= 0.3
        C, covs = C[good], covs[good]
        rank = np.argsort(-(covs[:, :3].sum(axis=1) + 0.5 * covs[:, 3]))[: self.MAX_JUDGED]
        return [C[r] for r in rank]

    def _judge(self, q) -> Optional[CardHit]:
        hits = self._judge_batch([q])
        return hits[0]

    def _profiles(self, A, B, n, t0, t1, reach=(-1.5, -0.75, 0.0, 0.75, 1.5)):
        """Signed normal gradient along segments A->B (M,2), max |.| over
        small normal offsets. Returns (values (M,n), inside (M,n))."""
        d = B - A
        L = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
        u = d / L[:, None]
        nrm = np.stack([-u[:, 1], u[:, 0]], axis=1)
        ts = np.linspace(t0, t1, n)
        base = A[:, None, :] + ts[None, :, None] * d[:, None, :]            # M,n,2
        best = np.zeros(base.shape[:2])
        inside = None
        gx, gy, W, H = self.ev.gx, self.ev.gy, self.ev.w, self.ev.h
        for off in reach:
            pt = base + off * nrm[:, None, :]
            xi = np.round(pt[..., 0]).astype(np.int64)
            yi = np.round(pt[..., 1]).astype(np.int64)
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            if off == 0.0:
                inside = ok
            xi = np.clip(xi, 0, W - 1)
            yi = np.clip(yi, 0, H - 1)
            v = gx[yi, xi] * nrm[:, None, 0] + gy[yi, xi] * nrm[:, None, 1]
            v = np.where(ok, v, 0.0)
            best = np.where(np.abs(v) > np.abs(best), v, best)
        return best, inside

    def _support(self, A, B, n=28, t0=0.12, t1=0.88):
        v, inside = self._profiles(A, B, n, t0, t1)
        strong = np.abs(v) >= self.ev.thresh
        pos = (strong & (v > 0)).sum(axis=1)
        neg = (strong & (v < 0)).sum(axis=1)
        sup = np.maximum(pos, neg) / float(n)
        return np.where(inside.mean(axis=1) < 0.6, 0.0, sup)

    def _face_differs(self, Q: np.ndarray, max_background_frac: float = 0.55) -> np.ndarray:
        """For quads Q (N,4,2): is the middle of the quad unlike what lies just
        outside it? Samples a 7x7 grid over the central 60% of the face
        (through the quad's own perspective) and, for each side separately, 9
        points 8% of the side's length outside it. A face point counts as
        background when it is within that side's own spread of that side's
        median colour (Lab); if most of the face matches what is outside any
        one side, the quad is a patch of counter, not a card. Per side,
        because outside a real quad on a counter there is wood on one side,
        the next card on another and the floor past the counter's edge."""
        N = len(Q)
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        g = np.linspace(0.2, 0.8, 7)
        uu, vv = np.meshgrid(g, g)
        core_uv = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], axis=1)        # 49,3
        t = np.linspace(0.12, 0.88, 9)
        out = np.ones(N, bool)
        H, W = self.lab.shape[:2]

        def at(pts):
            xi = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
            yi = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
            return self.lab[yi, xi]

        for i in range(N):
            q = Q[i].astype(np.float32)
            try:
                M = cv2.getPerspectiveTransform(src, q)
            except cv2.error:
                continue
            core = core_uv @ M.T
            cv = at(core[:, :2] / core[:, 2:3])
            c = q.mean(axis=0)
            worst = 0.0
            for k in range(4):
                a, b = q[k], q[(k + 1) % 4]
                d = b - a
                n = np.array([-d[1], d[0]], np.float32)
                n /= max(float(np.hypot(*n)), 1e-6)
                if np.dot(n, (a + b) / 2 - c) < 0:
                    n = -n
                L = float(np.hypot(*d))
                ring = a[None, :] + t[:, None] * d[None, :] + 0.08 * L * n[None, :]
                inside = (ring[:, 0] >= 0) & (ring[:, 0] < W) & (ring[:, 1] >= 0) & (ring[:, 1] < H)
                if inside.sum() < 5:
                    continue
                rv = at(ring[inside])
                bg = np.median(rv, axis=0)
                spread = float(np.median(np.linalg.norm(rv - bg, axis=1)))
                thr = min(max(8.0, 2.0 * spread), 16.0)
                worst = max(worst, float((np.linalg.norm(cv - bg, axis=1) <= thr).mean()))
            out[i] = worst <= max_background_frac
        return out

    def _judge_batch(self, quads) -> list:
        """Judge many quads at once; None for the ones that are not cards.

        * Shape through the perspective (card_pose, batched).
        * Edge support of each side: fraction with a consistent-polarity edge.
        * Corners must be where the edges STOP. Just past each corner, along
          each side's own line, there should be no edge of the same polarity:
          a quad cut out of longer lines (half a card, a card plus the table
          edge, a row of cards) has edges running straight through its
          "corners". Sampled 3-11% of the side beyond the corner: past the
          rounded corner, and short of the next card in a binder row.
        """
        if len(quads) == 0:
            return []
        Q = np.array([enforce_portrait(order_quad(np.asarray(q, dtype=np.float64))) for q in quads])
        N = len(Q)
        aspect, tilt = card_pose_batch(Q, (self.h, self.w), self.fov, self.expected_tilt)
        A = Q.reshape(-1, 2)
        B = np.roll(Q, -1, axis=1).reshape(-1, 2)
        sup = self._support(A, B).reshape(N, 4)
        D = B - A
        o_fwd = self._support(B + 0.03 * D, B + 0.11 * D, n=8, t0=0.0, t1=1.0).reshape(N, 4)
        o_back = self._support(A - 0.03 * D, A - 0.11 * D, n=8, t0=0.0, t1=1.0).reshape(N, 4)
        ranked = -np.sort(-sup, axis=1)
        ok = np.isfinite(aspect) & (np.abs(aspect - CARD_ASPECT) <= ASPECT_TOL) & (tilt <= MAX_TILT_DEG)
        ok &= ranked[:, 2] >= SIDE_OK
        # the fourth side may be under a finger or another card, but not
        # absent altogether: a quad with one side on nothing is three edges
        # and a guess
        ok &= ranked[:, 3] >= 0.2
        # side k is a line across the middle of something bigger when BOTH of
        # its neighbours run on past its ends: half a card, not a card
        for k in range(4):
            ok &= ~((o_fwd[:, (k - 1) % 4] > 0.5) & (o_back[:, (k + 1) % 4] > 0.5))
        over = np.concatenate([o_fwd, o_back], axis=1)
        ok &= (over > 0.6).sum(axis=1) < 3
        # A card's face is not the counter. Lines from the counter's own edge,
        # the side of one card and the end of another can close a big quad
        # around bare wood that is well supported on every side; its middle
        # looks like what surrounds it, a card's never does.
        idx = np.nonzero(ok)[0]
        if len(idx):
            ok[idx] &= self._face_differs(Q[idx])
        support = ranked[:, :3].mean(axis=1)
        score = (support + 0.35 * ranked[:, 3] - 0.6 * np.abs(aspect - CARD_ASPECT)
                 - 0.25 * over.mean(axis=1))
        if self.expected_tilt is not None:
            score = score - 0.3 * np.minimum(1.0, np.abs(tilt - self.expected_tilt) / TILT_PRIOR_DEG)
        out = []
        for i in range(N):
            if not ok[i]:
                out.append(None)
                continue
            out.append(CardHit(Q[i], tuple(float(x) for x in sup[i]), float(support[i]),
                               float(aspect[i]), float(tilt[i]),
                               bool(ranked[i, 3] >= SIDE_OK), float(score[i])))
        return out

    def _refine(self, hit: CardHit) -> CardHit:
        """Snap each side to the gradient ridge along it (sub-pixel), then
        re-judge. Sides with too little support keep their line."""
        q = hit.quad
        lines = []
        rms: list = []
        for k in range(4):
            a, b = q[k], q[(k + 1) % 4]
            d = b - a
            L = float(np.hypot(*d))
            u = d / L
            nrm = np.array([-u[1], u[0]])
            ts = np.linspace(0.1, 0.9, 40)
            offs = np.arange(-4.0, 4.01, 0.5)
            prof = []
            for o in offs:
                p = a[None, :] + ts[:, None] * d[None, :] + o * nrm[None, :]
                xi = np.clip(np.round(p[:, 0]).astype(int), 0, self.w - 1)
                yi = np.clip(np.round(p[:, 1]).astype(int), 0, self.h - 1)
                prof.append(self.ev.gx[yi, xi] * nrm[0] + self.ev.gy[yi, xi] * nrm[1])
            prof = np.array(prof)                       # offsets x samples
            sign = 1.0 if np.abs(np.clip(prof, 0, None)).sum() >= np.abs(np.clip(prof, None, 0)).sum() else -1.0
            sp = prof * sign
            best = sp.argmax(axis=0)
            val = sp.max(axis=0)
            good = val >= self.ev.thresh
            if good.sum() < 8:
                lines.append(None)
                continue
            # parabolic sub-sample peak
            o = offs[best].astype(float)
            for c in np.nonzero(good)[0]:
                bi = best[c]
                if 0 < bi < len(offs) - 1:
                    y0, y1, y2 = sp[bi - 1, c], sp[bi, c], sp[bi + 1, c]
                    den = y0 - 2 * y1 + y2
                    if abs(den) > 1e-9:
                        o[c] += 0.5 * (y0 - y2) / den * 0.5
            pts = a[None, :] + ts[:, None] * d[None, :] + o[:, None] * nrm[None, :]
            pts = pts[good].astype(np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((np.array([x0, y0]), np.array([vx, vy])))
            r = (pts[:, 0] - x0) * (-vy) + (pts[:, 1] - y0) * vx
            rms.append(float(np.sqrt(np.mean(r * r))))
        corners = []
        for k in range(4):
            l1, l2 = lines[(k - 1) % 4], lines[k]
            if l1 is None or l2 is None:
                corners.append(q[k])
                continue
            (p1, v1), (p2, v2) = l1, l2
            A = np.array([v1, -v2]).T
            if abs(np.linalg.det(A)) < 1e-6:
                corners.append(q[k])
                continue
            t = np.linalg.solve(A, p2 - p1)
            corners.append(p1 + t[0] * v1)
        rq = np.array(corners)
        if not _convex(rq) or np.abs(rq - q).max() > 6.0:
            return hit
        rq = enforce_portrait(order_quad(rq))
        again = self._judge(rq)
        if again is None or again.score < hit.score - 0.05:
            return hit
        res = max(0.5, float(np.mean(rms))) if rms else 1.0
        return CardHit(again.quad, again.side_support, again.support, again.aspect,
                       again.tilt_deg, again.complete, again.score, res)

    def _pick(self, hits: Sequence[CardHit]) -> Optional[CardHit]:
        """Best-supported, then outward: the card over its printed frame
        (about 1.2x the area) and the sleeve over the card (about 1.1x) win
        when nearly as well supported; the card around its art window (2-3x)
        wins when the window sits in it the way card art does. Not further
        out than that: a quad around two overlapping cards also contains
        each of them. (Tried instead: score + lambda * log(area). On the field
        frames it found the card from 141 of 168 aim points at lambda 0.2 and
        112 at 0.3, against 156 for this walk.)"""
        if not hits:
            return None
        area = lambda h: abs(cv2.contourArea(h.quad.astype(np.float32)))
        best = max(hits, key=lambda h: h.score)
        return self._pick_walk(hits, best, area)

    def _pick_walk(self, hits, best, area):
        cur = best

        def contains(outer, inner):
            # by area: a sleeve-edge quad and the card inside it can cross by
            # a few pixels at a corner and still be one inside the other
            ai = area(inner)
            if ai <= 0:
                return False
            inter, _ = cv2.intersectConvexConvex(outer.quad.astype(np.float32),
                                                 inner.quad.astype(np.float32))
            return inter >= 0.9 * ai

        while True:
            a = area(cur)
            # the card over its printed frame, the sleeve over the card
            outer = [h for h in hits
                     if h.score >= best.score - 0.1 and a * 1.01 < area(h) <= a * 1.35
                     and contains(h, cur)]
            if not outer:
                # Aimed at the artwork: the art window of most cards is itself
                # a card-shaped rectangle (turned 90 degrees), with clean,
                # straight, consistent-polarity sides, and it wins on support.
                # The card around it is 2-3x its area and fully contains it.
                outer = [h for h in hits
                         if h.score >= best.score - 0.2 and a * 1.6 < area(h) <= a * 3.2
                         and contains(h, cur) and _is_window_of(cur.quad, h.quad)]
            if not outer:
                return cur
            cur = max(outer, key=lambda h: h.score)

    def _extensions(self, hit: CardHit) -> list:
        """Quads that keep three sides of ``hit`` and move the fourth out to a
        parallel line further along the other two (working pixels)."""
        L = self.lines
        if len(L) == 0:
            return []
        Q = hit.quad
        c = Q.mean(axis=0)
        ct, st = np.cos(L.theta), np.sin(L.theta)
        out = []
        for k in range(4):
            a, b = Q[k], Q[(k + 1) % 4]
            d = b - a
            Ls = float(np.hypot(*d))
            if Ls < 1e-6:
                continue
            u = d / Ls
            n = np.array([-u[1], u[0]])
            if np.dot(n, (a + b) / 2 - c) < 0:
                n = -n
            # the two neighbouring sides, as point + direction, pointing past side k
            p1, v1 = a, a - Q[(k - 1) % 4]
            p2, v2 = b, b - Q[(k + 2) % 4]
            span = abs(float(np.dot((a + b) / 2 - (Q[(k + 2) % 4] + Q[(k + 3) % 4]) / 2, n)))
            # lines nearly parallel to side k, beyond it by 0.15-2.5 spans
            dth = np.abs(((L.theta - math.atan2(n[1], n[0])) + math.pi / 2) % math.pi - math.pi / 2)
            mid = (a + b) / 2
            dist = (L.rho - (ct * mid[0] + st * mid[1]))
            # sign: along n from the side
            nn = ct * n[0] + st * n[1]
            dist = dist * np.sign(np.where(np.abs(nn) < 1e-9, 1.0, nn))
            ok = (dth <= math.radians(12.0)) & (dist >= 0.15 * span) & (dist <= 2.5 * span)
            for i in np.nonzero(ok)[0]:
                nr = np.array([ct[i], st[i]])
                den1, den2 = float(nr @ v1), float(nr @ v2)
                if abs(den1) < 1e-9 or abs(den2) < 1e-9:
                    continue
                t1 = (L.rho[i] - float(nr @ p1)) / den1
                t2 = (L.rho[i] - float(nr @ p2)) / den2
                if t1 <= 0 or t2 <= 0:
                    continue
                e1, e2 = p1 + t1 * v1, p2 + t2 * v2
                s1, s2 = e1 + 0.12 * (e2 - e1), e1 + 0.88 * (e2 - e1)
                tt1 = L.along(np.array([i]), s1[None, :])
                tt2 = L.along(np.array([i]), s2[None, :])
                cov = float(L.covered(np.array([i]), tt1, tt2)[0]) / max(abs(float(tt2[0] - tt1[0])), 1.0)
                if cov < 0.3:
                    continue
                q = Q.copy()
                q[k], q[(k + 1) % 4] = e1, e2
                if _convex(q):
                    out.append(q)
        return out

    def _complete(self, hit: CardHit) -> CardHit:
        """Part of a card that looks like a whole one, turned 90 degrees.

        A card's top border, its two sides and the bottom edge of its art
        window make a quad with clean, strong, consistent-polarity sides that
        is card-shaped as a LANDSCAPE card, and on a soft frame it out-scores
        the real card, whose bottom edge is a faint sleeve line. The field
        frames of 2.17.0 (counter_*) are this: the reader locked onto the
        top half of a sleeved card. The real card is the same three sides and
        a further line parallel to the fourth; it is found directly here
        rather than hoping it survived the candidate shortlist, which on busy
        holo artwork is crowded with small fully-edged quads.

        The whole card wins when it is a card, contains the part, and the part
        sits in it the way a card's top half does. Not when what it adds is
        itself card-shaped: two cards side by side make a landscape quad
        around both, and each card is then half of it -- but there what is
        added is a whole card, and here it is the text box."""
        ext = [h for h in self._judge_batch(self._extensions(hit)) if h is not None]
        if not ext:
            return hit
        ha = abs(cv2.contourArea(hit.quad.astype(np.float32)))

        best = None
        for e in ext:
            ea = abs(cv2.contourArea(e.quad.astype(np.float32)))
            if not (1.3 * ha <= ea <= 3.2 * ha) or e.score < hit.score - 0.45:
                continue
            inter, _ = cv2.intersectConvexConvex(e.quad.astype(np.float32), hit.quad.astype(np.float32))
            if inter < 0.9 * ha:
                continue
            # laid out in the whole card the way a card's top part is: full
            # width, from one end, stopping 0.4-0.7 short of the other --
            # measured in the card's own (un-projected) frame, where "which
            # way is long" survives a 40 degree tilt that makes the part look
            # square. A card lying on another reaches much further (0.25-0.36
            # short, desk_spread) and is left alone.
            if not _is_window_of(hit.quad, e.quad):
                continue
            # the card's own sides run on along what is added (on counter_*
            # 0.44 each; an extension up across the gap onto the next card
            # has 0.0-0.3), and what is added is not a card of its own
            if min(self._added_side_support(hit.quad, e.quad)) < 0.35:
                continue
            if self._added_is_card(hit.quad, e.quad):
                continue
            if best is None or e.score > best.score:
                best = e
        return best if best is not None else hit

    @staticmethod
    def _added_corners(part, whole):
        """(corner of part, corner of whole) pairs along the two sides that
        the whole card adds to the part."""
        P = np.asarray(part, np.float64)
        W = np.asarray(whole, np.float64)
        dmin = np.array([np.min(np.hypot(*(P - w).T)) for w in W])
        far = np.argsort(-dmin)[:2]
        return [(P[int(np.argmin(np.hypot(*(P - W[i]).T)))], W[i]) for i in far]

    def _added_side_support(self, part, whole):
        return [float(self._support(a[None, :], b[None, :], n=16, t0=0.1, t1=0.9)[0])
                for a, b in self._added_corners(part, whole)]

    def _runs_off(self, hit: CardHit) -> bool:
        """Is this the visible part of a card that runs out of the frame?

        Too close over a counter, the phone shows a card's top half and cuts
        off the rest (three of the five counter_* frames). The top border, the
        sides and the bottom of the art window are then a clean quad that is
        card-shaped lying SIDEWAYS -- and a sideways card whose sides carry on
        past its edge, where making it an upright card would take it out of
        the frame, is that half. It is refused, so the user is told to show
        the whole card instead of being shown half of one."""
        q = hit.quad                                # TL, TR, BR, BL; 0-1 short
        short = q[1] - q[0]
        long_ = q[2] - q[1]
        ls = float(np.hypot(*short))
        ll = float(np.hypot(*long_))
        if ls < 1e-6 or ll < 1e-6:
            return False
        # sideways: the short side runs up the screen
        if abs(short[1]) / ls < math.cos(math.radians(40.0)):
            return False
        # the part's long sides (0-3 and 1-2 run along; ends are 0-1 and 2-3):
        # an upright card would continue the long sides' neighbours -- which
        # here are the SHORT sides 0-1 and 2-3 -- past one of the long sides.
        # Upright, the card is ~1.4 x the part's long side tall.
        need = CARD_ASPECT * ll - ls
        if need < 0.25 * ls:
            return False
        # The upright card's height is estimated without the perspective (a
        # tilted card's far end is foreshortened, its near end stretched), so
        # a completion that ends within 6% of the border counts as running
        # off: its bottom edge would be too near the border to be seen whole.
        m = 0.06 * min(self.h, self.w)
        for a_end, b_end in (((0, 1), (3, 2)), ((1, 0), (2, 3))):
            # extend 0->1 and 3->2 past corners 1 and 2 (or back past 0 and 3)
            p1, p2 = q[a_end[1]], q[b_end[1]]
            v1 = q[a_end[1]] - q[a_end[0]]
            v2 = q[b_end[1]] - q[b_end[0]]
            v1 = v1 / max(float(np.hypot(*v1)), 1e-9)
            v2 = v2 / max(float(np.hypot(*v2)), 1e-9)
            e1, e2 = p1 + need * v1, p2 + need * v2
            out = any(not (m <= e[0] <= self.w - 1 - m and m <= e[1] <= self.h - 1 - m)
                      for e in (e1, e2))
            if not out:
                continue
            reach = 0.15 * need
            c1 = float(self._support(p1[None, :] + 3 * v1, p1[None, :] + reach * v1,
                                     n=8, t0=0.0, t1=1.0)[0])
            c2 = float(self._support(p2[None, :] + 3 * v2, p2[None, :] + reach * v2,
                                     n=8, t0=0.0, t1=1.0)[0])
            if max(c1, c2) >= 0.75:
                return True
        return False

    def _added_is_card(self, part: np.ndarray, whole: np.ndarray) -> bool:
        """Is the piece of ``whole`` outside ``part`` card-shaped itself?"""
        (pa, wa), (pb, wb) = self._added_corners(part, whole)
        piece = np.array([wa, wb, pb, pa], np.float64)
        if not _convex(piece):
            piece = np.array([wa, wb, pa, pb], np.float64)
            if not _convex(piece):
                return False
        asp, _ = card_pose(order_quad(piece), (self.h, self.w), self.fov)
        return bool(np.isfinite(asp) and abs(asp - CARD_ASPECT) <= ASPECT_TOL)

    def card_at(self, point, nearest: bool = True, _scaled: bool = False) -> Optional[CardHit]:
        """The card under ``point`` (source pixels), or with ``nearest`` the
        closest card to it when the point is on the table; None if none."""
        p = np.asarray(point, float) * self.scale
        cands = [q for q in self._candidates(p)
                 if cv2.pointPolygonTest(q.astype(np.float32), (float(p[0]), float(p[1])), False) >= 0]
        hits = [h for h in self._judge_batch(cands) if h is not None]
        best = self._pick(hits)
        if best is None:
            if not nearest:
                return None
            # Not on a card: the reticle is on the table between cards. Look
            # outward in rings and take the first card found -- the nearest,
            # to within a ring.
            R = max(self.h, self.w)
            for radius in (0.09 * R, 0.17 * R, 0.26 * R):
                ring = []
                for k in range(8):
                    a = k * math.pi / 4
                    q = p + radius * np.array([math.cos(a), math.sin(a)])
                    if 0 <= q[0] < self.w and 0 <= q[1] < self.h:
                        ring.append(q)
                found = []
                for q in ring:
                    hit = self.card_at(q / self.scale, nearest=False, _scaled=True)
                    # a seed near a card's edge can land on part of its face;
                    # asking again from the middle of what it found gets the
                    # whole card
                    for _ in range(2):
                        if hit is None:
                            break
                        again = self.card_at(hit.centre / self.scale, nearest=False, _scaled=True)
                        if again is None or np.allclose(again.quad, hit.quad, atol=1.0):
                            break
                        hit = again
                    # A card chosen without being aimed at must be plainly a
                    # card: all four sides seen, and not a quad around the
                    # aim point itself (from there nothing was found).
                    if (hit is not None and hit.complete and cv2.pointPolygonTest(
                            hit.quad.astype(np.float32), (float(p[0]), float(p[1])), False) < 0):
                        found.append(hit)
                if found:
                    return self._to_source(min(found, key=lambda h: float(np.linalg.norm(h.centre - p))))
            return None
        best = self._refine(self._complete(best))
        if self._runs_off(best):
            self.refusal = RUNS_OFF
            return None
        return best if _scaled else self._to_source(best)

    def all_cards(self, max_cards: int = 16, _scaled: bool = False) -> list[CardHit]:
        """Every card in the frame: seeds on a grid, one card per seed, then
        overlaps resolved in favour of the better-supported card."""
        found: list[CardHit] = []
        step = max(self.min_side * 1.6, min(self.h, self.w) / 7.0)
        ys = np.arange(step / 2, self.h, step)
        xs = np.arange(step / 2, self.w, step)
        covered = np.zeros((self.h, self.w), np.uint8)
        for y in ys:
            for x in xs:
                hits = [h for h in self._judge_batch(self._candidates((x, y))) if h is not None]
                best = self._pick(hits)
                if best is None:
                    continue
                best = self._refine(self._complete(best))
                if self._runs_off(best):
                    continue
                cv2.fillPoly(covered, [best.quad.astype(np.int32)], 1)
                found.append(best)
        found.sort(key=lambda h: -h.score)
        kept: list[CardHit] = []
        for h in found:
            m1 = np.zeros((self.h, self.w), np.uint8)
            cv2.fillPoly(m1, [h.quad.astype(np.int32)], 1)
            dup = False
            for k in kept:
                m2 = np.zeros((self.h, self.w), np.uint8)
                cv2.fillPoly(m2, [k.quad.astype(np.int32)], 1)
                inter = int((m1 & m2).sum())
                if inter > 0.3 * min(int(m1.sum()), int(m2.sum())):
                    dup = True
                    break
            if not dup:
                kept.append(h)
            if len(kept) >= max_cards:
                break
        self._all = kept
        return kept if _scaled else [self._to_source(h) for h in kept]

    def judge(self, quad) -> Optional[CardHit]:
        """Judge a quad given in source pixels (None: not a card)."""
        h = self._judge(np.asarray(quad, dtype=np.float64).reshape(4, 2) * self.scale)
        return None if h is None else self._to_source(h)

    def _to_source(self, h: CardHit) -> CardHit:
        if self.scale == 1.0:
            return h
        return CardHit(h.quad / self.scale, h.side_support, h.support, h.aspect,
                       h.tilt_deg, h.complete, h.score, h.residual_px / self.scale)


def find_card_at(image: np.ndarray, point=None, fov_deg: float = NOMINAL_FOV_DEG,
                 expected_tilt_deg: Optional[float] = None) -> CardHit:
    """The card under ``point`` (default: frame centre). Raises DetectionError."""
    h, w = image.shape[:2]
    if point is None:
        point = (w / 2.0, h / 2.0)
    hit = SceneSearch(image, fov_deg, expected_tilt_deg=expected_tilt_deg).card_at(point)
    if hit is None:
        raise DetectionError("no card under the reticle -- aim the centre of the view at a card")
    return hit


def find_all_cards(image: np.ndarray, fov_deg: float = NOMINAL_FOV_DEG) -> list[CardHit]:
    return SceneSearch(image, fov_deg).all_cards()

