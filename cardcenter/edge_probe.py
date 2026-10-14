"""Straightness and colour split of the card's four cut edges, read from the photo.

Two measurements that decide whether two optical effects matter, before anything is modelled for them:

* **Straightness (sagitta).** A card's cut edges are straight. In a photo they bow when the lens has radial distortion
  or the holder's face is curved (a toploader that bows outward is a weak cylindrical lens). Each edge is located at
  ~40 points along its length; a quadratic through them gives the sagitta, how far the middle of the edge sits off the
  chord between its ends, in mm (positive bows outward, away from the card). The RMS about the best straight line is
  reported too. Lens distortion bows all four edges the same way, scaled by their distance from the image centre; a
  curved holder bows the edges across the curve only.
* **Colour split.** The same edge located separately in the red, green and blue channels. Lateral chromatic
  aberration in the lens, or dispersion in the plastic, moves the channels apart along the edge normal. The median
  red-minus-green and blue-minus-green offsets per side, in pixels (positive is outward), say how large it is. The
  plastic's own dispersion is a few micrometres (a small fraction of a pixel), so a large split is the lens.

Pure OpenCV and NumPy; nothing here changes the measurement, it is reported beside it.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .types import STANDARD_CARD_H_MM, STANDARD_CARD_W_MM

SIDES = ("top", "right", "bottom", "left")
_SIDE_MM = {"top": STANDARD_CARD_W_MM, "bottom": STANDARD_CARD_W_MM, "left": STANDARD_CARD_H_MM, "right": STANDARD_CARD_H_MM}
SAMPLES = 41
STEP_PX = 0.25
MIN_STEP_GRADIENT = 2.0     # grey levels per pixel at the edge; flatter than that is not an edge
MIN_POINTS = 12


def order_quad(quad) -> np.ndarray:
    """(4, 2) corners as top-left, top-right, bottom-right, bottom-left."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    s, d = q.sum(1), q[:, 1] - q[:, 0]
    return np.array([q[np.argmin(s)], q[np.argmin(d)], q[np.argmax(s)], q[np.argmax(d)]])


_SMOOTH = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)     # sigma 0.75 px at 0.25 px steps
_SMOOTH /= _SMOOTH.sum()


def _peak(profile: np.ndarray) -> Optional[tuple[float, float]]:
    """(sub-sample index of the steepest step, its gradient per pixel) or None. The profile is smoothed first:
    bilinear sampling makes the raw gradient flat-topped, and a parabola on a plateau has no sub-sample peak."""
    g = np.abs(np.gradient(np.convolve(profile, _SMOOTH, mode="same")))
    g[:9] = 0
    g[-9:] = 0
    i = int(np.argmax(g))
    if i <= 0 or i >= len(g) - 1:
        return None
    a, b, c = g[i - 1], g[i], g[i + 1]
    den = a - 2 * b + c
    off = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
    return i + float(np.clip(off, -0.5, 0.5)), float(b / STEP_PX)


def probe_edges(image: np.ndarray, quad, search_mm: float = 0.8) -> dict:
    """Per side: sagitta and line RMS in mm, colour split in px, and how many points located the edge."""
    img = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img = img.astype(np.float32)
    h, w = img.shape[:2]
    q = order_quad(quad)
    centre = q.mean(0)
    out: dict = {}
    for k, side in enumerate(SIDES):
        a, b = q[k], q[(k + 1) % 4]
        L = float(np.linalg.norm(b - a))
        if L < 20:
            out[side] = {"points": 0}
            continue
        tang = (b - a) / L
        nrm = np.array([-tang[1], tang[0]])
        if np.dot(nrm, (a + b) / 2 - centre) < 0:
            nrm = -nrm                                       # outward
        px_mm = L / _SIDE_MM[side]
        R = max(6.0, search_mm * px_mm)
        offs = np.arange(-R, R + STEP_PX, STEP_PX)
        ts = np.linspace(0.08, 0.92, SAMPLES)
        pts = a[None, :] + (ts * L)[:, None] * tang[None, :]                     # (S, 2)
        grid = pts[:, None, :] + offs[None, :, None] * nrm[None, None, :]        # (S, P, 2)
        inside = ((grid[..., 0] >= 1) & (grid[..., 0] <= w - 2) & (grid[..., 1] >= 1) & (grid[..., 1] <= h - 2)).all(1)
        mx, my = grid[..., 0].astype(np.float32), grid[..., 1].astype(np.float32)
        prof = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)  # (S, P, 3) B,G,R
        lum = prof.mean(2)
        u, s_lum, rg, bg = [], [], [], []
        for i in range(SAMPLES):
            if not inside[i]:
                continue
            pk = _peak(lum[i])
            if pk is None or pk[1] < MIN_STEP_GRADIENT:
                continue
            pos = offs[0] + pk[0] * STEP_PX
            u.append(2 * ts[i] - 1)
            s_lum.append(pos)
            ch = [_peak(prof[i, :, c]) for c in range(3)]
            if all(x is not None and x[1] >= MIN_STEP_GRADIENT for x in ch):
                bpos, gpos, rpos = (offs[0] + x[0] * STEP_PX for x in ch)
                rg.append(rpos - gpos)
                bg.append(bpos - gpos)
        if len(u) < MIN_POINTS:
            out[side] = {"points": len(u)}
            continue
        u_a, s_a = np.asarray(u), np.asarray(s_lum)
        (c2, c1, c0), cov = np.polyfit(u_a, s_a, 2, cov=True)
        l1, l0 = np.polyfit(u_a, s_a, 1)
        rms_line = float(np.sqrt(np.mean((s_a - (l1 * u_a + l0)) ** 2)))
        # midpoint (u=0) relative to the chord through u=-1 and u=+1: c0 - (c0 + c2) = -c2; outward positive
        out[side] = {
            "points": int(len(u)),
            "sagitta_mm": round(float(-c2) / px_mm, 4),
            "sagitta_sigma_mm": round(float(np.sqrt(max(cov[0, 0], 0.0))) / px_mm, 4),
            "line_rms_mm": round(rms_line / px_mm, 4),
            "px_per_mm": round(px_mm, 3),
            "red_minus_green_px": round(float(np.median(rg)), 3) if len(rg) >= MIN_POINTS else None,
            "blue_minus_green_px": round(float(np.median(bg)), 3) if len(bg) >= MIN_POINTS else None,
        }
    return out
