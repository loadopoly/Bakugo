"""Full-resolution strips along the card's edges, for the live measurement.

The live frame is 540 px across: enough to track the card, not to measure it
(4.5 px/mm needed; the 2.22.0 field frames had 3.4-4.4 at distances the
camera could focus). Centering is measured at the card's four edges -- the
card's outer edge and the printed frame just inside it -- so the phone sends,
beside the tracking frame, four strips cut from the camera's full-resolution
frame along the tracked edges: ~6 mm outside to ~11 mm inside each side, at
~11 px/mm. Only the region the measurement reads is sent at the resolution it
needs; the rest of the view goes at tracking resolution. (The same scheduling
as streamed geometry detail in a renderer: fetch detail where the error
budget needs it, and send less, not nothing, when the link is slow -- the
phone lowers the strips' resolution when its round trips run long.)

Here the strips are laid over the tracking frame, upsampled to the strips'
scale, so the existing measurement runs unchanged on the mosaic. A strip is
only as good as the outline that placed it, and the phone cuts it with the
outline from the previous reply: ``covers`` checks that the outline the
server tracks now has its outer edge and border inside the strips before the
mosaic is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from .types import STANDARD_CARD_H_MM, STANDARD_CARD_W_MM

MAX_MOSAIC_PIXELS = 9_000_000
MAX_SCALE = 8.0


@dataclass
class DetailMosaic:
    image: np.ndarray            # the mosaic, in detail pixels
    origin: tuple                # frame coordinates of the mosaic's (0, 0)
    m: float                     # detail pixels per frame pixel
    mask: np.ndarray             # 1 where a strip supplied the pixels
    frame_shape: tuple           # (h, w) of the tracking frame

    def to_mosaic(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, dtype=np.float64) - np.asarray(self.origin)) * self.m

    @property
    def parent(self) -> tuple:
        """(x, y, full_w, full_h) of the mosaic inside the frame, scaled to
        detail pixels: the lens geometry frame_card_for_measure needs."""
        h, w = self.frame_shape
        return (self.origin[0] * self.m, self.origin[1] * self.m, w * self.m, h * self.m)


def build_mosaic(frame: np.ndarray, strips: Sequence[tuple], m: float) -> Optional[DetailMosaic]:
    """``strips``: (image, (x, y, w, h) in frame pixels). None when there is
    nothing usable."""
    if not strips or not (1.0 < m <= MAX_SCALE):
        return None
    fh, fw = frame.shape[:2]
    rects = []
    for img, r in strips:
        if img is None or img.size == 0 or len(r) != 4:
            continue
        x, y, w, h = (float(v) for v in r)
        if w <= 1 or h <= 1:
            continue
        rects.append((img, x, y, w, h))
    if not rects:
        return None
    x0 = max(0, int(np.floor(min(r[1] for r in rects))))
    y0 = max(0, int(np.floor(min(r[2] for r in rects))))
    x1 = min(fw, int(np.ceil(max(r[1] + r[3] for r in rects))))
    y1 = min(fh, int(np.ceil(max(r[2] + r[4] for r in rects))))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    W, H = int(round((x1 - x0) * m)), int(round((y1 - y0) * m))
    if W * H > MAX_MOSAIC_PIXELS:
        return None
    base = frame[y0:y1, x0:x1]
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    mosaic = cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC)
    mask = np.zeros((H, W), np.uint8)
    for img, x, y, w, h in rects:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        dx, dy = int(round((x - x0) * m)), int(round((y - y0) * m))
        dw, dh = int(round(w * m)), int(round(h * m))
        if dw < 2 or dh < 2:
            continue
        if abs(img.shape[1] - dw) > 1 or abs(img.shape[0] - dh) > 1:
            img = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
        dh, dw = img.shape[:2]
        sx0, sy0 = max(0, -dx), max(0, -dy)
        tx0, ty0 = max(0, dx), max(0, dy)
        tx1, ty1 = min(W, dx + dw), min(H, dy + dh)
        if tx1 <= tx0 or ty1 <= ty0:
            continue
        mosaic[ty0:ty1, tx0:tx1] = img[sy0:sy0 + (ty1 - ty0), sx0:sx0 + (tx1 - tx0)]
        mask[ty0:ty1, tx0:tx1] = 1
    if not mask.any():
        return None
    return DetailMosaic(mosaic, (float(x0), float(y0)), float(m), mask, (fh, fw))


def covers(mosaic: DetailMosaic, quad_frame: np.ndarray,
           out_mm: float = 3.0, in_mm: float = 7.0) -> bool:
    """Do the strips hold every side from ``out_mm`` outside the card to
    ``in_mm`` inside it (the edge, a sleeve's margin, and the border)?"""
    q = mosaic.to_mosaic(np.asarray(quad_frame, dtype=np.float64).reshape(4, 2))
    ppm = 0.5 * (np.linalg.norm(q[1] - q[0]) / STANDARD_CARD_W_MM
                 + np.linalg.norm(q[3] - q[0]) / STANDARD_CARD_H_MM)
    if not np.isfinite(ppm) or ppm <= 0:
        return False
    c = q.mean(axis=0)
    H, W = mosaic.mask.shape
    ts = np.linspace(0.1, 0.9, 12)
    for k in range(4):
        a, b = q[k], q[(k + 1) % 4]
        d = b - a
        n = np.array([-d[1], d[0]]) / max(float(np.linalg.norm(d)), 1e-9)
        if float(n @ ((a + b) / 2 - c)) < 0:
            n = -n                                            # outward
        for off in (out_mm, -in_mm):
            pts = a[None, :] + ts[:, None] * d[None, :] + off * ppm * n[None, :]
            xi = np.round(pts[:, 0]).astype(int)
            yi = np.round(pts[:, 1]).astype(int)
            if (xi < 0).any() or (yi < 0).any() or (xi >= W).any() or (yi >= H).any():
                return False
            if not mosaic.mask[yi, xi].all():
                return False
    return True
