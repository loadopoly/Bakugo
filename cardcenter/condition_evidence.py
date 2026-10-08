"""Corners, edges and surface across many views: what AR adds over a photo.

A grader does not look at a card once. They tilt it under a lamp so the
reflection runs across the face, and they turn it over. The live session can
do the same with the views it already takes, and each kind of evidence gets
better from more views in its own way:

* WHITENING is never ADDED by a view: glare and a tilted card's side face
  can only make a border look whiter than it is. So across views the low end
  of the readings is the estimate (the 30th percentile once there are three;
  the mean of two before that), not the mean or the worst.
* LOSS (material missing at a corner, a nick in an edge) is geometry, the
  same in every view, so its median is the estimate and its scatter the
  uncertainty -- floored by the resolution, since views at the same px/mm
  share the same blur.
* SURFACE only exists under a highlight, and a thin dark line inside one is a
  scratch only if the same place, seen without the highlight, shows no such
  line in the print. Each view records which cells sat under a highlight and
  what the card looks like where none was; a candidate becomes a defect when
  two views see it at the same place and the diffuse print there is clean.
  Coverage -- how much of the face has been under a highlight at all -- is
  reported, and below ``min_coverage`` the surface is not assessed.

Front and back are kept apart (``face`` from ``condition.classify_face``):
the physical corner at the front's top-left is the back's top-right, and the
grading combines the two faces of each physical corner and side.

Nothing here grades. ``summary()`` returns estimates with uncertainties and
says which aspects have evidence; ``grading.predict_overall_grade`` turns that
into grade probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .condition import (
    CORNERS,
    SIDES,
    SURFACE_CELL_MM,
    ConditionView,
    CornerReading,
    EdgeReading,
    SurfaceReading,
    load_condition_standards,
)
from .types import STANDARD_CARD_H_MM, STANDARD_CARD_W_MM

SURFACE_REF_PX_PER_MM = 8.0
MAX_DIFFUSE_VIEWS = 6
MAX_READINGS = 40
CONFIRM_DIST_MM = 0.7
CONFIRM_ANGLE_DEG = 25.0

# the back of a card seen face-down is mirrored left to right
BACK_TO_FRONT_CORNER = {
    "top_left": "top_right", "top_right": "top_left",
    "bottom_right": "bottom_left", "bottom_left": "bottom_right",
}
BACK_TO_FRONT_SIDE = {"left": "right", "right": "left", "top": "top", "bottom": "bottom"}


@dataclass(frozen=True)
class Estimate:
    """One reading pooled across views: a value in mm (or a fraction) with a
    1-sigma uncertainty, the number of views behind it, and the finest
    detail those views could see."""

    value: float
    sigma: float
    n: int
    floor: float = 0.0

    def to_dict(self) -> dict:
        return {"value": round(self.value, 3), "sigma": round(self.sigma, 3),
                "n": self.n, "floor": round(self.floor, 3)}


def _low(vals: list) -> float:
    """The low end of readings that can only be inflated (whitening)."""
    v = sorted(vals)
    if len(v) == 1:
        return float(v[0])
    if len(v) == 2:
        return float(0.5 * (v[0] + v[1]))
    return float(np.percentile(v, 30))


def _spread(vals: list) -> float:
    if len(vals) < 3:
        return 0.0
    a = np.asarray(vals, float)
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


@dataclass
class _SurfaceAccum:
    shape: tuple = (int(round(STANDARD_CARD_H_MM * SURFACE_REF_PX_PER_MM)),
                    int(round(STANDARD_CARD_W_MM * SURFACE_REF_PX_PER_MM)))
    inspected: Optional[np.ndarray] = None
    views: int = 0                      # resolvable views with a highlight
    foil_views: int = 0
    candidates: list = field(default_factory=list)   # (view, x, y, L, dx, dy)
    diffuse: list = field(default_factory=list)      # (gray, mask) at REF res
    reasons: dict = field(default_factory=dict)

    def add(self, r: SurfaceReading, view_idx: int) -> None:
        if r.diffuse is not None and r.diffuse_mask is not None:
            g = cv2.resize(r.diffuse, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_AREA)
            m = cv2.resize(r.diffuse_mask.astype(np.uint8), (self.shape[1], self.shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 0
            self.diffuse.append((g, m))
            if len(self.diffuse) > MAX_DIFFUSE_VIEWS:
                self.diffuse.pop(0)
        if not r.resolvable:
            self.reasons[r.reason] = self.reasons.get(r.reason, 0) + 1
            return
        if r.reason.startswith("the highlight is full of structure"):
            self.foil_views += 1
            return
        if r.inspected is None or r.glare_frac < 0.01:
            return
        if self.inspected is None:
            self.inspected = np.zeros_like(r.inspected, bool)
        elif self.inspected.shape != r.inspected.shape:
            return
        self.inspected |= r.inspected
        self.views += 1
        for c in r.candidates:
            self.candidates.append((view_idx,) + tuple(c))
        if len(self.candidates) > 400:
            self.candidates = self.candidates[-400:]

    @property
    def coverage(self) -> float:
        return float(self.inspected.mean()) if self.inspected is not None else 0.0

    def _reference(self) -> Optional[np.ndarray]:
        if not self.diffuse:
            return None
        stack = np.stack([g.astype(np.float32) for g, _ in self.diffuse])
        masks = np.stack([m for _, m in self.diffuse])
        stack = np.where(masks, stack, np.nan)
        import warnings

        with warnings.catch_warnings():
            # cells never seen without a highlight stay NaN: "unknown"
            warnings.simplefilter("ignore", RuntimeWarning)
            ref = np.nanmedian(stack, axis=0)
        return ref

    def _printed(self, ref: Optional[np.ndarray], c) -> bool:
        """Is there a dark line in the print (no highlight) where candidate
        ``c`` sits? Then it is artwork, not a scratch."""
        if ref is None:
            return False
        _, x, y, L, dx, dy = c
        s = SURFACE_REF_PX_PER_MM
        ts = np.linspace(-L / 2.0, L / 2.0, max(4, int(L * s)))
        xs = (x + ts * dx) * s
        ys = (y + ts * dy) * s
        ok = (xs >= 0) & (ys >= 0) & (xs < ref.shape[1] - 1) & (ys < ref.shape[0] - 1)
        if ok.sum() < 3:
            return False
        valid = np.nan_to_num(ref, nan=np.nanmedian(ref) if np.isfinite(ref).any() else 128.0)
        # darkest within +-0.5 mm of the line, against the neighbourhood
        k = max(3, int(round(1.0 * s)) | 1)
        local_min = cv2.erode(valid.astype(np.float32), np.ones((k, k), np.uint8))
        local_med = cv2.medianBlur(np.clip(valid, 0, 255).astype(np.uint8), 2 * int(0.8 * s) + 1).astype(np.float32)
        xi, yi = xs[ok].astype(int), ys[ok].astype(int)
        known = np.isfinite(ref[yi, xi])
        if known.mean() < 0.5:
            return False          # never seen without a highlight there: can't tell
        dark = (local_min[yi, xi] < local_med[yi, xi] - 12.0)
        return bool(dark[known].mean() >= 0.5)

    def defects(self) -> list:
        """Candidates seen in two or more views, close and parallel, that the
        diffuse print does not explain."""
        ref = self._reference()
        out = []
        used = set()
        cands = self.candidates
        for i, a in enumerate(cands):
            if i in used:
                continue
            group = [i]
            for j in range(i + 1, len(cands)):
                b = cands[j]
                if b[0] == a[0] or j in used:
                    continue
                if math.hypot(a[1] - b[1], a[2] - b[2]) > CONFIRM_DIST_MM + 0.25 * max(a[3], b[3]):
                    continue
                cosang = abs(a[4] * b[4] + a[5] * b[5])
                if cosang < math.cos(math.radians(CONFIRM_ANGLE_DEG)):
                    continue
                group.append(j)
            views = {cands[g][0] for g in group}
            if len(views) < 2:
                continue
            used.update(group)
            if self._printed(ref, a):
                continue
            L = max(cands[g][3] for g in group)
            out.append({"x_mm": round(a[1], 1), "y_mm": round(a[2], 1),
                        "length_mm": round(L, 1), "views": len(views)})
        return out


@dataclass
class _FaceEvidence:
    corners: dict = field(default_factory=lambda: {c: [] for c in CORNERS})
    edges: dict = field(default_factory=lambda: {s: [] for s in SIDES})
    surface: _SurfaceAccum = field(default_factory=_SurfaceAccum)
    views: int = 0
    reasons: dict = field(default_factory=dict)

    def add(self, v: ConditionView, view_idx: int) -> None:
        self.views += 1
        for name, r in v.corners.items():
            if r.resolvable:
                self.corners[name].append(r)
                del self.corners[name][:-MAX_READINGS]
            else:
                self.reasons[f"corner:{r.reason}"] = self.reasons.get(f"corner:{r.reason}", 0) + 1
        for side, r in v.edges.items():
            if r.resolvable:
                self.edges[side].append(r)
                del self.edges[side][:-MAX_READINGS]
            else:
                self.reasons[f"edge:{r.reason}"] = self.reasons.get(f"edge:{r.reason}", 0) + 1
        self.surface.add(v.surface, view_idx)

    # -- pooled estimates ---------------------------------------------------

    def corner(self, name: str) -> Optional[Estimate]:
        rs: list[CornerReading] = self.corners[name]
        if not rs:
            return None
        loss = [r.loss_mm for r in rs]
        sig = float(np.median([r.sigma_mm for r in rs]))
        floor_ = float(max(0.08, min(1.0 / max(r.px_per_mm, 1e-6) for r in rs)))
        loss_v = float(np.median(loss))
        loss_s = math.hypot(sig / math.sqrt(min(len(rs), 4)), _spread(loss) / math.sqrt(len(rs)))
        wr = [r.whitening_mm for r in rs if r.whitening_resolvable]
        if wr:
            w_v = _low(wr)
            w_s = math.hypot(sig / math.sqrt(min(len(wr), 4)), _spread(wr) / math.sqrt(len(wr)))
        else:
            w_v, w_s = 0.0, 0.0
        if w_v > loss_v:
            return Estimate(w_v, max(0.03, w_s), len(rs), floor_)
        return Estimate(loss_v, max(0.03, loss_s), len(rs), floor_)

    def corner_whitening_known(self, name: str) -> bool:
        return any(r.whitening_resolvable for r in self.corners[name])

    def edge_whitening(self, side: str) -> Optional[Estimate]:
        rs: list[EdgeReading] = [r for r in self.edges[side] if r.whitening_resolvable]
        if not rs:
            return None
        vals = [r.whitening_frac for r in rs]
        v = _low(vals)
        n_pos = max(8.0, float(np.median([r.usable_frac for r in rs])) * 160.0 / 4.0)
        s = math.hypot(math.sqrt(max(v * (1 - v), 0.01) / n_pos), _spread(vals) / math.sqrt(len(vals)))
        return Estimate(v, max(0.02, s), len(rs),
                        float(max(0.08, min(1.0 / max(r.px_per_mm, 1e-6) for r in rs))))

    def edge_nick(self, side: str) -> Optional[Estimate]:
        rs: list[EdgeReading] = self.edges[side]
        if not rs:
            return None
        vals = [r.nick_max_mm for r in rs]
        v = float(np.median(vals))
        floor_ = float(max(0.08, min(1.0 / max(r.px_per_mm, 1e-6) for r in rs)))
        s = math.hypot(max(floor_ * 0.6, float(np.median([r.noise_mm for r in rs]))) / math.sqrt(min(len(rs), 4)),
                       _spread(vals) / math.sqrt(len(rs)))
        return Estimate(v, max(0.04, s), len(rs), floor_)

    def summary(self) -> dict:
        cons = load_condition_standards()
        corners = {c: self.corner(c) for c in CORNERS}
        edges = {s: {"whitening": self.edge_whitening(s), "nick": self.edge_nick(s)} for s in SIDES}
        surf = self.surface
        cov = surf.coverage
        surface_assessed = (surf.views >= 2 and cov >= cons["surface"]["min_coverage"]
                            and surf.foil_views <= surf.views)
        return {
            "views": self.views,
            "corners": corners,
            "edges": edges,
            "surface": {
                "assessed": surface_assessed,
                "coverage": cov,
                "views": surf.views,
                "foil": surf.foil_views > surf.views,
                "defects": surf.defects() if surface_assessed else [],
            },
        }


class ConditionEvidence:
    """Condition readings for one card, pooled over every view of it."""

    def __init__(self) -> None:
        self.faces = {"front": _FaceEvidence(), "back": _FaceEvidence()}
        self.n_views = 0
        self.last_view: Optional[ConditionView] = None
        self.last_face: Optional[str] = None

    def add(self, view: ConditionView) -> None:
        face = view.face if view.face in self.faces else "front"
        self.faces[face].add(view, self.n_views)
        self.n_views += 1
        self.last_view = view
        self.last_face = face

    @classmethod
    def from_view(cls, view: ConditionView) -> "ConditionEvidence":
        e = cls()
        e.add(view)
        return e

    def has(self, face: str) -> bool:
        return self.faces[face].views > 0

    def summary(self) -> dict:
        return {f: fe.summary() for f, fe in self.faces.items() if fe.views}

    # -- what to do next ------------------------------------------------------

    def next_action(self) -> Optional[str]:
        """The one change of view that would add the most missing evidence,
        in words a person at a counter can act on. None when nothing is
        missing that another view could supply."""
        if not self.n_views:
            return None
        front = self.faces["front"]
        face = front if front.views else self.faces["back"]
        lv = self.last_view
        cons = load_condition_standards()["resolution"]
        missing_c = [c for c in CORNERS if not face.corners[c]]
        missing_e = [s for s in SIDES if not face.edges[s]]
        if missing_c or missing_e:
            if lv is not None:
                res = [r for r in list(lv.corners.values()) + list(lv.edges.values())
                       if not r.resolvable and "px/mm" in r.reason]
                if res:
                    worst = min(r.px_per_mm for r in res)
                    return (f"corners and edges need ~{cons['corner_min_px_per_mm']:.0f} px/mm "
                            f"(this view {worst:.1f}) -- move closer or zoom in")
                soft = [r for r in list(lv.corners.values()) + list(lv.edges.values())
                        if not r.resolvable and "soft" in r.reason]
                if soft:
                    return "edges too soft to read -- hold still and let it focus"
                bg = [r for r in list(lv.corners.values()) + list(lv.edges.values())
                      if not r.resolvable and "background" in r.reason]
                if bg:
                    return "the card's edge blends into the surface -- put it on a contrasting one"
            names = [c.replace("_", "-") for c in missing_c] + missing_e
            return "still to see clearly: " + ", ".join(names[:3])
        s = face.surface
        if s.foil_views > s.views:
            return None
        cov = s.coverage
        if cov < load_condition_standards()["surface"]["min_coverage"]:
            return (f"surface {int(round(100 * cov))}% inspected -- tilt the card slowly so the "
                    "light's reflection sweeps across it")
        if not self.faces["back"].views:
            return "front done -- flip the card to check the back's corners and edges"
        return None
