"""The information floor of a detected card outline, and what it tells the AR
loop to do.

``information.py`` derives the Cramer-Rao bound on locating an edge from the
channel that carried it (contrast C, per-pixel noise sigma_n, blur sigma_p,
pixel pitch, independent rows N), and the Poisson (shot-noise) version with
the sensor's interstitial fill-factor losses folded into its quantum
efficiency. Until now that floor was computed only on the *rectified* card
inside the measurement. This module computes it on the *camera frame*, along
the four sides of the quad that ``geometry.find_card_quad`` or
``ar.track_quad`` returned, so the same physics can steer detection and
tracking:

* ``quad_information`` samples intensity profiles across each side, per
  colour channel (a border can be iso-luminant against the table, which is why
  ``compute_edge_gradient`` also takes the per-channel maximum), and keeps the
  channel with the best contrast-to-noise. From that it reports, per side, the
  ``ChannelConditions``, the CR floor on the side's position, whether the side
  is resolvable at all (contrast > 2 sigma_n), and the shot-noise ratio.
* Corner uncertainty follows from two fitted lines: a line fitted over the
  middle 80% of a side and extrapolated to its end has
  sigma_end = sigma_offset * sqrt(1 + 3 * (0.5 / 0.4)^2) (offset plus slope
  terms for uniform sampling), and two such lines meeting at angle theta give
  sigma_corner = sqrt(sigma_end_a^2 + sigma_end_b^2) / sin(theta).
* ``capture_advice`` ranks what the user should change by the factor it would
  cut the floor by, from sigma_mm ~ (sigma_n / C) * sqrt(sigma_p * pitch / N)
  / px_per_mm with N ~ side length ~ px_per_mm: moving closer scales the floor
  as ppm^-1.5, contrast as 1/C, blur as sqrt(sigma_p).

Rows are counted as independent only once per blur width
(N_eff = sampled length / max(1, 2 sigma_p)). Neighbouring rows closer than
that share noise through the optics and demosaic, and counting them separately
would claim information the frame does not carry; the bound stays a floor.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np

from .information import (
    ChannelConditions,
    SensorModel,
    cramer_rao_edge_px,
    cramer_rao_ratio_pp,
    shot_noise_consistency,
)

SIDE_NAMES = ("top", "right", "bottom", "left")
_END_FACTOR = math.sqrt(1.0 + 3.0 * (0.5 / 0.4) ** 2)
NOMINAL_BORDER_MM = 3.0
TARGET_PX_PER_MM = 10.0
TARGET_SNR = 40.0
TARGET_PSF_PX = 0.8


@dataclass(frozen=True)
class SideInformation:
    side: str
    channel: ChannelConditions
    channel_index: int          # 0,1,2 = B,G,R
    sigma_px: float             # CR floor on the side's position (all rows)
    sigma_end_px: float         # at the side's end points, after extrapolation
    resolvable: bool
    background_level: float
    shot_ratio: float
    length_px: float
    offset_px: float = 0.0      # where the edge actually is, from the quad side (+ outward)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channel"] = asdict(self.channel)
        # Non-finite -> None: Python's json writes inf as the bare token
        # Infinity, which no browser's JSON.parse accepts.
        return {k: ((round(v, 4) if math.isfinite(v) else None) if isinstance(v, float) else v)
                for k, v in d.items()}


@dataclass(frozen=True)
class Advice:
    key: str                    # closer | contrast | steady | light
    message: str
    gain: float                 # factor by which the floor would drop


@dataclass(frozen=True)
class QuadInformation:
    sides: tuple
    corner_sigma_px: tuple
    px_per_mm: float
    sigma_cr_pp: float          # CR floor on the centering ratio, nominal borders
    shot_ratio: float           # worst side
    advice: tuple

    @property
    def resolvable(self) -> bool:
        return all(s.resolvable for s in self.sides)

    @property
    def unresolved_sides(self) -> tuple:
        return tuple(s.side for s in self.sides if not s.resolvable)

    @property
    def worst_corner_sigma_px(self) -> float:
        return max(self.corner_sigma_px)

    @property
    def limiting(self) -> Optional[str]:
        return self.advice[0].key if self.advice else None

    def scaled(self, factor: float) -> "QuadInformation":
        """The same outline measured on an image resized by ``factor``
        (e.g. from the 540 px tracking frame to full resolution): positional
        sigmas scale with the image."""
        return QuadInformation(
            sides=self.sides,
            corner_sigma_px=tuple(c * factor for c in self.corner_sigma_px),
            px_per_mm=self.px_per_mm * factor,
            sigma_cr_pp=self.sigma_cr_pp,
            shot_ratio=self.shot_ratio,
            advice=self.advice,
        )

    def to_dict(self) -> dict:
        def r(x):
            return round(float(x), 4) if math.isfinite(float(x)) else None

        return {
            "resolvable": self.resolvable,
            "unresolved_sides": list(self.unresolved_sides),
            "corner_sigma_px": [r(c) for c in self.corner_sigma_px],
            "px_per_mm": r(self.px_per_mm),
            "sigma_cr_pp": r(self.sigma_cr_pp),
            "shot_ratio": r(self.shot_ratio),
            "limiting": self.limiting,
            # gain is inf when a side cannot be seen at all (no finite floor to
            # improve on); JSON has no Infinity, so it goes out as null.
            "advice": [{"key": a.key, "message": a.message,
                        "gain": round(float(a.gain), 2) if math.isfinite(float(a.gain)) else None}
                       for a in self.advice],
            "sides": [{"side": s.side, "sigma_px": r(s.sigma_px), "resolvable": s.resolvable,
                       "contrast": r(s.channel.contrast), "noise": r(s.channel.noise_sigma),
                       "psf_px": r(s.channel.psf_sigma_px), "rows": s.channel.rows,
                       "offset_px": r(s.offset_px),
                       "shot_ratio": r(s.shot_ratio)} for s in self.sides],
        }


def _profiles(img: np.ndarray, a: np.ndarray, b: np.ndarray, inner: float, outer: float,
              samples: int):
    """Intensity profiles across the side a->b. Rows: positions along the
    side (middle 80%); columns: offsets from -inner (inside the card) to
    +outer (outside). Returns (profiles [C x rows x offsets], offsets, length)."""
    edge = b - a
    L = float(np.linalg.norm(edge))
    t = edge / L
    n = np.array([t[1], -t[0]])          # outward for a clockwise quad, y down
    ts = np.linspace(0.1, 0.9, samples)
    offs = np.arange(-float(np.floor(inner)), float(np.floor(outer)) + 1e-9, 1.0)
    base = a[None, :] + ts[:, None] * edge[None, :]
    pts = base[:, None, :] + offs[None, :, None] * n[None, None, :]
    mx = pts[..., 0].astype(np.float32)
    my = pts[..., 1].astype(np.float32)
    chans = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[2])]
    out = [cv2.remap(ch.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE) for ch in chans]
    return np.stack(out), offs, L


def _channel_from_profile(P: np.ndarray, offs: np.ndarray, length_px: float):
    """P: rows x offsets for one colour channel. The edge is located per row
    (strongest gradient) and rows are aligned on it before the step is
    measured, so a quad side a few pixels off the true edge still yields the
    edge's own contrast, and the median per-row location is reported as the
    side's offset."""
    g = np.abs(np.gradient(P, axis=1))
    # OUTERMOST STRONG TRANSITION, not the strongest. Inside the cut edge the
    # printed frame (border -> artwork) is often as strong as the cut edge
    # itself; find_card_quad settles the same ambiguity by nesting (the
    # outermost card-shaped region is the card), and so does this.
    strong = g >= 0.5 * g.max(axis=1, keepdims=True)
    last = g.shape[1] - 1 - np.argmax(strong[:, ::-1], axis=1)
    lo = np.clip(last - 1, 0, g.shape[1] - 1)
    hi = np.clip(last + 1, 0, g.shape[1] - 1)
    ks = np.where(g[np.arange(len(g)), lo] > g[np.arange(len(g)), last], lo,
                  np.where(g[np.arange(len(g)), hi] > g[np.arange(len(g)), last], hi, last))
    k0 = int(np.median(ks))
    # Align rows on their own edge, clipped to a band around the median so a
    # glare streak on one row cannot drag it across the window.
    shift = np.clip(ks - k0, -3, 3)
    idx = np.clip(np.arange(P.shape[1])[None, :] + shift[:, None], 0, P.shape[1] - 1)
    P = np.take_along_axis(P, idx, axis=1)
    mean_prof = np.median(P, axis=0)
    grad = np.abs(np.gradient(mean_prof))
    k = k0
    if 0 < k < len(grad) - 1:
        k = k - 1 + int(np.argmax(grad[k - 1:k + 2]))
    # Flat bands on either side, local to this edge: they start past the
    # edge's own transition (a blurred edge is several pixels wide) and stop
    # before the next strong transition (the printed frame inside, a
    # neighbouring card outside), so the step and the noise belong to this
    # edge alone.
    peak = float(grad[k])
    run_lo, run_hi = k, k
    while run_lo > 0 and grad[run_lo - 1] >= 0.25 * peak:
        run_lo -= 1
    while run_hi < len(grad) - 1 and grad[run_hi + 1] >= 0.25 * peak:
        run_hi += 1
    strong_cols = np.where(grad >= 0.5 * peak)[0]
    left_stop = strong_cols[strong_cols < run_lo]
    right_stop = strong_cols[strong_cols > run_hi]
    i1 = run_lo - 1
    i0 = int(left_stop.max()) + 2 if len(left_stop) else 0
    j0 = run_hi + 2
    j1 = int(right_stop.min()) - 1 if len(right_stop) else P.shape[1]
    i0 = max(i0, i1 - 8)
    j1 = min(j1, j0 + 8)
    inner = P[:, max(0, i0):max(0, i1)]
    outer = P[:, min(j0, P.shape[1]):max(min(j0, P.shape[1]), min(j1, P.shape[1]))]
    if inner.shape[1] < 2 or outer.shape[1] < 2:
        return None
    inner_level = float(np.median(inner))
    outer_level = float(np.median(outer))
    contrast = abs(inner_level - outer_level)

    def mad_noise(block):
        resid = block - np.median(block, axis=1, keepdims=True)
        return float(np.median(np.abs(resid))) * 1.4826

    noise = max(0.5, math.sqrt(0.5 * (mad_noise(inner) ** 2 + mad_noise(outer) ** 2)))
    psf = 1.0
    if contrast > 2 * noise:
        lo = min(inner_level, outer_level)
        seg = mean_prof if mean_prof[0] < mean_prof[-1] else mean_prof[::-1]
        t10, t90 = lo + 0.1 * contrast, lo + 0.9 * contrast
        a10 = np.where(seg >= t10)[0]
        a90 = np.where(seg >= t90)[0]
        if len(a10) and len(a90):
            psf = max(0.4, abs(int(a90[0]) - int(a10[0])) / 2.563)
    rows_eff = max(1, int(0.8 * length_px / max(1.0, 2.0 * psf)))
    ch = ChannelConditions(contrast=max(contrast, 1e-6), noise_sigma=noise,
                           psf_sigma_px=psf, pixel_pitch_px=1.0, rows=rows_eff)
    # sub-pixel edge location on the aligned mean profile (parabola on |grad|)
    sub = 0.0
    if 0 < k < len(grad) - 1:
        den = grad[k - 1] - 2 * grad[k] + grad[k + 1]
        if abs(den) > 1e-9:
            sub = 0.5 * (grad[k - 1] - grad[k + 1]) / den
    offset = float(offs[0] + k + sub)
    return ch, 0.5 * (inner_level + outer_level), offset


def quad_information(
    image: np.ndarray,
    quad: np.ndarray,
    *,
    px_per_mm: Optional[float] = None,
    sensor: Optional[SensorModel] = None,
    samples: int = 40,
    border_mm: float = NOMINAL_BORDER_MM,
) -> QuadInformation:
    """Measure the information floor along each side of ``quad`` (TL, TR, BR,
    BL, clockwise, image pixels) in ``image``."""
    from .geometry import order_quad
    from .types import STANDARD_CARD_H_MM, STANDARD_CARD_W_MM

    sensor = sensor or SensorModel()
    q = order_quad(np.asarray(quad, dtype=np.float64).reshape(4, 2))
    lengths = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    if min(lengths) < 8.0:
        raise ValueError("quad too small to measure")
    if px_per_mm is None:
        short = 0.5 * (min(lengths[0], lengths[1]) + min(lengths[2], lengths[3]))
        long_ = 0.5 * (max(lengths[0], lengths[1]) + max(lengths[2], lengths[3]))
        px_per_mm = 0.5 * (short / STANDARD_CARD_W_MM + long_ / STANDARD_CARD_H_MM)
    # Wide enough to find the edge when the outline is off it. Contour halos
    # sit OUTSIDE the card (measured: 11 px on a 218 px side at tracking
    # resolution), so the window reaches further inward than outward.
    inner_px = float(np.clip(0.10 * min(lengths), 8.0, 40.0))
    outer_px = float(np.clip(0.05 * min(lengths), 6.0, 24.0))

    sides = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        P, offs, L = _profiles(image, a, b, inner_px, outer_px, samples)
        best = None
        for c in range(P.shape[0]):
            got = _channel_from_profile(P[c], offs, L)
            if got is None:
                continue
            ch, bg, off = got
            if best is None or ch.snr > best[0].snr:
                best = (ch, bg, c, off)
        if best is None:
            ch, bg, c, off = ChannelConditions(1e-6, 1.0, 1.0, 1.0, 1), 0.0, 0, 0.0
        else:
            ch, bg, c, off = best
        sigma = cramer_rao_edge_px(ch)
        ratio, _ = shot_noise_consistency(ch, bg, sensor)
        sides.append(SideInformation(
            side=SIDE_NAMES[i], channel=ch, channel_index=c, sigma_px=sigma,
            sigma_end_px=sigma * _END_FACTOR, resolvable=ch.contrast > 2.0 * ch.noise_sigma,
            background_level=bg, shot_ratio=ratio, length_px=L, offset_px=off))

    corners = []
    for i in range(4):
        prev_side, next_side = sides[(i - 1) % 4], sides[i]
        u = q[i] - q[(i - 1) % 4]
        v = q[(i + 1) % 4] - q[i]
        sin_t = abs(float(u[0] * v[1] - u[1] * v[0])) / max(
            float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-9)
        corners.append(math.hypot(prev_side.sigma_end_px, next_side.sigma_end_px)
                       / max(sin_t, 0.2))

    worst = max(sides, key=lambda s: s.sigma_px)
    sigma_pp = cramer_rao_ratio_pp(worst.channel, border_mm, border_mm, px_per_mm)
    shot = max(s.shot_ratio for s in sides)
    info = QuadInformation(tuple(sides), tuple(corners), float(px_per_mm),
                           float(sigma_pp), float(shot), ())
    return QuadInformation(info.sides, info.corner_sigma_px, info.px_per_mm,
                           info.sigma_cr_pp, info.shot_ratio, capture_advice(info))


def capture_advice(info: QuadInformation, max_shot_ratio: float = 1.6) -> tuple:
    """What to change, ranked by how much it would lower the floor."""
    worst = max(info.sides, key=lambda s: s.sigma_px)
    ch = worst.channel
    out = []
    if info.px_per_mm < TARGET_PX_PER_MM:
        out.append(Advice("closer", f"move closer: {info.px_per_mm:.1f} px/mm, "
                          f"aim for {TARGET_PX_PER_MM:.0f}",
                          (TARGET_PX_PER_MM / max(info.px_per_mm, 1e-6)) ** 1.5))
    if ch.snr < TARGET_SNR:
        out.append(Advice("contrast", f"the {worst.side} edge barely stands out "
                          f"(contrast/noise {ch.snr:.0f}); use a background that "
                          "contrasts with the card edge", TARGET_SNR / max(ch.snr, 1e-6)))
    if ch.psf_sigma_px > 1.2:
        out.append(Advice("steady", f"edges are soft ({ch.psf_sigma_px:.1f} px blur); "
                          "hold steady and let it focus",
                          math.sqrt(ch.psf_sigma_px / TARGET_PSF_PX)))
    if math.isfinite(info.shot_ratio) and info.shot_ratio > max_shot_ratio:
        # More light lowers read-noise and compression share relative to signal;
        # the gain is bounded by how far above the photon floor the frame sits.
        out.append(Advice("light", f"noise is {info.shot_ratio:.1f}x the photon floor; "
                          "add light", min(math.sqrt(info.shot_ratio / max_shot_ratio), 3.0)))
    for s in info.sides:
        if not s.resolvable:
            out.append(Advice("contrast", f"the {s.side} edge is not visible against "
                              "the background", float("inf")))
            break
    return tuple(sorted((a for a in out if a.gain > 1.05), key=lambda a: -a.gain))


def refine_quad_by_information(quad: np.ndarray, info: QuadInformation,
                               max_shift_px: Optional[float] = None) -> Optional[np.ndarray]:
    """Move each side onto the edge the information measurement found, then
    re-intersect. Returns None (keep the input) unless every side is
    resolvable and every shift is within ``max_shift_px``; a partial
    correction would bend the outline toward whichever sides happened to
    succeed."""
    from .geometry import order_quad

    if not info.resolvable:
        return None
    q = order_quad(np.asarray(quad, dtype=np.float64).reshape(4, 2))
    lines = []
    for i, side in enumerate(info.sides):
        if max_shift_px is not None and abs(side.offset_px) > max_shift_px:
            return None
        a, b = q[i], q[(i + 1) % 4]
        t = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
        n = np.array([t[1], -t[0]])
        p = a + side.offset_px * n
        lines.append((n[0], n[1], -float(n @ p)))
    out = []
    for i in range(4):
        l1, l2 = lines[(i - 1) % 4], lines[i]
        det = l1[0] * l2[1] - l2[0] * l1[1]
        if abs(det) < 1e-9:
            return None
        out.append([(l1[1] * l2[2] - l2[1] * l1[2]) / det,
                    (l1[2] * l2[0] - l2[2] * l1[0]) / det])
    return np.array(out, dtype=np.float64)


def snap_to_information(image: np.ndarray, quad: np.ndarray, *, samples: int = 40,
                        px_per_mm: Optional[float] = None):
    """Measure the outline; if every side found its edge somewhere else,
    move the sides there and measure again. The move is kept only if the
    second measurement puts every side on its edge (|offset| < 1.5 px).
    Returns (quad, QuadInformation, moved)."""
    info = quad_information(image, quad, samples=samples, px_per_mm=px_per_mm)
    if all(abs(s.offset_px) < 1.5 for s in info.sides):
        return np.asarray(quad, dtype=np.float64).reshape(4, 2), info, False
    refined = refine_quad_by_information(quad, info)
    if refined is None:
        return np.asarray(quad, dtype=np.float64).reshape(4, 2), info, False
    try:
        info2 = quad_information(image, refined, samples=samples, px_per_mm=px_per_mm)
    except ValueError:
        return np.asarray(quad, dtype=np.float64).reshape(4, 2), info, False
    if info2.resolvable and all(abs(s.offset_px) < 1.5 for s in info2.sides):
        return refined, info2, True
    return np.asarray(quad, dtype=np.float64).reshape(4, 2), info, False


def locate_card(image: np.ndarray, prefer_point=None, min_area_frac: float = 0.008,
                search=None, judged_only: bool = False):
    """Find the card at ``prefer_point``, then snap it with ``snap_to_information``.

    Two detectors, because each fails where the other does not:

    * ``geometry.find_card_quad`` (threshold, contour, nesting) is the more
      precise when the card stands out from a plain dark mat -- its outline
      is sub-pixel there -- and it knows about slabs and viewports.
    * ``scene.SceneSearch`` (straight edges, polarity, card shape through the
      perspective) is the one that works on a shop counter: silver edges on
      light wood, 20-45 degrees of tilt, sleeves, several cards touching. On
      the field frames in tests/fixtures/field find_card_quad found the card
      in 0 of 17 checks; on synthetic cards against a light background its
      corners were 3-23 px out where the scene search's were 1-2 px.

    When they agree (IoU >= 0.85) the contour detector's outline is used;
    when they disagree the scene search's is, because it is the one whose
    choice was judged against the picture. Either way the information snap
    then decides WHERE each boundary is. Measured on synthetic captures at the
    540 px tracking size, find_card_quad's outline sat about one border width
    outside the card in 9 of 10 frames; after the snap the same frames were
    within 3 px of the full-resolution detection.

    ``search``: a ``SceneSearch`` already built on this same image (the live
    session builds one per frame to check its tracked outline, and building
    it is most of the cost). ``judged_only``: use the contour detector's
    outline only when the scene search judges it a card too. The live loop
    asks for this; alone, the contour detector returns shapes on a counter
    that are not cards.

    Returns (quad, contour, residual_px, QuadInformation, moved).
    """
    from .geometry import enforce_portrait, find_card_quad, order_quad
    from .scene import SceneSearch
    from .types import DetectionError

    h, w = image.shape[:2]
    aim = prefer_point if prefer_point is not None else (w / 2.0, h / 2.0)
    legacy = None
    try:
        legacy = find_card_quad(image, min_area_frac=min_area_frac, prefer_point=prefer_point)
    except DetectionError:
        legacy = None
    hit = None
    try:
        if search is None:
            search = SceneSearch(image, long_side=min(1400, max(h, w)))
        hit = search.card_at(aim)
    except (DetectionError, cv2.error, np.linalg.LinAlgError, ValueError):
        hit = None
    keep_legacy = False
    if legacy is not None and hit is not None:
        # The contour detector's outline stands when the picture supports it
        # about as well and it is the same card (the card inside its sleeve,
        # or the same edge found more precisely).
        judged = search.judge(legacy[0])
        keep_legacy = (judged is not None and judged.score >= hit.score - 0.15
                       and _quad_iou(legacy[0], hit.quad, (h, w)) >= 0.6)

    if hit is not None and not keep_legacy:
        quad = enforce_portrait(order_quad(hit.quad))
        contour = quad.reshape(-1, 1, 2).astype(np.int32)
        residual = float(hit.residual_px)
    elif legacy is not None and (not judged_only or (
            search is not None and search.judge(legacy[0]) is not None)):
        quad, contour, residual = legacy
    else:
        # The contour detector's own refusal ("shoot against a plain
        # contrasting background") is advice nobody can take at a shop
        # counter; say what the user can do there.
        raise DetectionError(
            (search.refusal if search is not None and search.refusal else "")
            or "no card found here -- point the centre of the view at a card, or tap the card on screen")
    try:
        snapped, info, moved = snap_to_information(image, quad)
    except ValueError:
        return quad, contour, residual, None, False
    if moved:
        snapped = enforce_portrait(order_quad(snapped))
    return snapped, contour, residual, info, moved


def _quad_iou(a, b, shape) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(4, 2)
    b = np.asarray(b, dtype=np.float32).reshape(4, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    ua = abs(cv2.contourArea(a)) + abs(cv2.contourArea(b)) - inter
    return float(inter / ua) if ua > 0 else 0.0
