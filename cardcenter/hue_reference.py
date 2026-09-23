"""Card-back hue references: learned, not hard-coded, and judged against the
noise the frame actually has.

``ingest.binder_sticker`` used two constants (Pokemon 108, Yu-Gi-Oh 12, in
OpenCV's 0..180 hue units) and a fixed tolerance of 18. This module replaces
them with:

* ``HueReference`` -- a franchise's back-design hue, its between-card spread,
  the number of confirmed backs behind it, and where it came from.
* ``HueRegistry`` -- resolved in order from ``BAKUGO_HUE_REFERENCES`` (a JSON
  file), the released ``back_hue`` artifact (``cardcenter.release_guard``;
  trained by the private trainer on confirmed backs), then the built-in
  defaults, which carry ``source`` so an uncalibrated reference is visible as
  such.
* ``measure_back_hue`` -- the circular mean hue of a back crop and its
  standard error. Per-pixel hue noise comes from the same channel model as
  ``information.py``: a chroma C with per-channel noise sigma_n moves the hue
  angle by about sigma_n / C radians, and pixels closer together than the
  blur are not independent, so N_eff = N / (2 sigma_p)^2.
* ``classify_back`` -- the distance to each reference in sigmas,
  sqrt(spread^2 + se^2), accepted within ``k_sigma`` (3, the same k the grade
  gate in ``confidence.py`` uses) and only when the runner-up is at least
  ``min_margin_sigma`` further away.

Price labels. Stickers vary by shop and many cards have none, so the label
mask is a profile, not a fixed hue band. White or unsaturated labels are
always excluded. A coloured label band (the orange price-gun bar of the
original binder set) is excluded only while it covers less than
``max_label_frac`` of the usable pixels: a label is small, and a band that
covers most of the back is the back itself. That keeps a brown Yu-Gi-Oh back
(hue near 12, inside the orange band) classifiable, which the fixed mask did
not.
"""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import cv2
import numpy as np

HUE_PERIOD = 180.0
_RAD_TO_HUE = HUE_PERIOD / (2.0 * math.pi)


@dataclass(frozen=True)
class HueReference:
    franchise: str
    hue: float              # OpenCV units, 0..180
    spread: float           # circular std across confirmed backs, same units
    n: int = 0              # confirmed backs behind the estimate
    source: str = "default"  # default | uncalibrated | released | file | in_situ

    @property
    def calibrated(self) -> bool:
        return self.source not in ("uncalibrated",)


# Pokemon 108 was measured on the 2026-09-14 binder session (blue swirl backs
# clustered 105-115). Yu-Gi-Oh 12 was never measured. Spread 6 reproduces the
# old fixed tolerance (18 = 3 sigma) until confirmed backs replace it.
DEFAULT_REFERENCES = (
    HueReference("pokemon", 108.0, 6.0, 0, "default"),
    HueReference("yugioh", 12.0, 6.0, 0, "uncalibrated"),
)


@dataclass(frozen=True)
class LabelProfile:
    """How price labels look in this capture set."""

    exclude_unsaturated: bool = True           # white / silver labels
    bands: tuple = ((0.0, 25.0),)              # coloured label hue ranges
    band_min_saturation: int = 80
    band_min_value: int = 80
    max_label_frac: float = 0.35

    @classmethod
    def none(cls) -> "LabelProfile":
        return cls(exclude_unsaturated=True, bands=())


@dataclass(frozen=True)
class HueMeasurement:
    hue: Optional[float]
    se: float               # standard error of the mean hue, OpenCV units
    spread: float           # circular std of hue inside this crop
    n_valid: int
    valid_frac: float
    label_frac: float       # fraction excluded as a coloured label
    noise_sigma: float
    note: str = ""


@dataclass(frozen=True)
class BackClassification:
    guess: Optional[str]
    confidence: float
    note: str
    distances: Mapping[str, float] = field(default_factory=dict)   # sigmas
    measurement: Optional[HueMeasurement] = None


def circ_dist(a: float, b: float) -> float:
    d = abs(a - b) % HUE_PERIOD
    return min(d, HUE_PERIOD - d)


def circular_mean_std(hues) -> tuple[float, float]:
    ang = np.asarray(hues, dtype=np.float64) / _RAD_TO_HUE
    c, s = float(np.cos(ang).mean()), float(np.sin(ang).mean())
    R = min(1.0, math.hypot(c, s))
    mean = (math.atan2(s, c) * _RAD_TO_HUE) % HUE_PERIOD
    std = math.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-12)))) * _RAD_TO_HUE
    return mean, std


def _noise_sigma(img: np.ndarray) -> float:
    """Per-pixel noise from the high-pass residual (MAD), grey levels."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    resid = g - cv2.blur(g, (3, 3))
    # a 3x3 box removes 1/9 of the pixel's own noise: scale back
    return max(0.5, float(np.median(np.abs(resid))) * 1.4826 / math.sqrt(1.0 - 1.0 / 9.0))


def measure_back_hue(back_bgr: np.ndarray, profile: LabelProfile = LabelProfile(),
                     psf_px: float = 1.0) -> HueMeasurement:
    if back_bgr is None or back_bgr.size == 0:
        return HueMeasurement(None, float("inf"), 0.0, 0, 0.0, 0.0, 0.0, "empty crop")
    hsv = cv2.cvtColor(back_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    usable = (v > 25) & (v < 250) & (s > 25)
    if profile.exclude_unsaturated:
        usable &= ~((s < 60) & (v > 150))
    label = np.zeros_like(usable)
    for lo, hi in profile.bands:
        label |= (h >= lo) & (h < hi) & (s > profile.band_min_saturation) & (v > profile.band_min_value)
    label &= usable
    n_usable = int(usable.sum())
    label_frac = float(label.sum()) / max(n_usable, 1)
    valid = usable & ~label if label_frac < profile.max_label_frac else usable
    total = h.size
    n = int(valid.sum())
    noise = _noise_sigma(back_bgr)
    if n < 0.15 * total:
        return HueMeasurement(None, float("inf"), 0.0, n, n / total, label_frac, noise,
                              "too little label-free, glare-free area to classify")
    hues = h[valid].astype(np.float64)
    mean, spread = circular_mean_std(hues)
    chroma = (s[valid].astype(np.float64) * v[valid].astype(np.float64)) / 255.0
    per_pixel = float(np.median(noise / np.maximum(chroma, 1.0))) * _RAD_TO_HUE
    n_eff = max(1.0, n / max(1.0, (2.0 * psf_px) ** 2))
    se = math.sqrt(per_pixel ** 2 + spread ** 2) / math.sqrt(n_eff)
    return HueMeasurement(mean, se, spread, n, n / total, label_frac, noise)


class HueRegistry:
    def __init__(self, references):
        self._refs = {r.franchise: r for r in references}

    def __iter__(self):
        return iter(self._refs.values())

    def get(self, franchise: str) -> Optional[HueReference]:
        return self._refs.get(franchise)

    def with_reference(self, ref: HueReference) -> "HueRegistry":
        refs = dict(self._refs)
        refs[ref.franchise] = ref
        return HueRegistry(refs.values())

    # flat payload, the shape release_guard accepts
    def to_params(self) -> dict:
        out = {}
        for r in self._refs.values():
            out[f"{r.franchise}_hue"] = round(r.hue, 3)
            out[f"{r.franchise}_spread"] = round(r.spread, 3)
            out[f"{r.franchise}_n"] = int(r.n)
        return out

    @classmethod
    def from_params(cls, params: Mapping, source: str, base: Optional["HueRegistry"] = None):
        refs = {r.franchise: r for r in (base or cls(DEFAULT_REFERENCES))}
        for key, value in params.items():
            if not key.endswith("_hue"):
                continue
            fr = key[: -len("_hue")]
            spread = float(params.get(f"{fr}_spread", refs.get(fr, HueReference(fr, 0, 6.0)).spread))
            n = int(params.get(f"{fr}_n", 0))
            refs[fr] = HueReference(fr, float(value) % HUE_PERIOD, max(spread, 0.5), n, source)
        return cls(refs.values())

    def to_json(self) -> str:
        return json.dumps([asdict(r) for r in self._refs.values()], indent=1)


_lock = threading.Lock()
_cached: Optional[HueRegistry] = None


def load_registry(refresh: bool = False) -> HueRegistry:
    global _cached
    with _lock:
        if _cached is not None and not refresh:
            return _cached
        reg = HueRegistry(DEFAULT_REFERENCES)
        try:
            from .release_guard import released_params

            params = released_params("back_hue")
            if params:
                reg = HueRegistry.from_params(params, "released", reg)
        except Exception:
            pass
        path = os.environ.get("BAKUGO_HUE_REFERENCES")
        if path and Path(path).is_file():
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                reg = HueRegistry.from_params(data, "file", reg)
            else:
                for row in data:
                    reg = reg.with_reference(HueReference(**{**row, "source": row.get("source", "file")}))
        _cached = reg
        return reg


def classify_back(meas: HueMeasurement, registry: Optional[HueRegistry] = None,
                  k_sigma: float = 3.0, min_margin_sigma: float = 1.0,
                  uncalibrated_discount: float = 0.6) -> BackClassification:
    registry = registry or load_registry()
    if meas.hue is None:
        return BackClassification(None, 0.0, meas.note, {}, meas)
    dist = {}
    for r in registry:
        dist[r.franchise] = circ_dist(meas.hue, r.hue) / math.hypot(r.spread, meas.se)
    ranked = sorted(dist.items(), key=lambda kv: kv[1])
    best, z = ranked[0]
    ref = registry.get(best)
    if z > k_sigma:
        return BackClassification(None, 0.0, f"mean hue {meas.hue:.0f} matches no reference "
                                  f"(nearest {best} at {z:.1f} sigma)", dist, meas)
    if len(ranked) > 1 and ranked[1][1] <= k_sigma and ranked[1][1] - z < min_margin_sigma:
        return BackClassification(None, 0.0, f"mean hue {meas.hue:.0f} is between {best} and "
                                  f"{ranked[1][0]}", dist, meas)
    conf = max(0.0, 1.0 - z / k_sigma)
    tag = ""
    if not ref.calibrated:
        conf *= uncalibrated_discount
        tag = " (uncalibrated)"
    return BackClassification(best, conf, f"mean hue {meas.hue:.0f} near{tag} {best} reference "
                              f"({z:.1f} sigma)", dist, meas)


def fit_reference(franchise: str, per_back_hues, source: str, min_spread: float = 2.0) -> HueReference:
    """A reference from confirmed backs: circular mean of their mean hues,
    spread = their circular std (floored: a handful of backs from one print
    run understates the spread across runs)."""
    mean, std = circular_mean_std(per_back_hues)
    return HueReference(franchise, round(mean, 3), round(max(std, min_spread), 3),
                        len(list(per_back_hues)), source)
