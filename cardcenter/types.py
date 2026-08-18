"""Core data types.

Design rule for this whole package: every physical quantity that is measured
carries an uncertainty. Nothing returns a bare float that was actually an
estimate. If we cannot bound the error on a number, we do not report the number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# Standard modern trading card, 2.5 x 3.5 inches.
STANDARD_CARD_W_MM = 63.5
STANDARD_CARD_H_MM = 88.9


@dataclass(frozen=True)
class Measured:
    """A scalar with a 1-sigma uncertainty."""

    value: float
    sigma: float = 0.0

    def interval(self, k: float = 1.96) -> tuple[float, float]:
        """Two-sided interval at k sigma. k=1.96 is ~95%."""
        return (self.value - k * self.sigma, self.value + k * self.sigma)

    def __add__(self, other: "Measured") -> "Measured":
        return Measured(self.value + other.value, math.hypot(self.sigma, other.sigma))

    def __sub__(self, other: "Measured") -> "Measured":
        return Measured(self.value - other.value, math.hypot(self.sigma, other.sigma))

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"{self.value:.4g}±{self.sigma:.2g}"


@dataclass(frozen=True)
class SlabSpec:
    """Optical description of the holder the card sits in.

    ``acrylic_thickness_mm`` is the thickness of *solid plastic* the light
    actually traverses on the viewed face -- NOT the overall slab depth. Air
    gaps and inner sleeves contribute no lateral displacement (n=1 gives zero
    shift), so lumping total slab thickness into this parameter overestimates
    the correction, often by 2x or more.

    Measure the face wall of your own slabs with calipers. The defaults below
    are plausible starting points, not published manufacturer specs, and are
    given generous uncertainties to reflect that.
    """

    name: str = "raw"
    acrylic_thickness_mm: float = 0.0
    acrylic_thickness_sigma_mm: float = 0.0
    refractive_index: float = 1.0
    refractive_index_sigma: float = 0.0

    @property
    def is_optically_active(self) -> bool:
        return self.acrylic_thickness_mm > 1e-6 and self.refractive_index > 1.0 + 1e-6

    @property
    def layers(self) -> tuple["SlabSpec", ...]:
        """Uniform interface with SlabStack so optics code handles both."""
        return (self,)


@dataclass(frozen=True)
class SlabStack:
    """Several plane-parallel layers in the optical path.

    Shooting a slabbed card inside a display case means light crosses the slab
    acrylic AND the case glass. Both displace, and the displacements add: each
    layer contributes t_i*(tan(theta1) - tan(theta2_i)) independently, because
    the ray re-emerges parallel after every layer. Air gaps between layers
    contribute nothing.
    """

    name: str
    parts: tuple[SlabSpec, ...]

    @property
    def layers(self) -> tuple[SlabSpec, ...]:
        return self.parts

    @property
    def is_optically_active(self) -> bool:
        return any(p.is_optically_active for p in self.parts)

    @property
    def acrylic_thickness_mm(self) -> float:
        return sum(p.acrylic_thickness_mm for p in self.parts)

    @property
    def refractive_index(self) -> float:
        """Thickness-weighted mean, for display only. The real computation
        iterates the layers; do not use this as if it were a single medium."""
        t = self.acrylic_thickness_mm
        if t <= 0:
            return 1.0
        return sum(p.acrylic_thickness_mm * p.refractive_index for p in self.parts) / t


def stack_slabs(name: str, *parts: SlabSpec) -> SlabStack:
    return SlabStack(name=name, parts=tuple(p for p in parts if p.is_optically_active))


# Refractive indices here are standard material values (sodium D line, ~589nm)
# and are solid. The thicknesses are estimates and flagged as such.
SLAB_PRESETS: dict[str, SlabSpec] = {
    "raw": SlabSpec(name="raw"),
    "penny_sleeve": SlabSpec(
        name="penny_sleeve",
        acrylic_thickness_mm=0.05,
        acrylic_thickness_sigma_mm=0.02,
        refractive_index=1.50,  # polypropylene
        refractive_index_sigma=0.02,
    ),
    "toploader": SlabSpec(
        name="toploader",
        acrylic_thickness_mm=0.55,
        acrylic_thickness_sigma_mm=0.15,
        refractive_index=1.58,  # PVC / polystyrene blend
        refractive_index_sigma=0.03,
    ),
    "psa": SlabSpec(
        name="psa",
        acrylic_thickness_mm=1.20,
        acrylic_thickness_sigma_mm=0.30,
        refractive_index=1.59,  # polystyrene
        refractive_index_sigma=0.02,
    ),
    "bgs": SlabSpec(
        name="bgs",
        acrylic_thickness_mm=1.60,
        acrylic_thickness_sigma_mm=0.40,
        refractive_index=1.49,  # PMMA acrylic
        refractive_index_sigma=0.01,
    ),
    "cgc": SlabSpec(
        name="cgc",
        acrylic_thickness_mm=1.40,
        acrylic_thickness_sigma_mm=0.40,
        refractive_index=1.49,
        refractive_index_sigma=0.01,
    ),
    # Retail display cases are typically 4-6mm soda-lime glass. n=1.52 is solid;
    # the thickness varies by shop and is given a wide sigma to reflect that you
    # will not be measuring the store's cabinet with calipers.
    "case_glass": SlabSpec(
        name="case_glass",
        acrylic_thickness_mm=5.0,
        acrylic_thickness_sigma_mm=1.5,
        refractive_index=1.52,
        refractive_index_sigma=0.02,
    ),
}


def _stack(name: str, *keys: str) -> "SlabStack":
    return SlabStack(name=name, parts=tuple(SLAB_PRESETS[k] for k in keys))


# Composite paths for shooting into a display case.
SLAB_STACKS: dict[str, "SlabStack"] = {
    "case_raw": _stack("case_raw", "case_glass"),
    "case_toploader": _stack("case_toploader", "toploader", "case_glass"),
    "case_psa": _stack("case_psa", "psa", "case_glass"),
    "case_bgs": _stack("case_bgs", "bgs", "case_glass"),
    "case_cgc": _stack("case_cgc", "cgc", "case_glass"),
}


def resolve_holder(name: str):
    """Look up a holder by name across both single slabs and stacks."""
    if name in SLAB_PRESETS:
        return SLAB_PRESETS[name]
    if name in SLAB_STACKS:
        return SLAB_STACKS[name]
    options = ", ".join(sorted(set(SLAB_PRESETS) | set(SLAB_STACKS)))
    raise KeyError(f"unknown holder '{name}'. Available: {options}")


@dataclass(frozen=True)
class CaptureSpec:
    """What we know about the camera.

    ``focal_px`` is the pinhole focal length in pixels. If it is None we cannot
    recover camera pose, which means we cannot do a per-pixel refraction
    correction. In that case the pipeline either skips the correction (raw
    cards, where it is a no-op anyway) or refuses to report a slabbed
    measurement as precise.
    """

    focal_px: Optional[float] = None
    principal_point: Optional[tuple[float, float]] = None
    # Physical cut tolerance of the card blank itself. Even a perfect
    # measurement of a real card inherits the die-cut variation.
    cut_tolerance_sigma_mm: float = 0.05

    def intrinsics(self, image_shape: tuple[int, int]) -> Optional[np.ndarray]:
        if self.focal_px is None:
            return None
        h, w = image_shape[:2]
        cx, cy = self.principal_point or (w / 2.0, h / 2.0)
        return np.array(
            [[self.focal_px, 0.0, cx], [0.0, self.focal_px, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @staticmethod
    def from_fov(fov_deg: float, image_shape: tuple[int, int]) -> "CaptureSpec":
        """Build intrinsics from horizontal field of view."""
        h, w = image_shape[:2]
        f = (w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        return CaptureSpec(focal_px=f)


@dataclass(frozen=True)
class BorderPair:
    """Two opposing border widths and the ratio between them."""

    axis: str  # "horizontal" or "vertical"
    low_name: str  # e.g. "left"
    high_name: str  # e.g. "right"
    low_mm: Measured
    high_mm: Measured

    @property
    def total_mm(self) -> float:
        return self.low_mm.value + self.high_mm.value

    @property
    def ratio_pct(self) -> Measured:
        """Percentage of the border budget on the *wider* side.

        Always >= 50. This is the convention graders use: a card described as
        60/40 has 60% of its border margin on one side. Which side is captured
        separately in ``skew_toward``.
        """
        a, b = self.low_mm.value, self.high_mm.value
        total = a + b
        if total <= 1e-9:
            return Measured(50.0, 50.0)
        wider, narrower = (a, b) if a >= b else (b, a)
        s_w = self.low_mm.sigma if a >= b else self.high_mm.sigma
        s_n = self.high_mm.sigma if a >= b else self.low_mm.sigma
        value = 100.0 * wider / total
        # d(w/(w+n))/dw = n/(w+n)^2 ; d/dn = -w/(w+n)^2
        sigma = 100.0 * math.hypot(narrower * s_w, wider * s_n) / (total**2)
        return Measured(value, sigma)

    @property
    def skew_toward(self) -> str:
        return self.low_name if self.low_mm.value >= self.high_mm.value else self.high_name

    def describe(self) -> str:
        r = self.ratio_pct
        lo, hi = r.interval()
        return (
            f"{self.axis}: {r.value:.1f}/{100 - r.value:.1f} "
            f"(95% CI {lo:.1f}-{hi:.1f}), wider on {self.skew_toward}"
        )


@dataclass
class DetectionQuality:
    """Honest reporting of how much the detector actually trusts itself."""

    outer_residual_px: float = 0.0
    inner_confidence: float = 0.0  # 0..1, from edge profile prominence
    inner_confidence_per_side: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    refraction_applied: bool = False
    max_refraction_shift_mm: float = 0.0

    @property
    def usable(self) -> bool:
        return self.inner_confidence >= 0.35 and self.outer_residual_px < 6.0


@dataclass
class CenteringResult:
    horizontal: BorderPair
    vertical: BorderPair
    quality: DetectionQuality
    px_per_mm: float
    corners_px: np.ndarray
    inner_rect_mm: tuple[float, float, float, float]  # l, t, r, b in card mm coords
    slab: SlabSpec
    rectified: Optional[np.ndarray] = None

    @property
    def worst_axis(self) -> BorderPair:
        """Graders take the worst of the two axes."""
        return max((self.horizontal, self.vertical), key=lambda p: p.ratio_pct.value)

    @property
    def worst_ratio(self) -> Measured:
        return self.worst_axis.ratio_pct

    def summary(self) -> str:
        lines = [
            self.horizontal.describe(),
            self.vertical.describe(),
            f"worst axis: {self.worst_axis.axis}",
        ]
        if self.quality.refraction_applied:
            lines.append(
                f"refraction correction applied "
                f"(max {self.quality.max_refraction_shift_mm:.3f} mm)"
            )
        for w in self.quality.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)


class DetectionError(RuntimeError):
    """Raised when the image cannot be measured. We fail loudly rather than
    returning a confident-looking number derived from a bad detection."""
