"""photometrology — one photogrammetric contract for every Holon app.

Canonical source: ``Metrology/photometrology.py`` in the umbrella repo.
Vendored byte-for-byte into Perceptopoly and StockOpoly by
``Metrology/sync.py``; ported to TypeScript for GeoGraph OCR
(``Loadopoly-OCR/src/lib/photoMetrology.ts``). ``golden_vectors.json`` pins
the numbers every port must reproduce. Stdlib only.

Model
-----
A physical length L seen in a photo spans p pixels on some plane, so the
photo's scale on that plane is ``s = L / p`` (mm per pixel). Every source of
scale is reduced to the same thing — ``ln s`` with a log-space sigma — and
fused by inverse variance. Working in log space makes errors relative, so a
1 mm miss on a card and a 1 ft miss on a rack beam are judged alike.

Sources of scale, each tagged with a source class:

* ``measured``  — a tape, caliper or a person's calibrated local standard
                  (finger, palm, shoe) in the frame.
* ``recorded``  — a known-size reference from a published spec (ISO card,
                  pegboard pitch, pallet footprint).
* ``projected`` — the camera alone: EXIF focus distance + 35 mm-equivalent
                  focal length. Describes the *focus* plane only, loosely.
* ``simulated`` — game/synthetic frames. Unit is ``tile``; never converts.

Local standards are personal: a body measurement belongs to one person and
is only ever applied to that person's photos. People are linked through
*transfer objects* — the same physical thing photographed by both — never by
assuming one person's finger equals another's.

Photos are identified across apps by content: ``photo_key = sha256:<hex>``
of the original bytes, so the same JPEG resolves to the same key in
StockOpoly, GeoGraph OCR and anywhere else.

Three further layers sit on the same scale model:

* **Resolution floors (the Planck gate).** The Planck length is the absolute
  floor; above it sit the quantum (shot-noise) and Cramér–Rao floors on
  locating an edge, the same physics as Bakugo's ``information.py``.
  ``planck_floor`` turns them into the smallest resolvable length on a plane;
  ``planck_gate`` is QUIPU's rule — a difference below the floor is not a
  displacement.
* **Interstitial space.** The physical space between things (card borders,
  clearances) via ``interstitial_gap``; and QUIPU's interstitial arc — the
  unmediated coherence between the Perceptopoly (perception), OCR (vision)
  and Bakugo (touch) outputs — applied to their metrology residuals
  (``interstitial_arc``, recorded, not routed).
* **Grading components.** Bakugo's aspects (centering, corners, edges,
  surface) as measurements: ``centering`` from the four borders in pixels
  (scale cancels), ``grading_component`` for defect lengths/areas. Mapping a
  measurement to a grade stays in Bakugo.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "SCHEMA", "VERSION", "SOURCE_CLASSES", "PRIVACY", "UNIT_TO_MM",
    "FULL_FRAME_DIAG_MM", "EXIF_SIGMA_LN", "OUTLIER_Z",
    "photo_key", "to_mm", "Scale", "Length", "LocalStandard", "Reference",
    "standard_from_frames", "scale_from_span", "scale_from_exif",
    "fuse", "measure", "relational", "tape_span", "grid_scale", "transfer_offsets",
    "error_stats", "public_view", "REFERENCES",
    # resolution floors / Planck gate
    "PLANCK_LENGTH_M", "PLANCK_LENGTH_MM", "cramer_rao_edge_px", "quantum_floor_edge_px",
    "planck_floor", "planck_gate",
    # interstitial space
    "interstitial_gap", "INTERSTITIAL_ARC", "interstitial_arc",
    # grading components
    "ASPECTS", "centering", "grading_component",
    # dynamic parameters
    "PRIOR_WEIGHT_N", "PARALLAX_PRIOR_MM", "shrink", "default_params", "params_from_stats",
    "exif_params", "grid_k", "fit_grid_sigma_k", "edge_sigma_from_profiles", "solve_parallax",
    "load_params",
]

SCHEMA = "holon.metrology/1"
VERSION = "1.0.0"
SOURCE_CLASSES = ("measured", "recorded", "projected", "simulated")
PRIVACY = ("public", "tenant", "personal")
UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "ft": 304.8}
FULL_FRAME_DIAG_MM = math.hypot(36.0, 24.0)  # 43.2666 mm, defines "35 mm equiv."
# EXIF focus distance is coarse (quantised, measured to the focus plane, worst
# at macro range). 0.35 in log space ≈ ±42 % at 1σ — a prior, not an anchor.
EXIF_SIGMA_LN = 0.35
OUTLIER_Z = 3.0
# grid_scale uncertainty constant, fitted on synthetic grids (see grid_scale)
GRID_SIGMA_K = 3.0


# ───────────────────────────────────────────────────────────────── identity
def photo_key(data: bytes | str) -> str:
    """Content key shared by every app: ``sha256:<hex>`` of the original bytes.

    Pass the file bytes, or a path (str) to read. Re-encoded or resized copies
    get a different key on purpose: their pixel geometry differs.
    """
    if isinstance(data, str):
        with open(data, "rb") as fh:
            data = fh.read()
    return "sha256:" + hashlib.sha256(data).hexdigest()


def to_mm(value: float, unit: str) -> float:
    if unit == "tile":
        raise ValueError("tile is a game unit with no physical size; keep it as tiles")
    try:
        return float(value) * UNIT_TO_MM[unit]
    except KeyError:
        raise ValueError(f"unknown unit {unit!r}; use one of {sorted(UNIT_TO_MM)}") from None


# ─────────────────────────────────────────────────────────────── value types
@dataclass(frozen=True)
class Scale:
    """mm per pixel on one plane of one photo, carried in log space."""

    ln_mm_per_px: float
    sigma_ln: float
    source_class: str
    basis: str = ""            # what produced it: "standard:<id>", "exif", ...
    plane: str = "default"

    def __post_init__(self) -> None:
        if self.source_class not in SOURCE_CLASSES:
            raise ValueError(f"source_class must be one of {SOURCE_CLASSES}")
        if not (self.sigma_ln > 0 and math.isfinite(self.sigma_ln)):
            raise ValueError("sigma_ln must be positive and finite")

    @property
    def mm_per_px(self) -> float:
        return math.exp(self.ln_mm_per_px)

    @property
    def rel_sigma(self) -> float:
        return math.expm1(self.sigma_ln)

    def to_record(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "kind": "scale", **asdict(self),
                "mm_per_px": round(self.mm_per_px, 9)}


@dataclass(frozen=True)
class Length:
    """A length in mm with a log-space sigma (relative uncertainty)."""

    mm: float
    sigma_ln: float
    source_class: str
    basis: str = ""

    @property
    def sigma_mm(self) -> float:
        return self.mm * self.sigma_ln

    def to_record(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "kind": "length", **asdict(self),
                "sigma_mm": round(self.sigma_mm, 6)}


@dataclass(frozen=True)
class LocalStandard:
    """One person's calibrated body (or personal-object) dimension.

    ``part`` names a repeatable definition, e.g. ``hand.breadth.mid_palm`` or
    ``finger.index.middle_phalanx.width``. ``evidence`` lists the frames and
    readings it was built from so anyone can audit it.
    """

    owner: str
    part: str
    value_mm: float
    sigma_mm: float
    method: str
    calibrated_at: str
    valid_until: str | None = None
    definition: str = ""
    evidence: tuple[dict[str, Any], ...] = ()
    privacy: str = "personal"

    def __post_init__(self) -> None:
        if self.value_mm <= 0 or self.sigma_mm <= 0:
            raise ValueError("value_mm and sigma_mm must be positive")
        if self.privacy not in PRIVACY:
            raise ValueError(f"privacy must be one of {PRIVACY}")

    @property
    def key(self) -> str:
        return f"{self.owner}:{self.part}"

    @property
    def sigma_ln(self) -> float:
        return math.log1p(self.sigma_mm / self.value_mm)

    def to_record(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"] = list(self.evidence)
        return {"schema": SCHEMA, "kind": "local_standard", **d}

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> "LocalStandard":
        fields = {k: rec[k] for k in (
            "owner", "part", "value_mm", "sigma_mm", "method", "calibrated_at")}
        return cls(**fields, valid_until=rec.get("valid_until"),
                   definition=rec.get("definition", ""),
                   evidence=tuple(rec.get("evidence") or ()),
                   privacy=rec.get("privacy", "personal"))


@dataclass(frozen=True)
class Reference:
    """A published, non-personal known size (source class ``recorded``)."""

    key: str
    name: str
    length_mm: float
    sigma_mm: float
    note: str = ""

    @property
    def sigma_ln(self) -> float:
        return math.log1p(self.sigma_mm / self.length_mm)


# Shared catalogue. Sigmas are manufacturing spread, not reading error.
REFERENCES: dict[str, Reference] = {r.key: r for r in (
    Reference("iso_id1.width", "ISO/IEC 7810 ID-1 card, long edge", 85.60, 0.13,
              "bank/credit card"),
    Reference("iso_id1.height", "ISO/IEC 7810 ID-1 card, short edge", 53.98, 0.13),
    Reference("tcg_standard.width", "Standard TCG card (Pokémon), short edge", 63.0, 0.5,
              "caliper reading 62.47 mm on 2026-08-14 sits inside this spread"),
    Reference("tcg_standard.height", "Standard TCG card (Pokémon), long edge", 88.0, 0.5),
    Reference("pegboard.pitch_1in", "Pegboard hole pitch (1 in grid)", 25.4, 0.25),
    Reference("rack.teardrop_pitch_2in", "Teardrop pallet-rack upright punch pitch", 50.8, 0.5,
              "industry-typical; confirm on site with one tape shot"),
    Reference("pallet.gma.length", "GMA pallet, 48 in side", 1219.2, 6.0),
    Reference("pallet.gma.width", "GMA pallet, 40 in side", 1016.0, 6.0),
    Reference("paper.us_letter.width", "US Letter, short edge", 215.9, 0.5),
    Reference("paper.a4.width", "A4, short edge", 210.0, 0.5),
)}


# ───────────────────────────────────────────────────────── building standards
def standard_from_frames(owner: str, part: str, frames: Sequence[dict[str, Any]], *,
                         method: str, calibrated_at: str,
                         systematic_rel: float = 0.0,
                         valid_until: str | None = None,
                         definition: str = "") -> LocalStandard:
    """Combine per-frame readings (``{"value_mm": .., "sigma_mm": .., ...}``).

    Value is the inverse-variance mean. Sigma combines the statistical term
    (larger of the propagated sigma and the observed frame-to-frame scatter
    over √n) with a systematic term that repeats in every frame — e.g. the
    parallax of a hand sitting behind the plane of the tape.
    """
    if not frames:
        raise ValueError("need at least one frame")
    w = [1.0 / (float(f["sigma_mm"]) ** 2) for f in frames]
    vals = [float(f["value_mm"]) for f in frames]
    mean = sum(wi * v for wi, v in zip(w, vals)) / sum(w)
    prop = 1.0 / math.sqrt(sum(w))
    n = len(vals)
    if n > 1:
        sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
        stat = max(prop, sd / math.sqrt(n))
    else:
        stat = prop
    sigma = math.hypot(stat, systematic_rel * mean)
    return LocalStandard(owner=owner, part=part, value_mm=round(mean, 4),
                         sigma_mm=round(sigma, 4), method=method,
                         calibrated_at=calibrated_at, valid_until=valid_until,
                         definition=definition, evidence=tuple(frames))


# ─────────────────────────────────────────────────────────── scale sources
def scale_from_span(length_mm: float, length_sigma_ln: float, pixels: float,
                    pixel_sigma: float, *, source_class: str, basis: str,
                    plane: str = "default") -> Scale:
    """A known length spanning ``pixels`` → mm/px on that plane."""
    if pixels <= 0:
        raise ValueError("pixels must be positive")
    s = math.hypot(length_sigma_ln, math.log1p(pixel_sigma / pixels))
    return Scale(math.log(length_mm / pixels), s, source_class, basis, plane)


def scale_from_exif(width_px: int, height_px: int, focal_35mm: float,
                    focus_distance_m: float, *,
                    sigma_ln: float | None = None, bias_ln: float = 0.0) -> Scale:
    """Camera-only prior at the focus plane.

    35 mm-equivalent focal length is defined on the full-frame diagonal, so
    mm/px at distance d is ``d · 43.27 / (f35 · diag_px)`` whatever the aspect.

    ``sigma_ln`` and ``bias_ln`` are dynamic: pass ``exif_params(lens, params)``
    to use what the field has learned for this lens (a positive bias means EXIF
    reads long, so it is subtracted). Without them, the static prior applies.
    """
    if focal_35mm <= 0 or focus_distance_m <= 0:
        raise ValueError("focal_35mm and focus_distance_m must be positive")
    diag_px = math.hypot(width_px, height_px)
    mm_per_px = focus_distance_m * 1000.0 * FULL_FRAME_DIAG_MM / (focal_35mm * diag_px)
    s = EXIF_SIGMA_LN if sigma_ln is None else sigma_ln
    return Scale(math.log(mm_per_px) - bias_ln, s, "projected",
                 "exif" if bias_ln == 0.0 else "exif:debiased", "focus")


def fuse(scales: Iterable[Scale]) -> dict[str, Any]:
    """Inverse-variance fusion in log space with a consistency check.

    Returns the fused ``Scale`` (source class of its strongest member), the
    reduced chi-square, and members more than ``OUTLIER_Z`` from the fused
    value excluding themselves (leave-one-out, so a lone EXIF prior that
    disagrees with a body standard is named, not averaged in silently).
    """
    items = list(scales)
    if not items:
        raise ValueError("nothing to fuse")
    planes = {s.plane for s in items if s.plane != "focus"}
    w = [1.0 / s.sigma_ln ** 2 for s in items]
    mu = sum(wi * s.ln_mm_per_px for wi, s in zip(w, items)) / sum(w)
    sig = 1.0 / math.sqrt(sum(w))
    chi2 = sum(wi * (s.ln_mm_per_px - mu) ** 2 for wi, s in zip(w, items))
    dof = max(len(items) - 1, 1)
    outliers = []
    for i, s in enumerate(items):
        rest = [(wj, t) for j, (wj, t) in enumerate(zip(w, items)) if j != i]
        if not rest:
            continue
        w_rest = sum(wj for wj, _ in rest)
        mu_rest = sum(wj * t.ln_mm_per_px for wj, t in rest) / w_rest
        z = (s.ln_mm_per_px - mu_rest) / math.sqrt(s.sigma_ln ** 2 + 1.0 / w_rest)
        if abs(z) > OUTLIER_Z:
            outliers.append({"basis": s.basis, "z": round(z, 3)})
    strongest = items[max(range(len(items)), key=lambda k: w[k])]
    fused = Scale(mu, sig, strongest.source_class, "fused:" + "+".join(
        sorted({s.basis for s in items})), next(iter(planes)) if len(planes) == 1 else "default")
    return {"scale": fused, "chi2_red": round(chi2 / dof, 4), "n": len(items),
            "outliers": outliers}


def measure(scale: Scale, pixels: float, pixel_sigma: float = 1.0, *,
            basis: str = "") -> Length:
    """Pixels on the scale's plane → mm."""
    if pixels <= 0:
        raise ValueError("pixels must be positive")
    s = math.hypot(scale.sigma_ln, math.log1p(pixel_sigma / pixels))
    return Length(scale.mm_per_px * pixels, s, scale.source_class, basis or scale.basis)


def relational(a_pixels: float, b_pixels: float, sigma_px: float = 1.0) -> dict[str, Any]:
    """Unit-free ratio a/b from one photo/plane — kept until either side gets a size."""
    r = a_pixels / b_pixels
    s = math.hypot(math.log1p(sigma_px / a_pixels), math.log1p(sigma_px / b_pixels))
    return {"schema": SCHEMA, "kind": "ratio", "ratio": r, "sigma_ln": s}


def tape_span(ticks: Sequence[tuple[float, float]], a_px: float, b_px: float,
              edge_sigma_px: float = 2.0) -> dict[str, float]:
    """Read a length off a tape/ruler in the photo.

    ``ticks`` are ``(pixel_position_along_tape, graduation_value)`` pairs for
    the major marks. A quadratic (or lower, with fewer ticks) maps pixels to
    graduations, which absorbs perspective tilt and lens distortion along the
    tape. Returns the span between ``a_px`` and ``b_px`` in tape units with a
    sigma from edge placement and fit residual.
    """
    if len(ticks) < 2:
        raise ValueError("need ≥ 2 ticks")
    deg = min(2, len(ticks) - 1)
    rows = [[x ** k for k in range(deg + 1)] for x, _ in ticks]
    coef = _solve(rows, [g for _, g in ticks])

    def u(x: float) -> float:
        return sum(c * x ** k for k, c in enumerate(coef))

    def du(x: float) -> float:
        return sum(k * c * x ** (k - 1) for k, c in enumerate(coef) if k)

    resid = [u(x) - g for x, g in ticks]
    dof = max(len(ticks) - deg - 1, 1)
    fit_rms = math.sqrt(sum(r * r for r in resid) / dof) if len(ticks) > deg + 1 else 0.0
    span = abs(u(b_px) - u(a_px))
    edge = math.hypot(du(a_px), du(b_px)) * edge_sigma_px
    return {"span": span, "sigma": math.hypot(edge, fit_rms * math.sqrt(2.0)),
            "units_per_px_mid": abs(du((a_px + b_px) / 2.0)), "fit_rms": fit_rms}


# ─────────────────────────────────────────────── grid pitch (shared skill)
def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Least squares via normal equations + Gaussian elimination (small n)."""
    n = len(a[0])
    ata = [[sum(r[i] * r[j] for r in a) for j in range(n)] for i in range(n)]
    atb = [sum(r[i] * bi for r, bi in zip(a, b)) for i in range(n)]
    m = [row[:] + [v] for row, v in zip(ata, atb)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            raise ValueError("degenerate grid points")
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def grid_scale(points_px: Sequence[tuple[float, float]],
               indices: Sequence[tuple[int, int]], pitch: float, *,
               at: tuple[float, float] | None = None, unit: str = "mm",
               residual_floor_px: float = 0.5, k: float | None = None) -> dict[str, Any]:
    """Scale from a regular grid seen in perspective (pegboard, upright
    punches, wire deck, floor tiles — or game tiles).

    ``points_px`` are image positions of grid nodes, ``indices`` their integer
    (column, row) on the grid, ``pitch`` the node spacing. Fits the plane →
    image homography (≥ 4 points, 6+ recommended) and returns the local
    scale at ``at`` (default: centroid) as the geometric mean of the two
    principal stretches — the same quantity a span measurement gives.

    ``unit="tile"`` is allowed: then the result is tiles per pixel and the
    scale's source class is ``simulated``. This is the skill Perceptopoly
    trains with exact tile truth and StockOpoly applies with physical pitch.
    """
    if len(points_px) != len(indices) or len(points_px) < 4:
        raise ValueError("need ≥ 4 matched points")
    rows, rhs = [], []
    for (u, v), (i, j) in zip(points_px, indices):
        x, y = i * pitch, j * pitch
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); rhs.append(u)
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y]); rhs.append(v)
    h = _solve(rows, rhs) + [1.0]

    def proj(x: float, y: float) -> tuple[float, float]:
        w = h[6] * x + h[7] * y + 1.0
        return (h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w

    res = [math.hypot(pu - u, pv - v) for (u, v), (pu, pv) in
           ((p, proj(i * pitch, j * pitch)) for p, (i, j) in zip(points_px, indices))]
    rms = math.sqrt(sum(r * r for r in res) / len(res))
    # invert homography numerically at the query point: find plane (x, y)
    cu, cv = at if at is not None else (
        sum(p[0] for p in points_px) / len(points_px),
        sum(p[1] for p in points_px) / len(points_px))
    x, y = 0.0, 0.0
    for _ in range(30):  # Newton on proj(x,y) = (cu,cv)
        pu, pv = proj(x, y)
        e = 1e-6 * max(pitch, 1.0)
        du_dx = (proj(x + e, y)[0] - pu) / e; dv_dx = (proj(x + e, y)[1] - pv) / e
        du_dy = (proj(x, y + e)[0] - pu) / e; dv_dy = (proj(x, y + e)[1] - pv) / e
        det = du_dx * dv_dy - du_dy * dv_dx
        if abs(det) < 1e-18:
            break
        ru, rv = cu - pu, cv - pv
        x += (dv_dy * ru - du_dy * rv) / det
        y += (-dv_dx * ru + du_dx * rv) / det
        if math.hypot(ru, rv) < 1e-9:
            break
    e = 1e-4 * max(pitch, 1.0)
    pu, pv = proj(x, y)
    j11 = (proj(x + e, y)[0] - pu) / e; j21 = (proj(x + e, y)[1] - pv) / e
    j12 = (proj(x, y + e)[0] - pu) / e; j22 = (proj(x, y + e)[1] - pv) / e
    px_per_unit = math.sqrt(abs(j11 * j22 - j12 * j21))  # √det = geo-mean stretch
    unit_per_px = 1.0 / px_per_unit
    # Uncertainty, calibrated on synthetic perspective grids with exact truth
    # (Perceptopoly metrology_skill): sd(ln scale) ≈ 3.0 · σ_node / (span · √n)
    # for 6–80 nodes. σ_node is estimated from the fit residual, de-biased for
    # the 8 homography parameters, and floored for noise-free/synthetic input.
    n = len(points_px)
    span_px = max(math.hypot(a[0] - b[0], a[1] - b[1])
                  for a in points_px for b in points_px)
    # rms is a 2-D residual: per-axis σ = rms/√2, then de-bias for 8 params
    node_sigma = rms / math.sqrt(2.0) * (math.sqrt(2 * n / (2 * n - 8)) if n > 4 else 1.0)
    kk = GRID_SIGMA_K if k is None else k  # dynamic: params["grid_sigma_k"]["value"]
    node_used = max(node_sigma, residual_floor_px)
    sig = math.log1p(kk * node_used / (span_px * math.sqrt(n)))
    # what a refit of k needs (see fit_grid_sigma_k)
    fit = {"node_sigma_px": node_used, "span_px": span_px, "n": n}
    if unit == "tile":
        return {"tiles_per_px": unit_per_px, "sigma_ln": sig, "rms_px": rms,
                "source_class": "simulated", "fit": fit}
    pitch_mm = to_mm(pitch, unit)
    mm_per_px = unit_per_px * (pitch_mm / pitch)
    return {"scale": Scale(math.log(mm_per_px), sig, "recorded", f"grid:{pitch_mm}mm"),
            "rms_px": rms, "fit": fit}


# ─────────────────────────────────────────────── linking people (transfer)
def transfer_offsets(links: Sequence[dict[str, Any]], anchor: str | None = None) -> dict[str, Any]:
    """Relative calibration of people's standards through shared objects.

    Each link is ``{"object": key, "owner": who, "ln_mm": ln(length measured
    via that owner's standard), "sigma_ln": ..}``. Every owner has an unknown
    log offset b_o (their standard's bias); every object an unknown true ln
    length. Solved by weighted least squares with the anchor owner's offset
    fixed at 0 (default: the owner with most links). Returns offsets as
    multiplicative factors: a factor of 1.01 means that owner's standard reads
    1 % long relative to the anchor.
    """
    owners = sorted({l["owner"] for l in links})
    objects = sorted({l["object"] for l in links})
    if len(owners) < 2:
        return {"anchor": owners[0] if owners else None, "factors": {o: 1.0 for o in owners},
                "linked": False}
    if anchor is None:
        anchor = max(owners, key=lambda o: sum(1 for l in links if l["owner"] == o))
    free = [o for o in owners if o != anchor]
    cols = {f"b:{o}": k for k, o in enumerate(free)}
    cols.update({f"x:{ob}": len(free) + k for k, ob in enumerate(objects)})
    rows, rhs = [], []
    for l in links:
        w = 1.0 / l["sigma_ln"]
        r = [0.0] * len(cols)
        r[cols[f"x:{l['object']}"]] = w
        if l["owner"] != anchor:
            r[cols[f"b:{l['owner']}"]] = w
        rows.append(r); rhs.append(w * l["ln_mm"])
    try:
        sol = _solve(rows, rhs)
    except ValueError:
        return {"anchor": anchor, "factors": {o: 1.0 for o in owners}, "linked": False}
    factors = {anchor: 1.0}
    factors.update({o: round(math.exp(sol[cols[f"b:{o}"]]), 6) for o in free})
    shared = [ob for ob in objects if len({l["owner"] for l in links if l["object"] == ob}) > 1]
    return {"anchor": anchor, "factors": factors, "linked": bool(shared),
            "shared_objects": shared}


# ───────────────────────────────────────── resolution floors (the Planck gate)
# Physical floor of any length: the Planck length, CODATA 2018
# (1.616255(18)e-35 m, relative standard uncertainty 1.1e-5). Nothing a camera
# measures comes near it; it is the bound every other floor sits above, and the
# clamp that keeps a floor from ever reporting zero.
PLANCK_LENGTH_M = 1.616255e-35
PLANCK_LENGTH_MM = PLANCK_LENGTH_M * 1000.0
_PHI_SQ_INTEGRAL = 1.0 / (2.0 * math.sqrt(math.pi))  # ∫φ², as in Bakugo/QUIPU


def cramer_rao_edge_px(contrast: float, noise_sigma: float, psf_sigma_px: float,
                       rows: int = 1, pixel_pitch_px: float = 1.0) -> float:
    """Lower bound on locating one edge, in px, for a Gaussian channel.

    Identical to Bakugo ``information.cramer_rao_edge_px``: Fisher information
    (C/σ_n)² · ∫φ² / (σ_psf · pitch) per row, times independent rows.
    """
    if contrast <= 0 or noise_sigma <= 0 or psf_sigma_px <= 0:
        return math.inf
    info = (contrast ** 2 / noise_sigma ** 2) * _PHI_SQ_INTEGRAL / (
        psf_sigma_px * pixel_pitch_px) * max(1, rows)
    return 1.0 / math.sqrt(info)


def quantum_floor_edge_px(contrast: float, background_level: float, psf_sigma_px: float,
                          rows: int = 1, pixel_pitch_px: float = 1.0, *,
                          full_well_e: float = 6000.0, bit_depth: int = 8,
                          iso: float = 100.0, base_iso: float = 100.0) -> float:
    """Standard Quantum Limit on locating one edge, in px (photon shot noise).

    Identical to Bakugo ``information.quantum_floor_px`` with its default
    phone ``SensorModel``. Light is quantised, so arrivals are Poisson and
    information is Σ(dμ/dθ)²/μ in photoelectrons. The sensor's interstitial
    pixel space (fill-factor dead area between photodiodes, behind the
    microlenses) is ordinary optical loss and is already inside the grey-level
    → electron conversion; it is not a separate term.
    """
    if contrast <= 0 or psf_sigma_px <= 0:
        return math.inf
    e_per_level = full_well_e / float(2 ** bit_depth - 1) / max(1.0, iso / base_iso)
    c_e = max(0.0, contrast) * e_per_level
    mu_e = max(0.0, max(background_level, contrast * 0.5)) * e_per_level
    if mu_e <= 0:
        return math.inf
    info = (c_e ** 2 / mu_e) * _PHI_SQ_INTEGRAL / (psf_sigma_px * pixel_pitch_px) * max(1, rows)
    return 1.0 / math.sqrt(info)


def planck_floor(scale: Scale, edge_sigma_px: float, *, edges: int = 2) -> Length:
    """Smallest length resolvable on this plane of this photo, in mm.

    A span is bounded by ``edges`` edge locations, each known to
    ``edge_sigma_px`` (use the larger of the Cramér–Rao and quantum floors, or
    the measured edge scatter). The floor is mm/px · σ_edge · √edges, never
    below the Planck length. QUIPU's Planck gate is the same rule in time: a
    change smaller than the resolution floor is not a displacement and is not
    written.
    """
    mm = max(scale.mm_per_px * edge_sigma_px * math.sqrt(edges), PLANCK_LENGTH_MM)
    return Length(mm, scale.sigma_ln, scale.source_class, f"planck_floor:{scale.basis}")


def planck_gate(delta_mm: float, floor_mm: float, k: float = 1.0) -> bool:
    """True when a difference is a real displacement (|Δ| ≥ k · floor)."""
    return abs(delta_mm) >= k * max(floor_mm, PLANCK_LENGTH_MM)


# ─────────────────────────────────────────────────────── interstitial space
def interstitial_gap(edge_a_px: float, edge_b_px: float, edge_sigma_px: float,
                     scale: Scale | None = None) -> dict[str, Any]:
    """The space between two things on one plane: a card's border between
    cut edge and print frame, a carton's clearance to a beam, the gap between
    stacked parts. Returns pixels, mm (when a scale is known) and whether the
    gap is resolvable at all (Planck gate against its own floor)."""
    gap_px = abs(edge_b_px - edge_a_px)
    sigma_px = edge_sigma_px * math.sqrt(2.0)
    out: dict[str, Any] = {"gap_px": gap_px, "sigma_px": sigma_px,
                           "resolvable": gap_px >= sigma_px}
    if scale is not None:
        mm = gap_px * scale.mm_per_px
        sig_mm = math.hypot(sigma_px * scale.mm_per_px, mm * scale.sigma_ln)
        out.update({"gap_mm": mm, "sigma_mm": sig_mm})
    return out


INTERSTITIAL_ARC = {"perception": "perceptopoly", "vision": "loadopoly-ocr", "touch": "bakugo"}


def interstitial_arc(residuals: dict[str, Sequence[float]]) -> dict[str, Any]:
    """QUIPU's interstitial measure applied to metrology residuals.

    ``residuals`` maps ``perception`` (Perceptopoly), ``vision`` (GeoGraph
    OCR) and ``touch`` (Bakugo) to each output's ln-scale residuals
    (estimate − consensus) on the same photos, in the same order. With
    C_ab = Σ a·b / √(Σa² Σb²):

        mediated     = C_pv · C_vt
        interstitial = |C_pt − C_pv·C_vt| / (1 + |C_pt|)  ∈ [0, 1)

    Zero when Perceptopoly's errors coincide with Bakugo's only through OCR;
    larger when they coincide directly. Same form as
    ``quipu.qpsi.interstitial`` (credited there to
    ``ueqgm_engine.interstitial_entanglement_score``), real-valued here. Like
    QUIPU's, it is recorded, not routed: nothing adjusts a scale from it.
    """
    p, v, t = (list(residuals[k]) for k in ("perception", "vision", "touch"))
    if not (len(p) == len(v) == len(t)) or len(p) < 2:
        raise ValueError("need ≥ 2 shared photos with residuals from all three outputs")

    def coh(a: list[float], b: list[float]) -> float:
        den = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
        return sum(x * y for x, y in zip(a, b)) / den if den > 0 else 0.0

    c_pv, c_vt, c_pt = coh(p, v), coh(v, t), coh(p, t)
    mediated = c_pv * c_vt
    return {"schema": SCHEMA, "kind": "interstitial_arc", "n": len(p),
            "c_pv": c_pv, "c_vt": c_vt, "c_pt": c_pt, "mediated": mediated,
            "interstitial": abs(c_pt - mediated) / (1.0 + abs(c_pt)),
            "routed": False}


# ───────────────────────────────────────────────────── grading components
# Bakugo's four aspects. Metrology owns the measurements; Bakugo owns the
# mapping to grades (grader tables disagree, and that ambiguity is not
# something better measurement can remove).
ASPECTS = ("centering", "corners", "edges", "surface")


def _ratio(a_px: float, b_px: float, border_sigma_px: float) -> tuple[float, float]:
    """r = 100·a/(a+b) and its sigma in percentage points (Bakugo partials)."""
    tot = a_px + b_px
    if tot <= 0:
        raise ValueError("borders must have positive total width")
    r = 100.0 * a_px / tot
    s = 100.0 * math.hypot(b_px * border_sigma_px, a_px * border_sigma_px) / tot ** 2
    return r, s


def centering(left_px: float, right_px: float, top_px: float, bottom_px: float,
              edge_sigma_px: float, *, scale: Scale | None = None,
              face: str = "front") -> dict[str, Any]:
    """Centering from the four interstitial borders (cut edge to print frame).

    Ratios are computed in pixels, so the photo's scale cancels (Bakugo's
    scale-invariance note); border widths in mm are added only when a scale is
    given. Each border is bounded by two edges → σ_border = σ_edge·√2. Each
    axis is reported as the larger side (55 for 55/45), with a Planck-gate flag
    saying whether its off-centre is resolvable at all. Field names match
    ``bakugo_scans``.
    """
    sb = edge_sigma_px * math.sqrt(2.0)
    h, hs = _ratio(left_px, right_px, sb)
    v, vs = _ratio(top_px, bottom_px, sb)
    h, v = max(h, 100.0 - h), max(v, 100.0 - v)
    worst_axis = "horizontal" if h >= v else "vertical"
    out: dict[str, Any] = {
        "schema": SCHEMA, "kind": "grading_component", "aspect": "centering", "face": face,
        "h_ratio_pct": h, "h_ratio_sigma": hs, "v_ratio_pct": v, "v_ratio_sigma": vs,
        "worst_axis": worst_axis,
        "worst_ratio_pct": h if worst_axis == "horizontal" else v,
        "worst_ratio_sigma": hs if worst_axis == "horizontal" else vs,
        "h_resolvable": planck_gate(left_px - right_px, sb * math.sqrt(2.0)),
        "v_resolvable": planck_gate(top_px - bottom_px, sb * math.sqrt(2.0)),
    }
    if scale is not None:
        mpp = scale.mm_per_px
        out.update({"left_mm": left_px * mpp, "right_mm": right_px * mpp,
                    "top_mm": top_px * mpp, "bottom_mm": bottom_px * mpp,
                    "px_per_mm": 1.0 / mpp})
    return out


def grading_component(aspect: str, name: str, pixels: float, edge_sigma_px: float,
                      scale: Scale, *, face: str = "front", area: bool = False) -> dict[str, Any]:
    """A measured defect for corners / edges / surface: corner wear length,
    edge chip length, surface scratch length — or, with ``area=True``, a
    surface defect area (``pixels`` is then a pixel count, mm² via scale²).
    Carries its Planck floor and whether the defect clears it."""
    if aspect not in ASPECTS or aspect == "centering":
        raise ValueError("use centering() for centering; aspect must be corners, edges or surface")
    floor = planck_floor(scale, edge_sigma_px)
    if area:
        mm2 = pixels * scale.mm_per_px ** 2
        # boundary uncertainty: perimeter of the equal-area disc × σ_edge
        perim_px = 2.0 * math.sqrt(math.pi * pixels)
        sig = mm2 * math.hypot(2.0 * scale.sigma_ln, perim_px * edge_sigma_px / pixels)
        value, unit, resolvable = mm2, "mm2", math.sqrt(mm2) >= floor.mm
    else:
        L = measure(scale, pixels, edge_sigma_px * math.sqrt(2.0))
        value, unit, sig, resolvable = L.mm, "mm", L.sigma_mm, L.mm >= floor.mm
    return {"schema": SCHEMA, "kind": "grading_component", "aspect": aspect, "name": name,
            "face": face, "value": value, "sigma": sig, "unit": unit,
            "planck_floor_mm": floor.mm, "resolvable": resolvable,
            "source_class": scale.source_class}


# ───────────────────────────────────────────── dynamic parameters (learned)
# Every constant above that is not physics starts as a prior and is replaced
# by what the data say as data accumulate. A prior is worth ``n0`` pseudo
# observations, so one photo cannot swing a parameter and a hundred can.
PRIOR_WEIGHT_N = 3.0
PARALLAX_PRIOR_MM = (6.0, 4.0)        # hand under expanded-metal mesh, before data


def shrink(prior: float, observed: float, n: float, n0: float = PRIOR_WEIGHT_N) -> float:
    """Precision-weighted blend of a prior and an observed value (n samples)."""
    return (n0 * prior + n * observed) / (n0 + n) if n > 0 else prior


def default_params() -> dict[str, Any]:
    return {"schema": SCHEMA, "kind": "calibration_params", "updated_at": None,
            "exif": {},  # lens → {"bias_ln", "sigma_ln", "n"}
            "grid_sigma_k": {"value": GRID_SIGMA_K, "n": 0, "source": "prior"}}


def params_from_stats(error_stats: Iterable[dict[str, Any]],
                      grid_fit: dict[str, Any] | None = None,
                      base: dict[str, Any] | None = None,
                      updated_at: str | None = None) -> dict[str, Any]:
    """Fold unit-free field statistics into the calibration parameters.

    EXIF per lens: bias = shrunk mean log error; sigma = shrunk RMS of the
    residual after removing that bias (prior: 0 bias, ``EXIF_SIGMA_LN``).
    Grid: k from ``fit_grid_sigma_k`` (prior ``GRID_SIGMA_K``).
    """
    p = json.loads(json.dumps(base)) if base else default_params()
    for st in error_stats:
        if st.get("kind") != "error_stat" or st.get("estimator") != "exif" or not st.get("n"):
            continue
        n = float(st["n"])
        mean = float(st["mean_log_err"])
        sd = st.get("sd_log_err")
        bias = shrink(0.0, mean, n)
        var_obs = float(sd) ** 2 if sd is not None else EXIF_SIGMA_LN ** 2
        sigma = math.sqrt(shrink(EXIF_SIGMA_LN ** 2, var_obs, max(n - 1.0, 0.0)))
        p["exif"][str(st.get("lens", ""))] = {"bias_ln": bias, "sigma_ln": sigma, "n": int(n)}
    if grid_fit and grid_fit.get("n"):
        n = float(grid_fit["n"])
        p["grid_sigma_k"] = {"value": math.sqrt(shrink(GRID_SIGMA_K ** 2, grid_fit["k"] ** 2, n)),
                             "n": int(n), "source": grid_fit.get("source", "fit")}
    p["updated_at"] = updated_at
    return p


def exif_params(lens: float | str | None, params: dict[str, Any] | None) -> dict[str, float]:
    """``sigma_ln``/``bias_ln`` for ``scale_from_exif`` from learned params."""
    key = "" if lens in (None, "") else str(int(float(lens)))
    row = (params or {}).get("exif", {}).get(key)
    if not row:
        return {"sigma_ln": EXIF_SIGMA_LN, "bias_ln": 0.0}
    return {"sigma_ln": float(row["sigma_ln"]), "bias_ln": float(row["bias_ln"])}


def grid_k(params: dict[str, Any] | None) -> float:
    return float(((params or {}).get("grid_sigma_k") or {}).get("value", GRID_SIGMA_K))


def fit_grid_sigma_k(samples: Iterable[dict[str, Any]], source: str = "fit") -> dict[str, Any]:
    """Refit the grid uncertainty constant from estimates with known truth.

    Each sample: ``log_err`` (ln estimate/truth) and the estimator's own
    ``fit`` dict (node_sigma_px, span_px, n). k = RMS of log_err measured in
    units of node_sigma/(span·√n). Truth comes from synthetic grids
    (Perceptopoly) or from grids with a tape across them (StockOpoly).
    """
    z = [s["log_err"] / (s["fit"]["node_sigma_px"] / (s["fit"]["span_px"] * math.sqrt(s["fit"]["n"])))
         for s in samples]
    if not z:
        return {"k": GRID_SIGMA_K, "n": 0, "source": "prior"}
    return {"k": math.sqrt(sum(x * x for x in z) / len(z)), "n": len(z), "source": source}


def edge_sigma_from_profiles(profiles: Sequence[Sequence[float]], *,
                             full_well_e: float = 6000.0, bit_depth: int = 8,
                             iso: float = 100.0) -> dict[str, Any]:
    """Measure how well one edge can be located, from the image itself.

    ``profiles`` are intensity samples taken across the edge (one per row
    along it, same length, edge near the middle). From them: contrast (plateau
    step), noise (robust scatter of rows about the mean profile), blur (10–90 %
    rise / 2.563) and the number of rows. The edge sigma is the larger of the
    Cramér–Rao and quantum floors — the dynamic input to ``planck_floor``.
    """
    rows = len(profiles)
    if rows < 2:
        raise ValueError("need ≥ 2 profiles across the edge")
    m = len(profiles[0])
    if m < 8 or any(len(p) != m for p in profiles):
        raise ValueError("profiles must share a length ≥ 8")
    mean = [sum(p[i] for p in profiles) / rows for i in range(m)]
    third = max(m // 3, 2)
    lo_level = sum(mean[:third]) / third
    hi_level = sum(mean[-third:]) / third
    contrast = abs(hi_level - lo_level)
    resid = sorted(abs(p[i] - mean[i]) for p in profiles for i in range(m))
    noise = 1.4826 * resid[len(resid) // 2] * math.sqrt(rows / max(rows - 1, 1))
    if contrast <= 0:
        return {"edge_sigma_px": math.inf, "contrast": 0.0, "noise": noise,
                "psf_sigma_px": None, "rows": rows, "limited_by": "no edge"}
    rising = hi_level > lo_level
    norm = [((v - lo_level) / (hi_level - lo_level)) for v in mean]
    if not rising:
        norm = [1.0 - x for x in norm]

    def cross(level: float) -> float:
        for i in range(m - 1):
            a, b = norm[i], norm[i + 1]
            if (a - level) * (b - level) <= 0 and a != b:
                return i + (level - a) / (b - a)
        return float(m)

    psf = max((cross(0.9) - cross(0.1)) / 2.563, 0.3)
    cr = cramer_rao_edge_px(contrast, max(noise, 1e-6), psf, rows)
    q = quantum_floor_edge_px(contrast, min(lo_level, hi_level), psf, rows,
                              full_well_e=full_well_e, bit_depth=bit_depth, iso=iso)
    return {"edge_sigma_px": max(cr, q), "cramer_rao_px": cr, "quantum_px": q,
            "contrast": contrast, "noise": noise, "psf_sigma_px": psf, "rows": rows,
            "limited_by": "channel noise" if cr >= q else "photon shot noise"}


def solve_parallax(frames: Sequence[dict[str, Any]],
                   prior: tuple[float, float] = PARALLAX_PRIOR_MM,
                   max_depth_mm: float = 60.0, step_mm: float = 0.1) -> dict[str, Any]:
    """Solve how far an object sits behind the reference plane, from frames
    taken at different focus distances.

    Each frame: ``raw_mm`` (read off the reference plane), ``sigma_mm``,
    ``z_mm`` (focus distance) and optional ``group`` (which object: frames of
    a palm and of a finger behind the same mesh share the depth d but each
    group has its own true size T_g). Model raw_i = T_g·z_i/(z_i + d). Scatter beyond
    the propagated sigmas (pose, framing) is estimated and added to every
    frame, so frames that disagree for other reasons cannot fake a depth; the
    prior then carries the answer until frames at clearly different distances
    agree. Returns the posterior mean and sd of d, and how informative the
    frames were (prior sd / posterior sd).
    """
    mu0, s0 = prior
    groups = [str(f.get("group", "_")) for f in frames]
    names = sorted(set(groups))
    n = len(frames)
    dof = n - len(names) - 1   # one true size per group + the shared depth
    # With no spare degree of freedom any depth fits exactly: the frames say
    # nothing about it, and the prior stands.
    if dof < 1 or len({round(float(f["z_mm"]), 1) for f in frames}) < 2:
        return {"depth_mm": mu0, "sigma_mm": s0, "informative": 1.0, "source": "prior",
                "scatter_mm": None,
                "note": "needs more frames at distinct focus distances than objects measured + 1"}
    raws = [float(f["raw_mm"]) for f in frames]
    sig = [float(f["sigma_mm"]) for f in frames]
    zs = [float(f["z_mm"]) for f in frames]
    grid = []
    d = 0.0
    while d <= max_depth_mm + 1e-9:
        grid.append(d)
        d += step_mm

    def corrected(dd: float) -> list[float]:
        return [r * (z + dd) / z for r, z in zip(raws, zs)]

    def group_chi2(vals: list[float], w: list[float]) -> float:
        total = 0.0
        for g in names:
            ix = [i for i, gi in enumerate(groups) if gi == g]
            t = sum(w[i] * vals[i] for i in ix) / sum(w[i] for i in ix)
            total += sum(w[i] * (vals[i] - t) ** 2 for i in ix)
        return total

    def posterior(tau2: float) -> tuple[float, float]:
        w = [1.0 / (s_ * s_ + tau2) for s_ in sig]
        logp = [-0.5 * group_chi2(corrected(dd), w) - 0.5 * ((dd - mu0) / s0) ** 2 for dd in grid]
        top = max(logp)
        pw = [math.exp(lp - top) for lp in logp]
        z = sum(pw)
        mean = sum(g * p for g, p in zip(grid, pw)) / z
        var = sum((g - mean) ** 2 * p for g, p in zip(grid, pw)) / z
        return mean, math.sqrt(max(var, 1e-12))

    # scatter beyond the stated sigmas, estimated from residuals at the current
    # fit, so the parallax signal itself is not counted as scatter
    tau2 = 0.0
    mean, post_sd = posterior(tau2)
    for _ in range(5):
        vals = corrected(mean)
        unit_w = [1.0] * n
        excess = group_chi2(vals, unit_w) / dof - sum(s_ * s_ for s_ in sig) / n
        new_tau2 = max(0.0, excess)
        if abs(new_tau2 - tau2) < 1e-6:
            break
        tau2 = new_tau2
        mean, post_sd = posterior(tau2)
    return {"depth_mm": mean, "sigma_mm": post_sd, "informative": s0 / post_sd,
            "source": "posterior", "scatter_mm": math.sqrt(tau2), "frames": n,
            "groups": len(names)}


def load_params(path: str | None = None) -> dict[str, Any]:
    """Calibration params from ``path`` or ``$HOLON_METROLOGY_PARAMS``; priors if none."""
    import os
    p = path or os.environ.get("HOLON_METROLOGY_PARAMS", "")
    if p and os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as fh:
                rec = json.load(fh)
            if rec.get("kind") == "calibration_params":
                return rec
        except (OSError, ValueError):
            pass
    return default_params()


# ───────────────────────────────────────────── what may leave a private store
def error_stats(residuals: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Unit-free error statistics grouped by (estimator, lens).

    Input rows: ``{"estimator": "exif"|"body"|"grid"|.., "lens": "24"|"12"|..,
    "log_err": ln(estimate / truth)}``. Output carries no photo, person,
    location or part identifiers — this is what Perceptopoly and QUIPU get.
    """
    groups: dict[tuple[str, str], list[float]] = {}
    for r in residuals:
        groups.setdefault((str(r["estimator"]), str(r.get("lens", ""))), []).append(
            float(r["log_err"]))
    out = []
    for (est, lens), xs in sorted(groups.items()):
        n = len(xs)
        mean = sum(xs) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else None
        out.append({"schema": SCHEMA, "kind": "error_stat", "estimator": est, "lens": lens,
                    "n": n, "mean_log_err": round(mean, 6),
                    "sd_log_err": round(sd, 6) if sd is not None else None})
    return out


def public_view(record: dict[str, Any]) -> dict[str, Any] | None:
    """Strip a record for sharing outside its owner's store.

    Personal standards never leave; ``error_stat`` and ``ratio`` records pass
    unchanged; scales/lengths drop their ``basis`` (it can name a person).
    """
    kind = record.get("kind")
    if kind == "local_standard" or record.get("privacy") == "personal":
        return None
    if kind in ("error_stat", "ratio"):
        return dict(record)
    out = {k: v for k, v in record.items() if k not in ("basis", "owner", "evidence")}
    return out
