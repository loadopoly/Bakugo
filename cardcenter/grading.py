"""Turn a measured centering ratio into a grade *band*.

Two independent things stop us from naming a single grade, and the tool keeps
them separate because the user's response to each is different:

  MEASUREMENT uncertainty -- the confidence interval on our own ratio. Fixed by
  better capture: more light, less tilt, a tripod, a higher-resolution sensor.

  STANDARDS ambiguity -- reputable sources disagree about where the thresholds
  actually sit, and graders reserve explicit discretion. No amount of better
  photography fixes this. It is irreducible from outside the grading room.

Reporting one number would hide both. Reporting a band without saying which one
is binding would leave the user unable to act. So we report the band and name
the dominant cause.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import numpy as np

from .types import Measured

if TYPE_CHECKING:
    from .learning import GradeOutcomeModel

Face = Literal["front", "back"]
_DATA = Path(__file__).parent / "data" / "standards.json"


@lru_cache(maxsize=1)
def load_standards() -> dict:
    with open(_DATA, "r", encoding="utf-8") as fh:
        return json.load(fh)


def available_graders() -> list[str]:
    return list(load_standards()["graders"].keys())


@dataclass(frozen=True)
class GradeBand:
    grader: str
    face: Face
    best: str
    worst: str
    ratio: Measured
    measurement_span: int
    standards_span: int
    limited_by: str
    grader_confidence: str
    notes: str

    @property
    def is_single(self) -> bool:
        return self.best == self.worst

    def describe(self) -> str:
        lo, hi = self.ratio.interval()
        head = (
            f"{self.grader} {self.face} centering ceiling: "
            + (self.best if self.is_single else f"{self.worst}-{self.best}")
        )
        body = (
            f"  measured {self.ratio.value:.1f}/{100 - self.ratio.value:.1f} "
            f"(95% CI {lo:.1f}-{hi:.1f})\n"
            f"  band limited by: {self.limited_by}\n"
            f"  table confidence: {self.grader_confidence}"
        )
        return head + "\n" + body


def _threshold_key(face: Face, variant: str) -> str:
    return f"{face}_{variant}"


def _tier_index_for(tiers: list[dict], ratio: float, face: Face, variant: str) -> int:
    """Index of the best tier whose threshold admits ``ratio``. Higher index = worse."""
    key = _threshold_key(face, variant)
    for i, tier in enumerate(tiers):
        if ratio <= tier[key] + 1e-9:
            return i
    return len(tiers) - 1


def grade_band(
    ratio: Measured,
    grader: str = "PSA",
    face: Face = "front",
    k_sigma: float = 1.96,
) -> GradeBand:
    """Map a worst-axis ratio to a band of plausible centering grades."""
    std = load_standards()
    graders = std["graders"]
    if grader not in graders:
        raise KeyError(
            f"unknown grader '{grader}'. Available: {', '.join(graders)}"
        )
    g = graders[grader]
    tiers = g["tiers"]

    lo, hi = ratio.interval(k_sigma)
    lo = max(50.0, lo)
    hi = max(50.0, hi)
    centre = max(50.0, ratio.value)

    # Best case: low end of our interval, judged by the most forgiving table.
    best_idx = _tier_index_for(tiers, lo, face, "lenient")
    # Worst case: high end of our interval, judged by the strictest table.
    worst_idx = _tier_index_for(tiers, hi, face, "strict")

    # Attribution. Hold the table fixed to isolate measurement span; hold the
    # ratio fixed to isolate standards span.
    meas_spans = [
        _tier_index_for(tiers, hi, face, v) - _tier_index_for(tiers, lo, face, v)
        for v in ("strict", "lenient")
    ]
    measurement_span = max(meas_spans)
    standards_span = _tier_index_for(tiers, centre, face, "strict") - _tier_index_for(
        tiers, centre, face, "lenient"
    )

    if measurement_span == 0 and standards_span == 0:
        limited_by = "neither; the measurement and the published tables agree"
    elif measurement_span > standards_span:
        limited_by = (
            "measurement uncertainty -- a steadier, better-lit, less-tilted "
            "capture would narrow this"
        )
    elif standards_span > measurement_span:
        limited_by = (
            "standards ambiguity -- sources disagree on this threshold, and "
            "better photography will not resolve it"
        )
    else:
        limited_by = "measurement uncertainty and standards ambiguity equally"

    return GradeBand(
        grader=grader,
        face=face,
        best=tiers[best_idx]["grade"],
        worst=tiers[worst_idx]["grade"],
        ratio=ratio,
        measurement_span=int(measurement_span),
        standards_span=int(standards_span),
        limited_by=limited_by,
        grader_confidence=g.get("confidence", "unknown"),
        notes=g.get("notes", ""),
    )


def all_grade_bands(
    ratio: Measured, face: Face = "front", k_sigma: float = 1.96
) -> dict[str, GradeBand]:
    return {
        name: grade_band(ratio, name, face, k_sigma) for name in available_graders()
    }


def caveat_text(grader: str) -> str:
    g = load_standards()["graders"].get(grader, {})
    lines = [g.get("notes", "")]
    if not g.get("subgrade_published", False):
        lines.append(
            "This grader does not publish a centering sub-grade, so centering "
            "only sets a ceiling on the overall grade. Corners, edges and "
            "surface can and often will land it lower."
        )
    if g.get("confidence") == "low":
        lines.append(
            "Threshold sourcing for this grader is weak. Treat the band as "
            "indicative and verify against current published standards."
        )
    return "\n".join(x for x in lines if x)


# ---------------------------------------------------------------------------
# Whole-card grade: centering, corners, edges, surface
# ---------------------------------------------------------------------------
#
# Until 2.24 this section estimated corners, edges and surface from how cleanly
# the card's OUTLINE was detected (line-fit residual, border-detector
# confidence), starting each at 10 and docking at most 2 points. Detection
# quality is a property of the photo, not the card, and the cap meant every
# soft live frame landed on exactly 10 - 2 = 8. Those three now come from
# measurements of the card (cardcenter.condition, pooled across views by
# cardcenter.condition_evidence) or are reported as NOT ASSESSED -- never
# assumed clean.
#
# Each aspect is a probability distribution over grades. Centering's comes from
# the ratio's uncertainty and from the spread between the strict and lenient
# published thresholds (both irreducible causes, kept). Corners and edges come
# from each corner's and side's pooled measurement, its uncertainty, and the
# category boundaries in data/condition_standards.json, enumerated exactly
# over the four corners / sides. The overall grade is the weakest aspect for
# PSA, CGC and SGC and the BGS rule for BGS, enumerated over the aspects.
# ``probabilities`` therefore means what it says: the chance of each grade
# given what was measured. While an aspect has no evidence the result is a
# CEILING (``complete`` False): the grade cannot be above it, and may be below.

CONDITION_NAMES: dict[float, str] = {
    10.0: "Gem Mint",
    9.5: "Gem Mint",
    9.0: "Mint",
    8.5: "Near Mint-Mint+",
    8.0: "Near Mint-Mint",
    7.5: "Near Mint+",
    7.0: "Near Mint",
    6.5: "Excellent-Mint+",
    6.0: "Excellent-Mint",
    5.5: "Excellent+",
    5.0: "Excellent",
    4.5: "Very Good-Excellent+",
    4.0: "Very Good-Excellent",
    3.0: "Very Good",
    2.0: "Good",
    1.0: "Poor",
}

ASPECTS = ("centering", "corners", "edges", "surface")
_PRUNE = 1e-5


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm(dist: Dict[float, float]) -> Dict[float, float]:
    t = sum(dist.values())
    if t <= 0:
        return {}
    return {k: v / t for k, v in dist.items() if v / t > _PRUNE}


def _key(g: float) -> str:
    return str(int(g)) if float(g).is_integer() else str(g)


@dataclass(frozen=True)
class AspectGrade:
    """One aspect's grade distribution and how it was arrived at."""

    aspect: str
    assessed: str                       # "no", "front", "back", "both"
    distribution: Dict[float, float]    # grade -> probability
    detail: str = ""

    @property
    def is_assessed(self) -> bool:
        return self.assessed != "no" and bool(self.distribution)

    @property
    def modal(self) -> Optional[float]:
        if not self.distribution:
            return None
        return max(self.distribution.items(), key=lambda kv: (kv[1], kv[0]))[0]

    @property
    def p_modal(self) -> float:
        return float(self.distribution.get(self.modal, 0.0)) if self.distribution else 0.0

    def to_dict(self) -> dict:
        from .condition import plain

        return plain({
            "assessed": self.assessed,
            "grade": None if self.modal is None else _key(self.modal),
            "p": round(self.p_modal, 2),
            "distribution": {_key(k): round(v, 3) for k, v in sorted(self.distribution.items(), reverse=True)
                             if v >= 0.005},
            "detail": self.detail,
        })


# -- centering ---------------------------------------------------------------


def centering_distribution(ratio: Measured, grader: str = "PSA", face: Face = "front",
                           n_thresholds: int = 9) -> Dict[float, float]:
    """P(centering grade) from the ratio's uncertainty and the spread between
    the strict and lenient published thresholds (each tier's threshold taken
    as uniformly somewhere between the two)."""
    std = load_standards()
    tiers = std["graders"].get(grader, std["graders"]["PSA"])["tiers"]
    r = max(50.0, float(ratio.value))
    sig = max(float(ratio.sigma), 0.05)
    meets = []
    for t in tiers:
        a = float(t[f"{face}_strict"])
        b = float(t[f"{face}_lenient"])
        ths = np.linspace(min(a, b), max(a, b), n_thresholds)
        meets.append(float(np.mean([_phi((th - r) / sig) for th in ths])))
    # tiers run best -> worst with thresholds non-decreasing, so meeting a
    # better tier implies meeting every worse one
    dist: Dict[float, float] = {}
    prev = 0.0
    for t, m in zip(tiers, meets):
        m = max(m, prev)
        g = float(t["grade"])
        dist[g] = dist.get(g, 0.0) + (m - prev)
        prev = m
    last = float(tiers[-1]["grade"])
    dist[last] = dist.get(last, 0.0) + max(0.0, 1.0 - prev)
    return _norm(dist)


# -- corners and edges ---------------------------------------------------------


def _categories(mu: float, sigma: float, bounds) -> list:
    """P(category k) for a non-negative quantity ~ N(mu, sigma), folded at 0."""
    sigma = max(sigma, 1e-6)
    cdf = [_phi((b - mu) / sigma) for b in bounds]
    p0 = cdf[0]
    out = [p0]
    for k in range(1, len(bounds)):
        out.append(max(0.0, cdf[k] - cdf[k - 1]))
    out.append(max(0.0, 1.0 - cdf[-1]))
    t = sum(out)
    return [x / t for x in out]


def _max_categories(a: list, b: list) -> list:
    """Category distribution of the worse of two independent readings."""
    ca = np.cumsum(a)
    cb = np.cumsum(b)
    cm = ca * cb
    return list(np.diff(np.concatenate([[0.0], cm])))


def corner_grade_rule(counts: list) -> float:
    """Grade from how many corners fall in each category (sharp, slight
    fraying, fraying, minor rounding, rounding, heavy rounding)."""
    c = list(counts) + [0] * (6 - len(counts))
    if c[5]:
        return 2.0 if c[5] >= 3 else 3.0
    if c[4]:
        return 3.0 if c[4] >= 3 else 4.0
    if c[3]:
        return 5.0
    if c[2]:
        return 6.0 if c[2] >= 2 else 7.0
    return {0: 10.0, 1: 9.0, 2: 8.0}.get(c[1], 7.0)


def edge_grade_rule(counts: list) -> float:
    """Grade from how many sides fall in each wear category."""
    c = list(counts) + [0] * (6 - len(counts))
    if c[5]:
        return 3.0
    if c[4]:
        return 4.0
    if c[3]:
        return 5.0
    if c[2]:
        return 6.0 if c[2] >= 2 else 7.0
    return {0: 10.0, 1: 9.0, 2: 8.0}.get(c[1], 7.0)


def _enumerate(per_item: list, rule) -> Dict[float, float]:
    """Exact grade distribution over independent per-item category
    distributions (4 items x 6 categories = 1296 combinations)."""
    dist: Dict[float, float] = {}
    n_cat = len(per_item[0])

    def rec(i, counts, p):
        if p < _PRUNE:
            return
        if i == len(per_item):
            g = rule(counts)
            dist[g] = dist.get(g, 0.0) + p
            return
        for k in range(n_cat):
            pk = per_item[i][k]
            if pk <= 0:
                continue
            counts[k] += 1
            rec(i + 1, counts, p * pk)
            counts[k] -= 1

    rec(0, [0] * n_cat, 1.0)
    return _norm(dist)


_CORNER_ABBR = {"top_left": "TL", "top_right": "TR", "bottom_right": "BR", "bottom_left": "BL"}


def _faces_of(summary: dict) -> list:
    return [f for f in ("front", "back") if f in summary]


def corner_aspect(summary: Optional[dict]) -> AspectGrade:
    from .condition import CORNERS
    from .condition_evidence import BACK_TO_FRONT_CORNER

    if not summary:
        return AspectGrade("corners", "no", {}, "not seen at full resolution")
    cons = load_condition_standards_safe()
    bounds = cons["corners"]["severity_bounds_mm"]
    names = cons["corners"]["category_names"]
    per = {c: [] for c in CORNERS}            # physical corner -> [cat dists]
    for face in _faces_of(summary):
        for c, est in summary[face]["corners"].items():
            if est is None:
                continue
            phys = c if face == "front" else BACK_TO_FRONT_CORNER[c]
            per[phys].append(_categories(est.value, est.sigma, bounds))
    missing = [c for c in CORNERS if not per[c]]
    if missing:
        return AspectGrade("corners", "no", {},
                           "not yet read: " + ", ".join(_CORNER_ABBR[c] for c in missing))
    items, words = [], []
    for c in CORNERS:
        d = per[c][0]
        for more in per[c][1:]:
            d = _max_categories(d, more)
        items.append(d)
        words.append(f"{_CORNER_ABBR[c]} {names[int(np.argmax(d))]}")
    faces = _faces_of(summary)
    assessed = "both" if faces == ["front", "back"] and all(
        summary["back"]["corners"][c] is not None for c in CORNERS) else faces[0]
    return AspectGrade("corners", assessed, _enumerate(items, corner_grade_rule), "; ".join(words))


def edge_aspect(summary: Optional[dict]) -> AspectGrade:
    from .condition import SIDES
    from .condition_evidence import BACK_TO_FRONT_SIDE

    if not summary:
        return AspectGrade("edges", "no", {}, "not seen at full resolution")
    cons = load_condition_standards_safe()
    wb = cons["edges"]["whitening_bounds_frac"]
    nb = cons["edges"]["nick_bounds_mm"]
    names = cons["edges"]["category_names"]
    per = {s: [] for s in SIDES}
    for face in _faces_of(summary):
        for side, e in summary[face]["edges"].items():
            nick = e["nick"]
            if nick is None:
                continue
            phys = side if face == "front" else BACK_TO_FRONT_SIDE[side]
            d = _categories(nick.value, nick.sigma, nb)
            w = e["whitening"]
            if w is not None:
                d = _max_categories(d, _categories(w.value, w.sigma, wb))
            per[phys].append(d)
    missing = [s for s in SIDES if not per[s]]
    if missing:
        return AspectGrade("edges", "no", {}, "not yet read: " + ", ".join(missing))
    items, words = [], []
    for s in SIDES:
        d = per[s][0]
        for more in per[s][1:]:
            d = _max_categories(d, more)
        items.append(d)
        words.append(f"{s} {names[int(np.argmax(d))]}")
    faces = _faces_of(summary)
    assessed = "both" if faces == ["front", "back"] and all(
        summary["back"]["edges"][s]["nick"] is not None for s in SIDES) else faces[0]
    return AspectGrade("edges", assessed, _enumerate(items, edge_grade_rule), "; ".join(words))


def surface_aspect(summary: Optional[dict]) -> AspectGrade:
    if not summary:
        return AspectGrade("surface", "no", {}, "not inspected")
    cons = load_condition_standards_safe()["surface"]
    names = cons["category_names"]
    faces_done, worst, worst_detail = [], 10.0, ""
    notes = []
    for face in _faces_of(summary):
        s = summary[face]["surface"]
        if s.get("foil"):
            notes.append(f"{face}: foil -- scratches can't be told from the pattern")
            continue
        if not s["assessed"]:
            notes.append(f"{face}: {int(round(100 * s['coverage']))}% under a highlight so far")
            continue
        faces_done.append(face)
        defects = s["defects"]
        n = len(defects)
        L = max((d["length_mm"] for d in defects), default=0.0)
        if n == 0:
            cat = 0
        elif L >= cons["crease_mm"] or n > 8:
            cat = 4
        elif n == 1 and L < 2.0:
            cat = 1
        elif n <= 3 and L < 6.0:
            cat = 2
        else:
            cat = 3
        g = (10.0, 9.0, 7.0, 5.0, 3.0)[cat]
        if g <= worst:
            worst = g
            worst_detail = f"{face}: {names[cat]}" + (f" ({n} mark{'s' if n != 1 else ''}, longest {L:.1f} mm)" if n else "")
    if not faces_done:
        return AspectGrade("surface", "no", {}, "; ".join(notes) or "not inspected")
    assessed = "both" if len(faces_done) == 2 else faces_done[0]
    return AspectGrade("surface", assessed, {worst: 1.0}, worst_detail)


def load_condition_standards_safe() -> dict:
    from .condition import load_condition_standards

    return load_condition_standards()


# -- composition -------------------------------------------------------------


def _overall_rule(grader: str, grades: tuple) -> float:
    if grader == "BGS":
        lowest = min(grades)
        avg = sum(grades) / len(grades)
        return max(1.0, round(min(avg, lowest + 0.5) * 2) / 2.0)
    g = min(grades)
    if grader == "PSA":
        return float(round(g)) if g >= 9.5 else float(math.floor(g))
    return max(1.0, round(g * 2) / 2.0)


def compose(aspects: List[AspectGrade], grader: str = "PSA") -> Dict[float, float]:
    """Exact distribution of the overall grade over independent aspects."""
    dists = [a.distribution for a in aspects if a.is_assessed]
    if not dists:
        return {}
    if grader != "BGS":
        # weakest governs: P(overall >= g) = prod P(aspect >= g)
        grades = sorted({g for d in dists for g in d}, reverse=True)
        surv = []
        for g in grades:
            p = 1.0
            for d in dists:
                p *= sum(v for k, v in d.items() if k >= g)
            surv.append(p)
        out: Dict[float, float] = {}
        prev = 0.0
        for g, p in zip(grades, surv):
            sg = _overall_rule(grader, (g,))
            out[sg] = out.get(sg, 0.0) + max(0.0, p - prev)
            prev = p
        return _norm(out)
    out = {}

    def rec(i, chosen, p):
        if p < _PRUNE:
            return
        if i == len(dists):
            g = _overall_rule(grader, tuple(chosen))
            out[g] = out.get(g, 0.0) + p
            return
        for k, v in dists[i].items():
            rec(i + 1, chosen + [k], p * v)

    rec(0, [], 1.0)
    return _norm(out)


@dataclass(frozen=True)
class CardGradePrediction:
    """Estimated overall card grade, condition classification, and subgrades.

    ``complete`` is True only when centering, corners, edges and surface all
    have evidence. Otherwise ``grade_score`` is the CEILING the assessed
    aspects allow (most likely value), ``grade_label`` says "max", and the
    unassessed subgrades are None.
    """

    grader: str
    grade_score: float
    grade_label: str
    condition_name: str
    centering_subgrade: float
    estimated_corners: Optional[float]
    estimated_edges: Optional[float]
    estimated_surface: Optional[float]
    grade_ceiling: GradeBand
    probabilities: dict[str, float]
    confidence: float
    summary: str
    used_learned: bool = False
    n_observations: int = 0
    complete: bool = False
    aspects: Dict[str, AspectGrade] = field(default_factory=dict)
    missing: tuple = ()

    def describe(self) -> str:
        def sub(v):
            return "not assessed" if v is None else f"{v:.1f}"

        lines = [
            f"=== {self.grader} ESTIMATED GRADE: {self.grade_label} ({self.condition_name}) ===",
            f"  Centering Subgrade : {self.centering_subgrade:.1f}",
            f"  Corners Subgrade   : {sub(self.estimated_corners)}",
            f"  Edges Subgrade     : {sub(self.estimated_edges)}",
            f"  Surface Subgrade   : {sub(self.estimated_surface)}",
            f"  Confidence         : {int(self.confidence * 100)}%",
            "  Probabilities      : " + ", ".join(f"{g}: {int(p*100)}%" for g, p in sorted(self.probabilities.items(), key=lambda kv: -kv[1])),
            f"  Centering Ceiling  : {self.grade_ceiling.best if self.grade_ceiling.is_single else f'{self.grade_ceiling.worst}-{self.grade_ceiling.best}'}",
            f"  Notes              : {self.summary}",
        ]
        if self.missing:
            lines.append("  Not assessed       : " + ", ".join(self.missing)
                         + " -- the grade above is a ceiling, not an estimate")
        if self.used_learned:
            lines.append(
                f"  Learned            : {self.n_observations} certified "
                f"{self.grader} observation(s) in this ratio band"
            )
        return "\n".join(lines)

    def aspects_dict(self) -> dict:
        return {k: v.to_dict() for k, v in self.aspects.items()}


def _snap_grade(score: float, grader: str) -> float:
    score = max(1.0, min(10.0, float(score)))
    if grader == "PSA":
        return float(round(score)) if score >= 9.5 else float(math.floor(score))
    return round(score * 2) / 2.0


def _ceiling_score(band: GradeBand) -> Optional[float]:
    from .learning import parse_issued_grade

    return parse_issued_grade(band.worst) or parse_issued_grade(band.best)


def condition_summary(condition: Any) -> Optional[dict]:
    """Accept a ConditionEvidence, a single ConditionView, or an already-made
    summary dict."""
    if condition is None:
        return None
    if isinstance(condition, dict):
        return condition
    from .condition import ConditionView
    from .condition_evidence import ConditionEvidence

    if isinstance(condition, ConditionView):
        condition = ConditionEvidence.from_view(condition)
    if isinstance(condition, ConditionEvidence):
        return condition.summary()
    raise TypeError(f"unsupported condition evidence: {type(condition).__name__}")


def predict_overall_grade(
    ratio: Measured,
    quality: Optional[Any] = None,
    geometry: Optional[Any] = None,
    grader: str = "PSA",
    face: Face = "front",
    model: Optional["GradeOutcomeModel"] = None,
    condition: Optional[Any] = None,
    back_ratio: Optional[Measured] = None,
) -> CardGradePrediction:
    """Grade distribution from every aspect that has evidence.

    ``ratio`` is the worst-axis centering of ``face``; ``back_ratio`` the
    back's, when it was measured. ``condition`` is the corner/edge/surface
    evidence (``ConditionEvidence``, a single ``ConditionView``, or its
    ``summary()``); without it only centering is assessed and the result is a
    ceiling. ``quality`` feeds only the certified-label model's quality band
    -- detection quality is no longer read as card condition. ``geometry`` is
    accepted for compatibility and unused.

    IDENTITY REDUCTION. With ``model is None`` or a model that has no certified
    observations in this ratio band, the result is the measured-evidence
    distribution. Certified labels move the overall grade and its
    probabilities only; they never raise the published centering ceiling, and
    they never train on this function's own output.
    """
    band = grade_band(ratio, grader=grader, face=face)
    cdist = centering_distribution(ratio, grader, face)
    centering_assessed = face
    if back_ratio is not None and face == "front":
        bdist = centering_distribution(back_ratio, grader, "back")
        cdist = compose([AspectGrade("c", "front", cdist), AspectGrade("c", "back", bdist)], grader="PSA"
                        if grader != "BGS" else "CGC")
        centering_assessed = "both"
    lo, hi = ratio.interval()
    centering = AspectGrade("centering", centering_assessed, cdist,
                            f"{max(50.0, ratio.value):.1f}/{100 - max(50.0, ratio.value):.1f} "
                            f"(95% CI {lo:.1f}-{hi:.1f})")
    summ = condition_summary(condition)
    aspects = {
        "centering": centering,
        "corners": corner_aspect(summ),
        "edges": edge_aspect(summ),
        "surface": surface_aspect(summ),
    }
    missing = tuple(k for k, a in aspects.items() if not a.is_assessed)
    complete = not missing
    dist = compose(list(aspects.values()), grader)
    probs = {_key(k): float(v) for k, v in sorted(dist.items(), reverse=True)}
    final_score = max(dist.items(), key=lambda kv: (kv[1], kv[0]))[0]
    conf = float(dist[final_score])

    def label(score: float) -> str:
        g = int(score) if float(score).is_integer() else score
        return f"{grader} {g}" if complete else f"{grader} {g} max"

    centering_sub = centering.modal if centering.modal is not None else 1.0
    final_score = max(1.0, min(10.0, final_score))
    cond_name = CONDITION_NAMES.get(final_score, "Authentic")
    grade_label = label(final_score)
    if complete:
        summary = (f"Predicted {grade_label} ({cond_name}) from measured centering, corners, "
                   f"edges and surface.")
    else:
        summary = (f"Ceiling {grade_label} ({cond_name}) from "
                   + ", ".join(k for k in ASPECTS if k not in missing)
                   + "; not assessed: " + ", ".join(missing)
                   + ". The card cannot grade above this and may grade below it.")
    inner_conf = getattr(quality, "inner_confidence", 0.9) if quality else 0.9
    used_learned = False
    n_obs = 0

    if model is not None:
        bin_counts = model.bin_counts(grader, ratio.value, inner_conf)
        n_obs = sum(bin_counts.values())
        if n_obs > 0:
            from .learning import parse_issued_grade

            blended = model.posterior(
                grader, ratio.value, inner_conf, heuristic=probs
            )
            if blended:
                ceiling = _ceiling_score(band)
                if ceiling is not None:
                    blended = {
                        k: v
                        for k, v in blended.items()
                        if (parse_issued_grade(k) or 0.0) <= ceiling + 1e-9
                    }
                    total = sum(blended.values())
                    if total > 0:
                        blended = {k: v / total for k, v in blended.items()}
                if blended:
                    probs = blended
                    expected = 0.0
                    mass = 0.0
                    for key, weight in blended.items():
                        score = parse_issued_grade(key)
                        if score is None:
                            continue
                        expected += score * weight
                        mass += weight
                    if mass > 0:
                        final_score = _snap_grade(expected / mass, grader)
                        if ceiling is not None:
                            final_score = min(final_score, ceiling)
                        cond_name = CONDITION_NAMES.get(final_score, "Authentic")
                        grade_label = label(final_score)
                    used_learned = True
                    conf = float(probs.get(_key(final_score), conf))
                    summary = (
                        f"Predicted {grade_label} ({cond_name}) from {n_obs} "
                        f"certified {grader} label(s) at "
                        f"{ratio.value:.1f}/{100 - ratio.value:.1f} centering, "
                        f"blended with the measured evidence."
                    )

    return CardGradePrediction(
        grader=grader,
        grade_score=float(final_score),
        grade_label=grade_label,
        condition_name=cond_name,
        centering_subgrade=float(centering_sub),
        estimated_corners=None if aspects["corners"].modal is None else float(aspects["corners"].modal),
        estimated_edges=None if aspects["edges"].modal is None else float(aspects["edges"].modal),
        estimated_surface=None if aspects["surface"].modal is None else float(aspects["surface"].modal),
        grade_ceiling=band,
        probabilities=probs,
        confidence=round(min(0.99, conf), 2),
        summary=summary,
        used_learned=used_learned,
        n_observations=n_obs,
        complete=complete,
        aspects=aspects,
        missing=missing,
    )


def predict_all_grades(
    ratio: Measured,
    quality: Optional[Any] = None,
    geometry: Optional[Any] = None,
    face: Face = "front",
    model: Optional["GradeOutcomeModel"] = None,
    condition: Optional[Any] = None,
    back_ratio: Optional[Measured] = None,
) -> dict[str, CardGradePrediction]:
    """Predict grades across all supported grading houses (PSA, BGS, CGC, SGC)."""
    summ = condition_summary(condition)
    return {
        g: predict_overall_grade(
            ratio, quality=quality, geometry=geometry, grader=g, face=face, model=model,
            condition=summ, back_ratio=back_ratio,
        )
        for g in available_graders()
    }
