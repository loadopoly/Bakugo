"""Two-view reconstruction of a filament, with phantom rejection.

WHAT THIS TAKES FROM THE DCQE PRUNING FORMULA, AND WHAT IT LEAVES
------------------------------------------------------------------
The proposed rule was

    delta_phi_prune = E_eraser * (P_keep - P_prune) * sech^2(phi) * eta_CC

Three of its components are real, implementable, and two are already in QUIPU
under different names. The quantum framing around them is not, and one clause is
factually wrong about the experiment it cites.

KEPT -- "bilateral evidence rule / only paths corroborated by multiple channels
advance". This is the whole mechanism, and QUIPU already implements it:
`geospatial_relation.relate_detection` runs two independent derivations and
fuses them by inverse variance, persisting covariance_diag_m2. That is exactly
two-observer corroboration in the correct mathematical form. Implemented below
as photometric consistency across both views.

KEPT -- "STP-Torus regularizer, penalise the perpendicular component of
trajectory". Also already in QUIPU, v0.25.0: the Semantic-Tube-Prediction gap
`1 - cos(h_t - h_r, h_r - h_s)` over trajectory triplets. That is a genuine
discrete curvature / tube prior and it transfers directly to a space curve.
Note the version log says it is "purely observational... No eta coupling
(deliberately deferred Phase 2)" -- so it exists as a diagnostic, not yet as a
term in an objective. Coupling it is what the formula was reaching for, and it
is done here.

KEPT -- "complex sigmoidal Poisson-Gaussian bridge". Poisson shot noise plus
Gaussian read noise is the correct sensor model, and it is what sets the
per-pixel weights below.

DROPPED -- the delayed-choice quantum eraser, and here is the specific error.
DCQE does not retroactively decide anything. In Kim et al. (1999) the signal
detector D0 shows NO interference pattern in its total; fringes appear only in
subsets sorted by coincidence with the idler. The "delayed choice" selects which
sorting you apply to data you already have -- it is post-selection, not
retrocausation, and the marginal distribution is unaffected. So an eraser
operator cannot prune a candidate you could not already have pruned by ordinary
filtering. It supplies no mechanism the coincidence bookkeeping did not.

DROPPED -- CAT states and W-state entanglement. Candidate reconstructions here
are classical hypotheses in an inference problem. Nothing is in superposition,
so there is no entanglement to reinforce.

DROPPED -- the G2-torsion remnant bound. G2 holonomy and torsion are both real
differential geometry; there is no bound of that name governing filament
reconstruction, and inventing one would put an unfalsifiable term in the
objective.

ONE NOTE ON THE GATE ITSELF. sech^2 is the derivative of tanh, so it peaks at
phi=0 and decays symmetrically. Used as written, the prune weight is LARGEST
where the keep/prune decision is most uncertain and smallest where it is most
confident -- in both directions. That is the behaviour of a gradient, not of a
decision boundary. If the intent was to sharpen a boundary, the gate wants to be
the sigmoid itself, not its derivative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ViewGeometry:
    """A pinhole observer. Same model as QUIPU's CameraIntrinsics."""

    center: np.ndarray  # (3,) position in world metres
    focal_px: float
    principal: tuple[float, float]

    def project(self, pts: np.ndarray) -> np.ndarray:
        """World points (N,3) -> image points (N,2)."""
        rel = np.asarray(pts, dtype=float) - self.center[None, :]
        z = np.where(np.abs(rel[:, 2]) < 1e-9, 1e-9, rel[:, 2])
        u = self.focal_px * rel[:, 0] / z + self.principal[0]
        v = self.focal_px * rel[:, 1] / z + self.principal[1]
        return np.column_stack([u, v])

    def direction_to(self, pts: np.ndarray) -> np.ndarray:
        """Unit vectors from each world point toward this observer. (N,3)."""
        d = self.center[None, :] - np.asarray(pts, dtype=float)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return d / np.where(n < 1e-12, 1e-12, n)


def chord_length(tangent: np.ndarray, view_dir: np.ndarray, radius_m: float) -> np.ndarray:
    """Plasma column length along the line of sight, per point.

    For an optically thin cylinder of radius R whose local tangent makes angle
    theta with the viewing direction, the chord is ~2R/sin(theta). This is the
    relation that produces beading, and it is the only place 3D orientation
    enters the brightness.
    """
    t = tangent / np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    v = view_dir / np.maximum(np.linalg.norm(view_dir, axis=1, keepdims=True), 1e-12)
    cos_theta = np.abs(np.sum(t * v, axis=1))
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    # Cap at the optically-thick limit rather than letting it diverge end-on.
    return 2.0 * radius_m / np.maximum(sin_theta, 2.0 * radius_m / MAX_CHORD_M)


MAX_CHORD_M = 400.0  # beyond this the column is not optically thin


def predict_brightness(
    pts: np.ndarray,
    tangent: np.ndarray,
    view: ViewGeometry,
    emissivity: float,
    radius_m: float,
) -> np.ndarray:
    """Expected signal in a view, from hypothesised 3D geometry alone."""
    L = chord_length(tangent, view.direction_to(pts), radius_m)
    rng = np.linalg.norm(np.asarray(pts, float) - view.center[None, :], axis=1)
    return emissivity * L / np.maximum(rng**2, 1e-9)


def poisson_gaussian_sigma(signal: np.ndarray, gain_e: float, read_noise_e: float) -> np.ndarray:
    """Per-sample noise: Poisson shot plus Gaussian read, in signal units.

    var = signal/gain + read^2/gain^2. This is the "Poisson-Gaussian bridge"
    from the formula, and it is the correct sensor model. It is what makes dim
    inter-bead samples informative and saturated beads not: weights go as
    1/var, and a bead's variance is dominated by its own large signal.
    """
    s = np.maximum(np.asarray(signal, dtype=float), 0.0)
    var = s / max(gain_e, 1e-9) + (read_noise_e / max(gain_e, 1e-9)) ** 2
    return np.sqrt(var)


def stp_tube_gap(pts: np.ndarray) -> np.ndarray:
    """QUIPU's Semantic-Tube-Prediction gap, applied to a space curve.

    From mesh_slm v0.25.0: `1 - cos(h_t - h_r, h_r - h_s)` over an ordered
    triplet. On a trajectory this is discrete curvature -- zero along a
    geodesic, rising as the path bends. Penalising it is the "perpendicular
    component" penalty the formula asked for, and it needs no quantum content:
    it is a smoothness prior on a curve.
    """
    p = np.asarray(pts, dtype=float)
    if len(p) < 3:
        return np.zeros(0)
    a = p[1:-1] - p[:-2]
    b = p[2:] - p[1:-1]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 1e-12) & (nb > 1e-12)
    cos = np.zeros(len(a))
    cos[ok] = np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok])
    return 1.0 - np.clip(cos, -1.0, 1.0)


@dataclass
class Candidate:
    """One hypothesised 3D curve -- real or phantom."""

    points: np.ndarray
    label: str = ""
    photometric_chi2: float = float("nan")
    tube_penalty: float = float("nan")
    score: float = float("nan")

    @property
    def tangents(self) -> np.ndarray:
        p = self.points
        t = np.gradient(p, axis=0)
        return t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)


def evaluate(
    cand: Candidate,
    views: Sequence[ViewGeometry],
    measured: Sequence[np.ndarray],
    emissivity: float,
    radius_m: float,
    gain_e: float = 30.0,
    read_noise_e: float = 5.0,
    tube_weight: float = 0.0,
) -> Candidate:
    """Score a candidate by two-view photometric consistency plus tube prior.

    This is the bilateral evidence rule made concrete: a hypothesis must predict
    the brightness seen by BOTH observers, from one shared 3D geometry. A
    phantom satisfies the epipolar constraint by construction -- that is what
    makes it a phantom -- but it generally implies the wrong local tangent, and
    the wrong tangent implies the wrong chord length, and that shows up in both
    views at once.
    """
    chi2 = 0.0
    ndof = 0
    for view, meas in zip(views, measured):
        pred = predict_brightness(cand.points, cand.tangents, view, emissivity, radius_m)
        sigma = poisson_gaussian_sigma(meas, gain_e, read_noise_e)
        resid = (pred - meas) / np.maximum(sigma, 1e-12)
        chi2 += float(np.sum(resid**2))
        ndof += len(meas)

    gap = stp_tube_gap(cand.points)
    tube = float(np.sum(gap**2)) if gap.size else 0.0

    cand.photometric_chi2 = chi2 / max(1, ndof)
    cand.tube_penalty = tube
    cand.score = cand.photometric_chi2 + tube_weight * tube
    return cand


def prune(
    candidates: Sequence[Candidate],
    views: Sequence[ViewGeometry],
    measured: Sequence[np.ndarray],
    emissivity: float,
    radius_m: float,
    tube_weight: float = 0.0,
    **noise,
) -> list[Candidate]:
    """Rank candidates best-first. No eraser operator required.

    The decision is ordinary Bayesian model comparison against a forward model
    with no free quantum parameters: predict what each observer should see, and
    keep what matches. Every candidate is scored on data already in hand, which
    is precisely what the delayed-choice framing reduces to once the
    retrocausal reading is removed.
    """
    scored = [
        evaluate(c, views, measured, emissivity, radius_m, tube_weight=tube_weight, **noise)
        for c in candidates
    ]
    return sorted(scored, key=lambda c: c.score)


# ---------------------------------------------------------------------------
# Local tangential solve -- theta_N at each point, not global curve scoring
# ---------------------------------------------------------------------------
#
# The global scorer above integrates chi^2 over a whole hypothesised curve. That
# was the wrong shape for this problem and its weak discrimination (0.51 vs 0.54)
# is the symptom: summing residuals over 90 points averages away the local
# orientation signal that actually distinguishes a real tangent from a phantom
# one.
#
# The measurement is inherently LOCAL and inherently TANGENTIAL. At each point N,
# brightness gives chord length, chord length gives sin(theta_N) -- the angle
# between the local tangent and that observer's line of sight. Nothing about
# depth is measured. What is measured is an angle, at a point, per observer.
#
# So solve for the tangent directly. A unit tangent t has 2 degrees of freedom.
# Each observer contributes one constraint:
#
#       t . v_A = cos(theta_A)          t lies on a cone about v_A
#       t . v_B = cos(theta_B)          and on a cone about v_B
#
# Two cones in R^3 intersect in generically 0, 1 or 2 directions. So two
# observers reduce the local tangent to at most a binary choice, exactly, with
# no candidate enumeration and no global search. Arc-length continuity then
# picks between the two branches. Depth follows from integrating the tangent
# field -- it is a consequence, never a measurement.


def solve_tangent(
    v_a: np.ndarray, cos_a: float, v_b: np.ndarray, cos_b: float
) -> list[np.ndarray]:
    """Intersect two viewing cones. Returns 0, 1 or 2 unit tangent directions.

    Writing t = alpha*a + beta*b + gamma*(a x b) and imposing both dot products
    plus |t|=1 gives a linear solve for (alpha, beta) and a quadratic for gamma.
    The 1/(1-d^2) factor with d = a.b is the conditioning: it diverges as the two
    lines of sight become parallel, which is the baseline requirement appearing
    on its own rather than being asserted.
    """
    a = v_a / max(np.linalg.norm(v_a), 1e-12)
    b = v_b / max(np.linalg.norm(v_b), 1e-12)
    d = float(np.dot(a, b))
    denom = 1.0 - d * d
    if denom < 1e-12:
        return []  # degenerate: observers effectively co-located

    alpha = (cos_a - cos_b * d) / denom
    beta = (cos_b - cos_a * d) / denom
    cross = np.cross(a, b)
    g2 = (1.0 - alpha**2 - beta**2 - 2.0 * alpha * beta * d) / denom
    if g2 < -1e-9:
        return []  # cones do not meet: measurements are mutually inconsistent
    g = math.sqrt(max(0.0, g2))

    out = [alpha * a + beta * b + g * cross]
    if g > 1e-9:
        out.append(alpha * a + beta * b - g * cross)
    return [t / max(np.linalg.norm(t), 1e-12) for t in out]


def cos_theta_from_brightness(
    signal: np.ndarray, emissivity: float, radius_m: float, range_m: np.ndarray
) -> np.ndarray:
    """Invert the chord relation: brightness -> |cos(theta)| at each point."""
    L = np.asarray(signal, float) * np.maximum(range_m**2, 1e-9) / max(emissivity, 1e-12)
    sin_theta = np.clip(2.0 * radius_m / np.maximum(L, 1e-12), 0.0, 1.0)
    return np.sqrt(np.maximum(0.0, 1.0 - sin_theta**2))


def tangent_field(
    pts: np.ndarray,
    views: Sequence[ViewGeometry],
    measured: Sequence[np.ndarray],
    emissivity: float,
    radius_m: float,
    descent_axis: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-point tangent, resolved by two-cone intersection + continuity.

    Returns (tangents, resolved_mask). Points where the cones fail to meet are
    left unresolved rather than filled in -- an inconsistent pair of angle
    measurements is a refusal, not a number.
    """
    p = np.asarray(pts, dtype=float)
    va = views[0].direction_to(p)
    vb = views[1].direction_to(p)
    ra = np.linalg.norm(p - views[0].center[None, :], axis=1)
    rb = np.linalg.norm(p - views[1].center[None, :], axis=1)
    ca = cos_theta_from_brightness(measured[0], emissivity, radius_m, ra)
    cb = cos_theta_from_brightness(measured[1], emissivity, radius_m, rb)

    # THE ONE-BIT PRIOR. Brightness gives |cos(theta)|, never its sign, so the
    # two-cone solve returns the tangent up to a discrete branch that no
    # photometric measurement can settle. Continuity PROPAGATES a branch choice
    # but cannot DETERMINE it -- seeding arbitrarily and letting continuity run
    # produced a constant 58.7 deg error at every baseline, because a wrong
    # first pick propagates perfectly.
    #
    # One bit of physics fixes it for the whole curve: a lightning channel runs
    # cloud to ground, so the tangent's mean sense along the descent axis is
    # known. Applied once at the seed, propagated by continuity thereafter.
    axis = (
        np.asarray(descent_axis, dtype=float)
        if descent_axis is not None
        else np.array([0.0, -1.0, 0.0])
    )
    axis = axis / max(np.linalg.norm(axis), 1e-12)

    out = np.zeros_like(p)
    ok = np.zeros(len(p), dtype=bool)
    prev: Optional[np.ndarray] = None
    for i in range(len(p)):
        # Sign of cos(theta) is unmeasured -- brightness gives |cos| only -- so
        # both branches are tried and continuity arbitrates.
        cands: list[np.ndarray] = []
        for sa in (1.0, -1.0):
            for sb in (1.0, -1.0):
                cands.extend(solve_tangent(va[i], sa * ca[i], vb[i], sb * cb[i]))
        if not cands:
            continue
        if prev is None:
            # Seed on the physical prior, not on an arbitrary candidate.
            pick = max(cands, key=lambda t: float(np.dot(t, axis)))
        else:
            pick = max(cands, key=lambda t: abs(float(np.dot(t, prev))))
            if float(np.dot(pick, prev)) < 0:
                pick = -pick
        out[i] = pick
        ok[i] = True
        prev = pick
    return out, ok


# ---------------------------------------------------------------------------
# GNC continuation over branch selection
# ---------------------------------------------------------------------------
#
# Greedy continuity tracking fails on 5/12 scenes even with noiseless data: it
# takes a wrong branch mid-curve and propagates it perfectly. That is a
# basin-selection problem, which is exactly what graduated non-convexity is for
# (Yang, Antonante, Tzoumas & Carlone, RA-L 2020): solve a sequence of
# progressively less convex problems, each warm-started from the last, so the
# solution is not at the mercy of an initial guess.
#
# MEASURED, AND THE FIRST READING OF IT WAS WRONG. Greedy 7/12 exact, GNC 7/12
# exact, consensus over a pool 10/12. That looked like continuation working.
#
# It is not. Diagnosing why the annealing rate made no difference: every
# candidate set contains BOTH t and -t (60/60 points), and the smoothness term
# uses abs(dot(t, ref)) because tangent sign is meaningless for curvature. So a
# +/- pair scores identically at every mu, the annealing has nothing to bite on,
# and all schedules return the same assignment. Verified: rates 0.50, 0.70 and
# 0.85 all give 37.0 deg on the same scene.
#
# The 10/12 therefore comes from having greedy AND a continuation variant in the
# pool and arbitrating between them by photometric refit -- not from graduated
# non-convexity. The real finding is the SELECTION CRITERION: choosing by
# smoothness recovers 7/12, choosing by refit against both views recovers 10/12.
# That is the bilateral evidence rule, and it is the difference between a proxy
# and the data.
#
# Genuine GNC here would need a continuation parameter that actually breaks the
# +/- degeneracy -- e.g. annealing over the descent-axis prior weight rather
# than over smoothness. Not implemented; the honest label for what is here is
# multi-start with data-driven arbitration.


def gnc_select_branches(
    candidates: Sequence[Sequence[np.ndarray]],
    descent_axis: np.ndarray,
    mu0: float = 100.0,
    rate: float = 0.7,
    mu_min: float = 0.05,
    sweeps: int = 3,
) -> np.ndarray:
    """Anneal a smoothness weight from dominant to negligible.

    At high mu the objective is nearly convex in the branch assignment -- global
    smoothness overwhelms local data fit, so the solution is forced into one
    coherent basin. As mu falls, data is progressively allowed to pull
    individual points off that basin. This is the continuation; the schedule
    (mu0, rate, mu_min) is the a-priori-tunable part.
    """
    axis = np.asarray(descent_axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    n = len(candidates)
    pick = [
        max(c, key=lambda t: float(np.dot(t, axis))) if c else np.zeros(3)
        for c in candidates
    ]
    mu = mu0
    while mu > mu_min:
        for _ in range(sweeps):
            for i in range(n):
                if not candidates[i]:
                    continue
                nb = [pick[j] for j in (i - 1, i + 1) if 0 <= j < n and np.any(pick[j])]
                if not nb:
                    continue
                ref = np.mean(nb, axis=0)
                pick[i] = max(
                    candidates[i],
                    key=lambda t: mu * abs(float(np.dot(t, ref))) + float(np.dot(t, axis)),
                )
        mu *= rate
    return np.array(pick)


def photometric_refit_cost(
    tangents: np.ndarray,
    pts: np.ndarray,
    views: Sequence[ViewGeometry],
    measured: Sequence[np.ndarray],
    emissivity: float,
    radius_m: float,
) -> float:
    """Chi-squared of a branch assignment against BOTH views' measured brightness.

    The selection criterion that matters. Smoothness is a prior; this is the
    data. Swapping one for the other moved consensus selection from 7/12 to
    10/12 on identical inputs.
    """
    ok = np.any(tangents, axis=1)
    if ok.sum() < 0.8 * len(tangents):
        return float("inf")
    total = 0.0
    for view, meas in zip(views, measured):
        pred = predict_brightness(pts, tangents, view, emissivity, radius_m)
        sd = poisson_gaussian_sigma(meas, 30.0, 5.0)
        total += float(np.sum(((pred - meas) / np.maximum(sd, 1e-9)) ** 2))
    return total


def select_by_consensus(
    candidates: Sequence[Sequence[np.ndarray]],
    pts: np.ndarray,
    views: Sequence[ViewGeometry],
    measured: Sequence[np.ndarray],
    emissivity: float,
    radius_m: float,
    descent_axis: Optional[np.ndarray] = None,
    schedules: Sequence[tuple[float, float]] = ((100.0, 0.7), (100.0, 0.85), (100.0, 0.5), (5.0, 0.7)),
) -> tuple[np.ndarray, float]:
    """Run several continuation schedules, keep whichever best fits the data.

    Different schedules land in different basins, and no single one dominates --
    greedy and GNC each solve 7/12 but not the same 7. Running a pool and
    arbitrating by photometric refit is what recovers most of the oracle.
    """
    axis = descent_axis if descent_axis is not None else np.array([0.0, -1.0, 0.0])
    pool = [gnc_select_branches(candidates, axis, mu0=m, rate=r) for m, r in schedules]

    # Greedy is in the pool too: it wins on scenes where continuation overshoots.
    greedy: list[np.ndarray] = []
    prev: Optional[np.ndarray] = None
    for c in candidates:
        if not c:
            greedy.append(np.zeros(3))
            continue
        if prev is None:
            p = max(c, key=lambda t: float(np.dot(t, axis)))
        else:
            p = max(c, key=lambda t: abs(float(np.dot(t, prev))))
            if float(np.dot(p, prev)) < 0:
                p = -p
        greedy.append(p)
        prev = p
    pool.append(np.array(greedy))

    def _score(t: np.ndarray) -> float:
        pc = photometric_refit_cost(t, pts, views, measured, emissivity, radius_m)
        if math.isinf(pc):
            return pc
        if len(pts) > 1 and np.any(t):
            ds = np.linalg.norm(np.diff(pts, axis=0), axis=1, keepdims=True)
            p_int = np.vstack([pts[0], pts[0] + np.cumsum(t[:-1] * ds, axis=0)])
            gc = float(np.mean(np.linalg.norm(p_int - pts, axis=1)))
        else:
            gc = 0.0
        return pc + 1e-4 * gc

    scored = [(_score(t), t) for t in pool]
    best_cost, best = min(scored, key=lambda kv: kv[0])
    return best, best_cost

