"""Tests for two-view filament reconstruction.

This module documents a NEGATIVE result. The geometry is exact and the tests
below prove it; the method is nonetheless unusable at lightning ranges because
the cone-intersection conditioning amplifies photometric noise faster than any
achievable SNR gain suppresses it. These tests exist so nobody rebuilds it
believing it works, including whoever reads this next.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cardcenter.channel import (
    Candidate,
    ViewGeometry,
    chord_length,
    cos_theta_from_brightness,
    poisson_gaussian_sigma,
    predict_brightness,
    solve_tangent,
    stp_tube_gap,
    tangent_field,
)

EMIS, R = 6e6, 0.02


def _scene(n=90, phase=0.0):
    s = np.linspace(0, 1, n)
    pts = np.column_stack(
        [120 * np.sin(3.1 * s) + 40 * s, -900 * s + 900, 2000 + 260 * np.sin(5.3 * s + phase)]
    )
    t = np.gradient(pts, axis=0)
    return pts, t / np.linalg.norm(t, axis=1, keepdims=True)


def _views(baseline=900.0):
    return (
        ViewGeometry(np.array([0.0, 0.0, 0.0]), 2400.0, (960.0, 540.0)),
        ViewGeometry(np.array([baseline, 0.0, 0.0]), 2400.0, (960.0, 540.0)),
    )


# --------------------------------------------------------------------------
# The physics is right
# --------------------------------------------------------------------------


def test_chord_relation_is_symmetric_in_tangent_sign() -> None:
    """The two-to-one map, stated as a test. A segment tilted toward the
    observer and one tilted away give identical chord length, which is why no
    photometric measurement recovers the sign."""
    v = np.array([[0.0, 0.0, 1.0]])
    for ang in (20.0, 45.0, 70.0):
        a = np.array([[math.sin(math.radians(ang)), 0.0, math.cos(math.radians(ang))]])
        b = np.array([[math.sin(math.radians(-ang)), 0.0, math.cos(math.radians(-ang))]])
        assert chord_length(a, v, R) == pytest.approx(chord_length(b, v, R))


def test_brightness_inversion_is_exact() -> None:
    pts, tan = _scene()
    A, _ = _views()
    sig = predict_brightness(pts, tan, A, EMIS, R)
    rng = np.linalg.norm(pts - A.center[None, :], axis=1)
    got = cos_theta_from_brightness(sig, EMIS, R, rng)
    want = np.abs(np.sum(tan * A.direction_to(pts), axis=1))
    assert np.abs(got - want).max() < 1e-9


def test_two_cones_contain_the_true_tangent() -> None:
    """Two observers reduce the local tangent to a small discrete set that
    always includes the truth. This is the part that works."""
    pts, tan = _scene()
    A, B = _views()
    va, vb = A.direction_to(pts), B.direction_to(pts)
    ca = np.abs(np.sum(tan * va, axis=1))
    cb = np.abs(np.sum(tan * vb, axis=1))
    worst = 0.0
    for i in range(len(pts)):
        cands = []
        for sa in (1.0, -1.0):
            for sb in (1.0, -1.0):
                cands.extend(solve_tangent(va[i], sa * ca[i], vb[i], sb * cb[i]))
        assert cands
        best = max(abs(float(np.dot(t, tan[i]))) for t in cands)
        worst = max(worst, math.degrees(math.acos(min(1.0, best))))
    assert worst < 0.01


def test_solve_refuses_when_observers_are_colocated() -> None:
    v = np.array([0.0, 0.0, 1.0])
    assert solve_tangent(v, 0.5, v.copy(), 0.5) == []


def test_solve_refuses_inconsistent_angles() -> None:
    """Two mutually impossible angle measurements produce no tangent rather
    than a fabricated one."""
    assert solve_tangent(np.array([1.0, 0, 0]), 0.99, np.array([0, 1.0, 0]), 0.99) == []


def test_noiseless_recovery_is_bimodal_not_reliable() -> None:
    """SECOND NEGATIVE RESULT, and it is independent of noise.

    With perfect data and the one-bit seed, recovery across 12 scene phases is
    BIMODAL: 7/12 land at 0.000 deg and 5/12 exceed 20 deg, worst 63.6. The
    failure is discrete, not gradual -- continuity tracking takes a wrong branch
    somewhere mid-curve and propagates it perfectly thereafter, exactly as the
    arbitrary seed did before the prior was added. One bit fixes the START of
    the curve; it does nothing for branch swaps in the middle of it.

    An earlier single-seed check reported 0.001 deg and was taken as proof the
    method worked. It was one of the 7."""
    errs = []
    for phase in np.linspace(0, 6, 12):
        pts, tan = _scene(phase=phase)
        A, B = _views()
        meas = [predict_brightness(pts, tan, V, EMIS, R) for V in (A, B)]
        got, ok = tangent_field(pts, [A, B], meas, EMIS, R)
        d = np.abs(np.sum(got[ok] * tan[ok], axis=1))
        errs.append(np.median(np.degrees(np.arccos(np.clip(d, 0, 1)))))
    errs = np.array(errs)
    assert (errs < 1.0).sum() >= 4, "some scenes must recover exactly"
    assert (errs > 20.0).sum() >= 2, (
        "some scenes must fail badly -- if this stops firing, mid-curve branch "
        "tracking has been fixed and the finding needs redoing"
    )


def test_one_bit_prior_is_what_fixes_the_branch() -> None:
    """Seeding on the descent axis vs its opposite must change the answer --
    proving the branch is genuinely underdetermined by photometry alone."""
    pts, tan = _scene()
    A, B = _views()
    meas = [predict_brightness(pts, tan, V, EMIS, R) for V in (A, B)]
    down, _ = tangent_field(pts, [A, B], meas, EMIS, R, descent_axis=np.array([0.0, -1.0, 0.0]))
    up, _ = tangent_field(pts, [A, B], meas, EMIS, R, descent_axis=np.array([0.0, 1.0, 0.0]))
    assert not np.allclose(down, up)


# --------------------------------------------------------------------------
# And it still does not work
# --------------------------------------------------------------------------


def test_conditioning_diverges_as_baseline_shrinks() -> None:
    """1/(1-d^2) with d = cos(angle between lines of sight). This is the term
    that kills the method, and it is pure geometry."""
    pts, _ = _scene()
    A0 = ViewGeometry(np.array([0.0, 0.0, 0.0]), 2400.0, (960.0, 540.0))
    conds = []
    for base in (140.0, 400.0, 900.0, 1800.0):
        B = ViewGeometry(np.array([base, 0.0, 0.0]), 2400.0, (960.0, 540.0))
        d = float(np.mean(np.sum(A0.direction_to(pts) * B.direction_to(pts), axis=1)))
        conds.append(1.0 / (1.0 - d * d))
    assert all(a > b for a, b in zip(conds, conds[1:]))  # worse as baseline shrinks
    assert conds[0] > 100


@pytest.mark.parametrize("snr_gain", [1.0, 4.0, 16.0])
def test_accuracy_does_not_improve_with_photometric_snr(snr_gain: float) -> None:
    """THE NEGATIVE RESULT.

    Measured across baselines 140-1800m and SNR gains 1x-16x: resolved fraction
    rises from 26% to 87%, but median tangent error stays at 42-47 degrees
    throughout. Better photometry buys coverage and buys no accuracy, because
    the conditioning amplifies angular error faster than averaging suppresses
    it. A 16x SNR improvement is roughly 256 frames averaged; that is not a
    gap better engineering closes.
    """
    rng = np.random.default_rng(11)
    errs = []
    for seed in range(4):
        pts, tan = _scene(phase=seed * 1.3)
        A, B = _views(900.0)
        meas = []
        for V in (A, B):
            sig = predict_brightness(pts, tan, V, EMIS, R)
            sd = poisson_gaussian_sigma(sig, 30.0, 5.0) / snr_gain
            meas.append(np.maximum(0.0, sig + rng.normal(0, sd)))
        got, ok = tangent_field(pts, [A, B], meas, EMIS, R)
        if ok.sum() == 0:
            continue
        d = np.abs(np.sum(got[ok] * tan[ok], axis=1))
        errs.append(np.median(np.degrees(np.arccos(np.clip(d, 0, 1)))))
    assert errs
    assert np.median(errs) > 15.0, (
        "tangent error fell below 15 deg under noise -- if this fires, the "
        "conditioning result has changed and the negative finding needs redoing"
    )


def test_stp_tube_gap_is_zero_on_a_straight_line() -> None:
    """QUIPU's 1 - cos(h_t-h_r, h_r-h_s), applied to a curve, is discrete
    curvature: zero along a geodesic."""
    line = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
    assert np.allclose(stp_tube_gap(line), 0.0, atol=1e-12)


def test_stp_tube_gap_rises_with_curvature() -> None:
    s = np.linspace(0, 1, 40)
    gentle = np.column_stack([s, 0.05 * np.sin(2 * s), np.zeros_like(s)])
    sharp = np.column_stack([s, 0.05 * np.sin(20 * s), np.zeros_like(s)])
    assert stp_tube_gap(sharp).sum() > stp_tube_gap(gentle).sum()


def test_poisson_gaussian_weighting_favours_dim_samples() -> None:
    """Weights go as 1/var, and a bright sample's variance is dominated by its
    own shot noise -- so dim inter-bead samples carry more information per
    photon than saturated beads."""
    dim, bright = np.array([10.0]), np.array([10000.0])
    assert (poisson_gaussian_sigma(dim, 30.0, 5.0) / dim)[0] > (
        poisson_gaussian_sigma(bright, 30.0, 5.0) / bright
    )[0]


# --------------------------------------------------------------------------
# GNC continuation and consensus selection
# --------------------------------------------------------------------------


def _candidate_pool(pts, tan, views):
    from cardcenter.channel import cos_theta_from_brightness, solve_tangent

    meas = [predict_brightness(pts, tan, V, EMIS, R) for V in views]
    out = []
    for i in range(len(pts)):
        cos = []
        for V, m in zip(views, meas):
            r = np.linalg.norm(pts[i] - V.center)
            cos.append(cos_theta_from_brightness(np.array([m[i]]), EMIS, R, np.array([r]))[0])
        va = views[0].direction_to(pts[i : i + 1])[0]
        vb = views[1].direction_to(pts[i : i + 1])[0]
        cs = []
        for sa in (1.0, -1.0):
            for sb in (1.0, -1.0):
                cs.extend(solve_tangent(va, sa * cos[0], vb, sb * cos[1]))
        out.append(cs)
    return out, meas


def _tangent_error(got, truth):
    ok = np.any(got, axis=1)
    d = np.abs(np.sum(got[ok] * truth[ok], axis=1))
    return float(np.median(np.degrees(np.arccos(np.clip(d, 0, 1)))))


def test_consensus_beats_either_selector_alone() -> None:
    """THE RESULT THAT MADE GNC WORTH ADDING.

    Greedy solves 7/12 scenes exactly. GNC solves 7/12. They are not the same
    7 -- an oracle over the pool is 12/12, so the right assignment is always
    present. Arbitrating by photometric refit against both views recovers
    10/12; arbitrating by smoothness recovers only 7/12, because the smoothest
    assignment is not the true one.
    """
    from cardcenter.channel import gnc_select_branches, select_by_consensus

    views = _views(900.0)
    axis = np.array([0.0, -1.0, 0.0])
    greedy_ok = gnc_ok = consensus_ok = 0
    for phase in np.linspace(0, 6, 12):
        pts, tan = _scene(n=60, phase=phase)
        cands, meas = _candidate_pool(pts, tan, views)
        gnc = gnc_select_branches(cands, axis)
        best, _ = select_by_consensus(cands, pts, views, meas, EMIS, R)
        got, _ = tangent_field(pts, list(views), meas, EMIS, R)
        greedy_ok += _tangent_error(got, tan) < 1.0
        gnc_ok += _tangent_error(gnc, tan) < 1.0
        consensus_ok += _tangent_error(best, tan) < 1.0
    assert consensus_ok > max(greedy_ok, gnc_ok), (
        f"consensus {consensus_ok} did not beat greedy {greedy_ok} / gnc {gnc_ok}"
    )
    assert consensus_ok >= 9


def test_annealing_schedule_is_inert_and_that_is_documented() -> None:
    """NEGATIVE RESULT ABOUT THE CONTINUATION ITSELF.

    Every candidate set contains both t and -t, and the smoothness term uses
    abs(dot) because sign is meaningless for curvature. So a +/- pair scores
    identically at every mu and the annealing has nothing to act on: rates
    0.50, 0.70 and 0.85 all return the same assignment.

    The 10/12 consensus result is multi-start plus data-driven arbitration, NOT
    graduated non-convexity. This test exists so that claim stays honest -- if
    it ever starts failing, a schedule has become genuinely active and the
    labelling should be revisited."""
    from cardcenter.channel import gnc_select_branches

    views = _views(900.0)
    axis = np.array([0.0, -1.0, 0.0])
    for phase in np.linspace(0, 6, 6):
        pts, tan = _scene(n=60, phase=phase)
        cands, _ = _candidate_pool(pts, tan, views)
        fast = gnc_select_branches(cands, axis, rate=0.5)
        slow = gnc_select_branches(cands, axis, rate=0.85)
        assert np.allclose(fast, slow), (
            "annealing rate changed the result -- the sign degeneracy that made "
            "the schedule inert has been broken; re-check the GNC labelling"
        )


def test_photometric_cost_prefers_the_true_assignment() -> None:
    from cardcenter.channel import photometric_refit_cost

    views = _views(900.0)
    pts, tan = _scene(n=60, phase=0.0)
    meas = [predict_brightness(pts, tan, V, EMIS, R) for V in views]
    true_cost = photometric_refit_cost(tan, pts, views, meas, EMIS, R)
    wrong = tan.copy()
    wrong[20:40] = -wrong[20:40]
    assert photometric_refit_cost(wrong, pts, views, meas, EMIS, R) >= true_cost
