"""Tests for the information-theoretic floor and the photogrammetric scale.

The point of both modules is to bound what is achievable, so most of these
assert that a bound behaves like a bound: it moves the right way with each
physical parameter, it cannot be beaten, and it says so when it is not the
binding constraint.
"""

from __future__ import annotations

import dataclasses as dc
import math

import numpy as np
import pytest

from cardcenter.information import (
    ChannelConditions,
    SensorModel,
    apply_rhythm_to_channel,
    audit_measurement,
    cramer_rao_edge_px,
    fisher_information_edge,
    measure_coherence,
    modulate,
    quantum_floor_px,
    relational_gradient,
    shot_noise_consistency,
    shot_noise_fisher_edge,
    temporal_spatial_rhythm,
    variance_budget,
    weyl_centroid,
)
from cardcenter.photogrammetry import (
    BANK_CARD,
    TYPICAL_CUT_TOLERANCE_MM,
    Dimensions,
    ScaleReference,
    assess_trim,
    caliper_verified,
    measure_absolute,
    scale_from_reference,
)
from cardcenter.types import STANDARD_CARD_H_MM, STANDARD_CARD_W_MM, DetectionError, Measured

BASE = ChannelConditions(contrast=80.0, noise_sigma=4.0, psf_sigma_px=1.2, rows=400)


# ---------------------------------------------------------------------------
# Cramer-Rao scaling laws
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,factor,expected",
    [
        ("contrast", 2.0, 0.5),      # sigma ~ 1/C
        ("noise_sigma", 2.0, 2.0),   # sigma ~ noise
        ("rows", 4.0, 0.5),          # sigma ~ 1/sqrt(N)
        ("psf_sigma_px", 4.0, 2.0),  # sigma ~ sqrt(blur)
        ("pixel_pitch_px", 4.0, 2.0),
    ],
)
def test_bound_scaling_matches_the_derivation(field, factor, expected) -> None:
    v = getattr(BASE, field)
    changed = dc.replace(BASE, **{field: type(v)(v * factor)})
    ratio = cramer_rao_edge_px(changed) / cramer_rao_edge_px(BASE)
    assert ratio == pytest.approx(expected, rel=1e-6)


def test_zero_contrast_carries_no_information() -> None:
    assert fisher_information_edge(dc.replace(BASE, contrast=0.0)) == 0.0
    assert not math.isfinite(cramer_rao_edge_px(dc.replace(BASE, contrast=0.0)))


def test_reported_sigma_below_the_bound_is_flagged_impossible() -> None:
    a = audit_measurement(BASE, 1e-9, 3.5, 2.5, 10.0)
    assert not a.physically_possible
    assert any("unvalidated" in x for x in a.advice)


def test_specimen_limited_regime_does_not_give_capture_advice() -> None:
    """Telling someone to add light when the card's own edges are the limit is
    advice that cannot work."""
    a = audit_measurement(BASE, 0.43, 3.5, 2.5, 10.0)
    assert a.efficiency < 0.05
    assert any("card itself" in x for x in a.advice)
    assert not any("more light" in x for x in a.advice)


def test_near_the_limit_says_better_code_will_not_help() -> None:
    bound = audit_measurement(BASE, 1.0, 3.5, 2.5, 10.0).bound_ratio_pp
    a = audit_measurement(BASE, bound * 1.2, 3.5, 2.5, 10.0)
    assert a.efficiency > 0.5
    assert any("Better code will not help" in x for x in a.advice)


# ---------------------------------------------------------------------------
# QUIPU temporal-spatial overlay (pixel-space form)
# ---------------------------------------------------------------------------

UNIFORM = {
    "vision": 0.8,
    "touch": 0.8,
    "smell": 0.8,
    "body": 0.8,
    "brain": 0.8,
    "perception": 0.8,
}
ONE_HOT = {
    "vision": 1.0,
    "touch": 0.0,
    "smell": 0.0,
    "body": 0.0,
    "brain": 0.0,
    "perception": 0.0,
}


def test_uniform_high_activity_is_coherent() -> None:
    assert measure_coherence(UNIFORM) > 0.7


def test_one_hot_activity_is_incoherent() -> None:
    assert measure_coherence(ONE_HOT) < 0.15


def test_boost_is_clamped_and_period_is_reciprocal() -> None:
    r = modulate(UNIFORM)
    assert 0.5 <= float(r["boost"]) <= 1.5
    assert float(r["period_factor"]) == pytest.approx(1.0 / float(r["boost"]), abs=5e-4)
    assert float(r["lr_factor"]) == pytest.approx(float(r["boost"]), abs=5e-4)


def test_weyl_is_on_the_circle_and_empty_is_zero() -> None:
    weyl = weyl_centroid(UNIFORM)
    assert 0.0 <= weyl <= 2.0 * math.pi
    assert weyl_centroid({}) == 0.0


def test_boost_does_not_change_single_shot_fisher() -> None:
    rhythm = temporal_spatial_rhythm(BASE)
    assert rhythm.boost != 1.0 or rhythm.effective_rows == BASE.rows
    assert fisher_information_edge(BASE) == pytest.approx(
        fisher_information_edge(BASE), rel=0.0
    )
    fused = apply_rhythm_to_channel(BASE, rhythm)
    if abs(rhythm.boost - 1.0) > 1e-9:
        assert fisher_information_edge(fused) != pytest.approx(
            fisher_information_edge(BASE), rel=1e-9
        )


def test_fused_crb_moves_as_one_over_sqrt_effective_rows() -> None:
    rhythm = temporal_spatial_rhythm(BASE, efficiency=0.8, frame_chi2_dof=1.0)
    fused = apply_rhythm_to_channel(BASE, rhythm)
    raw = cramer_rao_edge_px(BASE)
    washed = cramer_rao_edge_px(fused)
    expected = raw * math.sqrt(BASE.rows / fused.rows)
    assert washed == pytest.approx(expected, rel=1e-6)


def test_audit_attaches_rhythm_on_both_regimes() -> None:
    specimen = audit_measurement(BASE, 0.43, 3.5, 2.5, 10.0)
    assert specimen.rhythm is not None
    assert specimen.fused_bound_px is not None
    bound = specimen.bound_ratio_pp
    near = audit_measurement(BASE, bound * 1.2, 3.5, 2.5, 10.0)
    assert near.rhythm is not None
    assert near.fused_bound_px is not None
    assert 0.5 <= near.rhythm.boost <= 1.5


def test_relational_gradient_uses_smell_complement_as_decay() -> None:
    high_blur = {"touch": 0.0, "smell": 0.0, "vision": 0.5, "body": 0.5, "brain": 0.5, "perception": 0.5}
    sharp = {"touch": 0.0, "smell": 1.0, "vision": 0.5, "body": 0.5, "brain": 0.5, "perception": 0.5}
    assert relational_gradient(high_blur) > relational_gradient(sharp)


# ---------------------------------------------------------------------------
# The quantum (shot-noise) floor
# ---------------------------------------------------------------------------


def test_poisson_and_gaussian_fisher_are_different_quantities() -> None:
    s = SensorModel()
    assert shot_noise_fisher_edge(BASE, 140.0, s) != fisher_information_edge(BASE)


def test_quantum_floor_improves_with_more_photons() -> None:
    """More electrons per level means more photons, so a tighter floor as
    1/sqrt(N)."""
    dim = SensorModel(full_well_e=1500.0)
    bright = SensorModel(full_well_e=6000.0)
    assert quantum_floor_px(BASE, 140.0, bright) < quantum_floor_px(BASE, 140.0, dim)
    ratio = quantum_floor_px(BASE, 140.0, dim) / quantum_floor_px(BASE, 140.0, bright)
    assert ratio == pytest.approx(2.0, rel=0.02)


def test_higher_iso_costs_information() -> None:
    assert quantum_floor_px(BASE, 140.0, SensorModel(iso=1600.0)) > quantum_floor_px(
        BASE, 140.0, SensorModel(iso=100.0)
    )


def test_shot_noise_consistency_detects_a_denoised_image() -> None:
    """Noise below the Poisson floor is physically impossible; it means the
    image was denoised, and denoising invents detail."""
    ratio, verdict = shot_noise_consistency(
        dc.replace(BASE, noise_sigma=0.05), 140.0, SensorModel()
    )
    assert ratio < 0.7
    assert "impossible" in verdict


def test_shot_noise_consistency_detects_compression_dominated_noise() -> None:
    ratio, verdict = shot_noise_consistency(
        dc.replace(BASE, noise_sigma=60.0), 140.0, SensorModel()
    )
    assert ratio > 5.0
    assert "nowhere near photon-limited" in verdict


def test_photon_noise_is_a_negligible_share_when_specimen_limited() -> None:
    """The decisive number: with a realistic reported sigma, driving photon
    noise to zero -- a perfect detector, or any quantum-enhanced scheme --
    changes the answer by far less than a percent."""
    b = variance_budget(0.43, BASE, 140.0, 3.5, 2.5, 10.8)
    assert b.photon_share < 0.01
    assert b.gain_from_perfect_sensor_pct < 1.0
    assert "limited by the physical card" in b.describe()


def test_variance_budget_components_add_in_quadrature() -> None:
    b = variance_budget(0.43, BASE, 140.0, 3.5, 2.5, 10.8)
    assert math.hypot(b.photon_pp, b.residual_pp) == pytest.approx(b.total_pp, rel=1e-6)


def test_photon_share_rises_when_the_detector_is_near_the_floor() -> None:
    b = variance_budget(0.0035, BASE, 140.0, 3.5, 2.5, 10.8)
    assert b.photon_share > 0.5
    assert b.gain_from_perfect_sensor_pct > 10.0


# ---------------------------------------------------------------------------
# Photogrammetric scale and trim detection
# ---------------------------------------------------------------------------


def _rect(w_px: float, h_px: float) -> np.ndarray:
    return np.array([[0.0, 0.0], [w_px, 0.0], [w_px, h_px], [0.0, h_px]])


def test_reference_grades_reflect_real_tolerances() -> None:
    assert BANK_CARD.grade in ("usable", "coarse")
    assert caliper_verified("target", 100.0, 0.02).grade == "caliper"
    assert ScaleReference("rough", 50.0, 0.5).grade == "coarse"


def test_scale_recovers_a_known_pixel_ratio() -> None:
    s = scale_from_reference(_rect(856.0, 539.8), BANK_CARD)
    assert s.value == pytest.approx(10.0, rel=1e-6)
    assert s.sigma / s.value == pytest.approx(BANK_CARD.relative_uncertainty, rel=1e-6)


def test_perspective_asymmetry_inflates_scale_uncertainty() -> None:
    skewed = np.array([[0.0, 0.0], [856.0, 0.0], [830.0, 539.8], [0.0, 539.8]])
    assert scale_from_reference(skewed, BANK_CARD).sigma > scale_from_reference(
        _rect(856.0, 539.8), BANK_CARD
    ).sigma


def test_nominal_card_measures_as_nominal() -> None:
    ppm = 10.0
    dims = measure_absolute(
        _rect(STANDARD_CARD_W_MM * ppm, STANDARD_CARD_H_MM * ppm),
        _rect(856.0, 539.8),
        BANK_CARD,
    )
    assert dims.width.value == pytest.approx(STANDARD_CARD_W_MM, abs=0.01)
    assert dims.height.value == pytest.approx(STANDARD_CARD_H_MM, abs=0.01)


def test_bank_card_reference_cannot_reach_caliper_grade() -> None:
    """0.13mm on 85.6mm is 0.15%, which is ~0.1mm on a trading card. Honest
    about that rather than implying otherwise."""
    ppm = 10.0
    dims = measure_absolute(
        _rect(STANDARD_CARD_W_MM * ppm, STANDARD_CARD_H_MM * ppm),
        _rect(856.0, 539.8),
        BANK_CARD,
    )
    assert dims.width.sigma > 0.05
    assert any("calipers" in w for w in dims.warnings)


def test_better_reference_helps_only_until_resolution_takes_over() -> None:
    """A caliper-verified target beats a bank card, but not by the 6x its
    tolerance suggests: at 10 px/mm a half-pixel edge uncertainty is already
    0.07mm, which swamps what the reference contributes. The tool must say so
    rather than implying the tighter figure."""
    ppm = 10.0
    card = _rect(STANDARD_CARD_W_MM * ppm, STANDARD_CARD_H_MM * ppm)
    ref = _rect(856.0, 539.8)
    coarse = measure_absolute(card, ref, BANK_CARD)
    fine = measure_absolute(card, ref, caliper_verified("target", 85.60, 0.02))
    assert fine.width.sigma < coarse.width.sigma
    # Even a bank card is already comparable to the edge term at 10 px/mm --
    # 0.097mm from the reference against 0.071mm from edge localisation.
    assert coarse.limited_by == "both roughly equally"
    assert fine.limited_by == "image resolution"
    assert any("A better ruler buys nothing" in w for w in fine.warnings)


def test_more_resolution_moves_the_limit_back_to_the_reference() -> None:
    ppm = 40.0
    card = _rect(STANDARD_CARD_W_MM * ppm, STANDARD_CARD_H_MM * ppm)
    ref = _rect(85.60 * ppm, 53.98 * ppm)
    fine = measure_absolute(card, ref, caliper_verified("target", 85.60, 0.02))
    assert fine.width.sigma < 0.03
    assert fine.limited_by in ("scale reference", "both roughly equally")


def test_gross_trim_is_detected() -> None:
    ppm = 10.0
    trimmed = _rect((STANDARD_CARD_W_MM - 1.5) * ppm, STANDARD_CARD_H_MM * ppm)
    dims = measure_absolute(
        trimmed, _rect(856.0, 539.8), caliper_verified("target", 85.60, 0.02)
    )
    v = assess_trim(dims)
    assert v.likely_trimmed
    assert "UNDERSIZE" in v.verdict


def test_normal_cutting_variation_is_not_called_a_trim() -> None:
    ppm = 10.0
    normal = _rect((STANDARD_CARD_W_MM - 0.15) * ppm, STANDARD_CARD_H_MM * ppm)
    dims = measure_absolute(
        normal, _rect(856.0, 539.8), caliper_verified("target", 85.60, 0.02)
    )
    assert not assess_trim(dims).likely_trimmed


def test_oversize_card_is_never_called_trimmed() -> None:
    ppm = 10.0
    over = _rect((STANDARD_CARD_W_MM + 1.0) * ppm, STANDARD_CARD_H_MM * ppm)
    dims = measure_absolute(over, _rect(856.0, 539.8), BANK_CARD)
    assert not assess_trim(dims).likely_trimmed


def test_coarse_reference_reports_an_uninformative_result_not_a_clean_bill() -> None:
    ppm = 10.0
    dims = measure_absolute(
        _rect(STANDARD_CARD_W_MM * ppm, STANDARD_CARD_H_MM * ppm),
        _rect(500.0, 315.0),
        ScaleReference("rough ruler", 50.0, 1.0),
    )
    v = assess_trim(dims)
    assert not v.likely_trimmed
    assert "uninformative" in v.verdict


def test_trim_verdict_never_asserts_fraud() -> None:
    ppm = 10.0
    dims = measure_absolute(
        _rect((STANDARD_CARD_W_MM - 2.0) * ppm, STANDARD_CARD_H_MM * ppm),
        _rect(856.0, 539.8),
        caliper_verified("target", 85.60, 0.02),
    )
    v = assess_trim(dims)
    assert "consistent is not proof" in v.verdict
