"""Tests for online learning.

The load-bearing test is identity reduction: with zero observations the
posterior decoder must reproduce the edit-distance snap it replaces. An adaptive
component you cannot switch off is one you cannot debug, and that contract is
borrowed directly from QUIPU's rADAM.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cardcenter.catalog import MIN_PX_TO_READ, CatalogEntry
from cardcenter.execution import optimal_liquidation
from cardcenter.learning import (
    ConfusionModel,
    EncounterPrior,
    Fill,
    ImpactEstimator,
    LearningStore,
    band_for,
    learning_report,
    posterior_decode,
)
from cardcenter.ocr import levenshtein


def _c(*numbers: str) -> list[CatalogEntry]:
    return [
        CatalogEntry(card_id=f"id-{n}", name="X", set_name="S", collector_number=n)
        for n in numbers
    ]


# --------------------------------------------------------------------------
# Identity reduction -- the contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reading,numbers",
    [
        ("263", ("263", "107", "42")),
        ("2s3", ("263", "107", "42")),
        ("107", ("263", "107", "142")),
        ("41", ("263", "107", "42")),
        ("999", ("263", "107", "42")),
    ],
)
def test_untrained_decoder_matches_edit_distance(reading, numbers) -> None:
    """With no observations, the argmax must be the minimum-distance candidate."""
    cands = _c(*numbers)
    d = posterior_decode(reading, cands, glyph_px=30.0)
    min_distance = min(levenshtein(reading, c.collector_number) for c in cands)
    # Ties are genuine ties; any minimum-distance candidate is correct behaviour.
    assert levenshtein(reading, d.best_number) == min_distance


def test_untrained_decoder_says_it_is_at_the_baseline() -> None:
    d = posterior_decode("263", _c("263", "107"), glyph_px=30.0)
    assert not d.used_learned_confusion
    assert not d.used_learned_prior
    assert any("edit-distance baseline" in w for w in d.warnings)


def test_likelihood_is_monotone_in_distance_when_untrained() -> None:
    m = ConfusionModel()
    base = m.log_likelihood("263", "263", "clean")
    one = m.log_likelihood("263", "203", "clean")
    two = m.log_likelihood("263", "204", "clean")
    assert base > one > two


# --------------------------------------------------------------------------
# The gate survives learning
# --------------------------------------------------------------------------


def test_learning_does_not_reopen_the_resolution_gate() -> None:
    """No amount of training raises the information content of a 6px glyph."""
    conf = ConfusionModel()
    for _ in range(500):
        conf.observe("263", "263", glyph_px=30.0)
    prior = EncounterPrior()
    for _ in range(500):
        prior.observe("id-263")

    d = posterior_decode("263", _c("263", "107"), glyph_px=6.0, confusion=conf, prior=prior)
    assert d.best_id is None
    assert not d.decisive
    assert any("does not raise the information content" in w for w in d.warnings)


def test_below_gate_observations_are_discarded() -> None:
    conf = ConfusionModel()
    conf.observe("263", "203", glyph_px=6.0)
    assert conf.observations() == 0


def test_band_assignment() -> None:
    assert band_for(MIN_PX_TO_READ - 1) == "below_gate"
    assert band_for(15.0) == "marginal"
    assert band_for(25.0) == "good"
    assert band_for(60.0) == "clean"


def test_marginal_band_always_warns() -> None:
    d = posterior_decode("263", _c("263"), glyph_px=15.0)
    assert any("confident wrong printing" in w for w in d.warnings)


# --------------------------------------------------------------------------
# What learning actually buys
# --------------------------------------------------------------------------


def test_learned_confusion_beats_edit_distance_on_a_known_substitution() -> None:
    """If this engine reliably reads 8 as 0 at this scale, a reading of '0'
    should favour the '8' printing -- which edit distance cannot express."""
    conf = ConfusionModel()
    for _ in range(200):
        conf.observe("8", "0", glyph_px=25.0)  # verified: truth 8, engine said 0
    cands = _c("8", "9")
    untrained = posterior_decode("0", cands, 25.0)
    trained = posterior_decode("0", cands, 25.0, confusion=conf)
    assert untrained.posterior["id-8"] == pytest.approx(untrained.posterior["id-9"])
    assert trained.posterior["id-8"] > 0.9


def test_encounter_prior_breaks_a_tie_that_edit_distance_cannot() -> None:
    """'11' is one edit from both '1' and '111'. If '1' is what actually shows
    up, the posterior should say so."""
    prior = EncounterPrior()
    for _ in range(300):
        prior.observe("id-1")
    cands = _c("1", "111")
    untrained = posterior_decode("11", cands, 25.0)
    trained = posterior_decode("11", cands, 25.0, prior=prior)
    assert untrained.posterior["id-1"] == pytest.approx(untrained.posterior["id-111"])
    assert trained.posterior["id-1"] > trained.posterior["id-111"]


def test_prior_can_make_a_decode_less_decisive_not_only_more() -> None:
    """Learning must be able to raise doubt, not only confirm. A reading that
    edit distance resolves uniquely should become uncertain when the prior
    strongly favours a different printing."""
    prior = EncounterPrior()
    for _ in range(4000):
        prior.observe("id-203")
    cands = _c("263", "203")
    untrained = posterior_decode("263", cands, 25.0)
    trained = posterior_decode("263", cands, 25.0, prior=prior)
    assert untrained.posterior["id-263"] > trained.posterior["id-263"]


def test_indecisive_posterior_is_reported_as_unresolved() -> None:
    d = posterior_decode("11", _c("1", "111"), 25.0)
    assert not d.decisive
    assert any("decisiveness threshold" in w for w in d.warnings)


def test_confusion_accuracy_is_reported() -> None:
    conf = ConfusionModel()
    for _ in range(8):
        conf.observe("26", "26", glyph_px=25.0)
    for _ in range(2):
        conf.observe("26", "20", glyph_px=25.0)
    assert conf.accuracy("good") == pytest.approx(18 / 20)


def test_length_mismatch_is_not_recorded_as_substitution() -> None:
    conf = ConfusionModel()
    conf.observe("263", "26", glyph_px=25.0)
    assert conf.observations("good") == 0


# --------------------------------------------------------------------------
# Impact calibration
# --------------------------------------------------------------------------


def _fills(gamma: float, eta: float, n: int, noise: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        cum = float(i)
        rate = 0.5 + 0.35 * (i % 5)
        slip = gamma * cum + eta * rate + rng.normal(0, noise)
        out.append(Fill(100.0, 100.0 - slip, cum, rate))
    return out


def test_too_few_fills_falls_back_to_assumed() -> None:
    est = ImpactEstimator()
    est.observe_many(_fills(0.4, 2.0, 3))
    p = est.estimate(daily_volatility=2.0, median_price=100.0)
    assert not p.calibrated
    assert "at least 4" in p.calibration_note


def test_recovers_known_impact_parameters() -> None:
    est = ImpactEstimator()
    est.observe_many(_fills(0.4, 2.0, 40, noise=0.02, seed=1))
    p = est.estimate(daily_volatility=2.0, median_price=100.0)
    assert p.calibrated
    assert p.permanent_gamma == pytest.approx(0.4, rel=0.15)
    assert p.temporary_eta == pytest.approx(2.0, rel=0.15)


def test_updates_are_order_independent() -> None:
    fills = _fills(0.4, 2.0, 20, noise=0.05, seed=2)
    a, b = ImpactEstimator(), ImpactEstimator()
    a.observe_many(fills)
    b.observe_many(list(reversed(fills)))
    ua, ub = a.uncertainty(), b.uncertainty()
    assert ua["eta"] == pytest.approx(ub["eta"], rel=1e-9)
    assert ua["gamma"] == pytest.approx(ub["gamma"], rel=1e-9)


def test_uncertainty_shrinks_with_more_fills() -> None:
    few, many = ImpactEstimator(), ImpactEstimator()
    few.observe_many(_fills(0.4, 2.0, 6, noise=0.3, seed=3))
    many.observe_many(_fills(0.4, 2.0, 120, noise=0.3, seed=3))
    assert many.uncertainty()["eta_sd"] < few.uncertainty()["eta_sd"]


def test_noisy_data_is_flagged_as_untrustworthy() -> None:
    est = ImpactEstimator()
    est.observe_many(_fills(0.4, 2.0, 8, noise=25.0, seed=4))
    p = est.estimate(daily_volatility=2.0, median_price=100.0)
    assert not p.calibrated


def test_non_physical_fit_falls_back_rather_than_being_used() -> None:
    """Selling that shows no price response must not produce a negative eta
    that would then generate a nonsense schedule."""
    est = ImpactEstimator()
    est.observe_many(_fills(0.0, -1.0, 20, noise=0.01, seed=5))
    p = est.estimate(daily_volatility=2.0, median_price=100.0)
    assert not p.calibrated
    assert "non-physical" in p.calibration_note


def test_calibrated_impact_feeds_a_schedule_without_the_assumed_warning() -> None:
    est = ImpactEstimator()
    est.observe_many(_fills(0.4, 2.0, 60, noise=0.02, seed=6))
    p = est.estimate(daily_volatility=2.0, median_price=100.0)
    s = optimal_liquidation(20, 30.0, p, risk_aversion=1e-3)
    assert s.impact_calibrated
    assert not any("dollar figures are not" in w for w in s.warnings)


# --------------------------------------------------------------------------
# Persistence and reporting
# --------------------------------------------------------------------------


def test_state_round_trips_through_the_store(tmp_path) -> None:
    conf = ConfusionModel()
    conf.observe("263", "203", glyph_px=25.0)
    prior = EncounterPrior()
    prior.observe("id-263")
    est = ImpactEstimator()
    est.observe_many(_fills(0.4, 2.0, 10, noise=0.05, seed=7))

    path = str(tmp_path / "learn.db")
    with LearningStore(path) as s:
        s.save_confusion(conf)
        s.save_prior(prior)
        s.save_impact(est)
    with LearningStore(path) as s:
        c2, p2, e2 = s.load_confusion(), s.load_prior(), s.load_impact()

    assert c2.observations("good") == conf.observations("good")
    assert p2.counts == prior.counts
    assert e2.n == est.n
    assert e2.uncertainty()["eta"] == pytest.approx(est.uncertainty()["eta"], rel=1e-9)


def test_empty_store_returns_baseline_models(tmp_path) -> None:
    with LearningStore(str(tmp_path / "e.db")) as s:
        assert s.load_confusion().observations() == 0
        assert s.load_prior().total_observations == 0
        assert s.load_impact().n == 0


def test_report_states_the_baseline_when_untrained() -> None:
    r = learning_report(ConfusionModel(), EncounterPrior(), ImpactEstimator())
    assert "identical to the edit-distance baseline" in r
    assert "not yet identifiable" in r


def test_report_always_states_the_circularity_rule() -> None:
    r = learning_report(ConfusionModel(), EncounterPrior(), ImpactEstimator())
    assert "ONLY from verified observations" in r
    assert "never fed back" in r or "is ever fed back" in r
