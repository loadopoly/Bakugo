"""Tests for identification and liquidity.

The theme in both: the module must refuse to collapse a genuine unknown into a
confident number. An ambiguous printing must stay ambiguous; a two-sale history
must not yield a flip period.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from cardcenter.catalog import (
    MIN_PX_TO_READ,
    CatalogEntry,
    CatalogUnavailable,
    LocalCatalog,
    distinguishing_features,
    identify,
    match_score,
    orb_signature,
    resolvable_features,
)
from cardcenter.liquidity import (
    THIN_MARKET_SALES,
    Sale,
    SalesHistory,
    estimate_flip,
    liquidity_adjusted_value,
    p_sold_within,
    sales_from_rows,
)
from cardcenter.synth import render_capture


def _entry(num: str, setname: str, price: float, **attrs) -> CatalogEntry:
    return CatalogEntry(
        card_id=f"{setname}-{num}",
        name="Test Card",
        set_name=setname,
        collector_number=num,
        prices={"usd": price},
        attributes=attrs,
    )


# --------------------------------------------------------------------------
# Resolution gating
# --------------------------------------------------------------------------


def test_collector_number_unreadable_at_arms_length() -> None:
    """1.5mm of text at 4 px/mm is 6 pixels, where measured accuracy is 17-42%
    AND the errors are confident-wrong. Leaning on the glass at 8 px/mm gives
    12 px, which measurement puts at the usable floor."""
    assert not resolvable_features(4.0)["collector_number"]
    assert resolvable_features(8.0)["collector_number"]  # 12px, measured floor
    assert resolvable_features(18.0)["collector_number"]


def test_whole_card_features_resolve_at_any_usable_scale() -> None:
    for ppm in (4.0, 8.0, 18.0):
        r = resolvable_features(ppm)
        assert r["border_color"]
        assert r["full_art"]


def test_finish_is_never_treated_as_resolvable() -> None:
    """Foil is a specular property, and through glass it is indistinguishable
    from glare. No resolution makes it readable from one frame."""
    for ppm in (4.0, 18.0, 100.0):
        assert not resolvable_features(ppm)["finish"]


def test_threshold_is_where_it_claims_to_be() -> None:
    from cardcenter.catalog import FEATURE_SIZE_MM

    size = FEATURE_SIZE_MM["collector_number"]
    just_under = (MIN_PX_TO_READ / size) * 0.99
    just_over = (MIN_PX_TO_READ / size) * 1.01
    assert not resolvable_features(just_under)["collector_number"]
    assert resolvable_features(just_over)["collector_number"]


# --------------------------------------------------------------------------
# Variant ambiguity
# --------------------------------------------------------------------------


def test_distinguishing_features_detects_collector_number_difference() -> None:
    entries = [_entry("1", "Alpha", 100.0), _entry("2", "Alpha", 5.0)]
    assert "collector_number" in distinguishing_features(entries)


def test_distinguishing_features_detects_finish_difference() -> None:
    entries = [
        _entry("1", "Alpha", 5.0, finishes=["nonfoil"]),
        _entry("1", "Alpha", 90.0, finishes=["foil"]),
    ]
    assert "finish" in distinguishing_features(entries)


def test_single_printing_has_nothing_to_distinguish() -> None:
    assert distinguishing_features([_entry("1", "Alpha", 5.0)]) == set()


def test_ambiguous_printing_is_not_collapsed_to_one_price() -> None:
    cat = LocalCatalog([_entry("12", "Alpha", 400.0), _entry("77", "Beta", 4.0)])
    ident = identify("Test Card", cat, px_per_mm=6.0)
    assert ident.is_ambiguous
    assert ident.resolved is None
    assert ident.price_spread == (4.0, 400.0)
    assert ident.spread_ratio == pytest.approx(100.0)
    assert "collector_number" in ident.unresolvable_features


def test_large_price_spread_blocks_an_offer() -> None:
    cat = LocalCatalog([_entry("12", "Alpha", 400.0), _entry("77", "Beta", 4.0)])
    ident = identify("Test Card", cat, px_per_mm=6.0)
    assert any("Do not make an offer" in w for w in ident.warnings)


def test_supplying_the_collector_number_resolves_it() -> None:
    """The realistic path at 1x: a human reads the number and types it."""
    cat = LocalCatalog([_entry("12", "Alpha", 400.0), _entry("77", "Beta", 4.0)])
    ident = identify("Test Card", cat, px_per_mm=6.0, known_collector_number="12")
    assert not ident.is_ambiguous
    assert ident.resolved is not None
    assert ident.resolved.best_price == 400.0


def test_single_candidate_resolves_without_help() -> None:
    cat = LocalCatalog([_entry("12", "Alpha", 400.0)])
    ident = identify("Test Card", cat, px_per_mm=4.0)
    assert ident.resolved is not None
    assert not ident.is_ambiguous


def test_foil_difference_always_warns() -> None:
    cat = LocalCatalog(
        [
            _entry("1", "Alpha", 5.0, finishes=["nonfoil"]),
            _entry("1", "Alpha", 90.0, finishes=["foil"]),
        ]
    )
    ident = identify("Test Card", cat, px_per_mm=30.0)
    assert any("foil" in w.lower() for w in ident.warnings)


def test_unknown_card_raises() -> None:
    with pytest.raises(CatalogUnavailable):
        identify("Nonexistent", LocalCatalog([]), px_per_mm=10.0)


# --------------------------------------------------------------------------
# Visual matching
# --------------------------------------------------------------------------


def test_orb_matches_the_same_card_under_different_capture() -> None:
    a, _, _ = render_capture(tilt_deg=0.0, seed=3)
    b, _, _ = render_capture(tilt_deg=14.0, seed=3, vignette=0.3, noise_sigma=4.0)
    inliers, frac = match_score(orb_signature(a), orb_signature(b))
    assert inliers >= 12
    assert frac > 0.3


def test_orb_survives_partial_glare() -> None:
    """Glare covers part of the card; local features must survive occlusion."""
    a, _, _ = render_capture(seed=4)
    b = a.copy()
    b[: b.shape[0] // 3, :] = 252
    inliers, _ = match_score(orb_signature(a), orb_signature(b))
    assert inliers >= 12


def test_orb_rejects_a_different_card() -> None:
    a, _, _ = render_capture(seed=5, art_bgr=(120, 70, 45), border_bgr=(60, 200, 240))
    b, _, _ = render_capture(seed=99, art_bgr=(30, 160, 60), border_bgr=(220, 220, 225))
    inliers_same, _ = match_score(orb_signature(a), orb_signature(a))
    inliers_diff, _ = match_score(orb_signature(a), orb_signature(b))
    assert inliers_diff < inliers_same


# --------------------------------------------------------------------------
# Liquidity
# --------------------------------------------------------------------------


def _history(n: int, days_apart: float = 6.0, price: float = 100.0) -> SalesHistory:
    start = datetime(2026, 1, 1)
    return SalesHistory(
        sales=[
            Sale(price=price, sold_at=start + timedelta(days=i * days_apart))
            for i in range(n)
        ],
        window_days=n * days_apart,
    )


def test_thin_market_refuses_a_flip_period() -> None:
    est = estimate_flip(_history(2))
    assert est.thin_market
    assert est.expected_days is None
    assert any("illiquid" in w for w in est.warnings)


def test_liquid_market_gives_a_flip_period_with_an_interval() -> None:
    est = estimate_flip(_history(20, days_apart=3.0))
    assert not est.thin_market
    assert est.expected_days is not None
    assert est.days_ci[0] < est.expected_days < est.days_ci[1]


def test_more_volume_means_faster_expected_sale() -> None:
    slow = estimate_flip(_history(6, days_apart=20.0))
    fast = estimate_flip(_history(6, days_apart=2.0))
    assert fast.expected_days < slow.expected_days
    assert fast.p_sold_30d > slow.p_sold_30d


def test_uncertainty_is_wider_with_fewer_sales() -> None:
    """A five-sale history must produce a wider relative interval than a
    fifty-sale one over the same window."""
    few = estimate_flip(SalesHistory(_history(5, 12.0).sales, window_days=60))
    many = estimate_flip(SalesHistory(_history(50, 1.2).sales, window_days=60))
    few_rel = (few.days_ci[1] - few.days_ci[0]) / few.expected_days
    many_rel = (many.days_ci[1] - many.days_ci[0]) / many.expected_days
    assert few_rel > many_rel


def test_asking_above_market_slows_the_sale() -> None:
    h = _history(15, days_apart=4.0, price=100.0)
    at_market = estimate_flip(h, ask_price=100.0)
    above = estimate_flip(h, ask_price=160.0)
    assert above.expected_days > at_market.expected_days
    assert above.p_sold_30d < at_market.p_sold_30d
    assert any("slows the sale" in w for w in above.warnings)


def test_asking_below_market_speeds_the_sale() -> None:
    h = _history(15, days_apart=4.0, price=100.0)
    assert estimate_flip(h, ask_price=70.0).expected_days < estimate_flip(
        h, ask_price=100.0
    ).expected_days


def test_probability_is_monotonic_in_time() -> None:
    n, T = 10, 60.0
    ps = [p_sold_within(d, n, T) for d in (1, 7, 30, 90, 365)]
    assert all(a < b for a, b in zip(ps, ps[1:]))
    assert 0.0 < ps[0] < ps[-1] < 1.0


def test_unfitted_elasticity_is_flagged_as_assumed() -> None:
    est = estimate_flip(_history(6))
    assert not est.elasticity_fitted
    assert any("default, not fitted" in w for w in est.warnings)


def test_inferred_window_is_flagged_as_optimistic() -> None:
    """Span-between-sales understates the true observation window, which makes
    everything look more liquid than it is."""
    start = datetime(2026, 1, 1)
    h = SalesHistory([Sale(100.0, start + timedelta(days=i * 5)) for i in range(8)])
    assert h.window_days is None
    assert any("understates" in w for w in estimate_flip(h).warnings)


def test_long_holding_period_is_called_out() -> None:
    est = estimate_flip(_history(8, days_apart=220.0))
    assert est.expected_days > 180
    assert any("Capital" in w for w in est.warnings)


def test_holding_discount_reduces_realised_value() -> None:
    slow = estimate_flip(_history(8, days_apart=60.0))
    assert slow.holding_discount < 1.0
    assert liquidity_adjusted_value(1000.0, slow) < 1000.0 * (1 - 0.13)


def test_rows_without_a_sold_date_are_dropped() -> None:
    rows = [
        {"price": "100", "sold_date": "2026-01-01", "grade": "9"},
        {"price": "150", "sold_date": "", "grade": "9"},  # an asking price
        {"price": "120", "sold_date": "2026-02-01", "grade": "9"},
    ]
    h = sales_from_rows(rows, grade="9")
    assert h.n == 2


def test_empty_history_raises() -> None:
    with pytest.raises(ValueError):
        estimate_flip(SalesHistory([]))


def test_elasticity_fit_needs_price_spread() -> None:
    """All comps at the same price give no lever arm; any fit would be noise."""
    h = _history(12, days_apart=3.0, price=100.0)
    eps, fitted = h.fit_elasticity()
    assert not fitted
    assert eps == pytest.approx(2.0)
