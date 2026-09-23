"""General price attribution: any source, any scope, no invented prices."""

from decimal import Decimal

import pytest

from cardcenter.pricing import (PriceAttribution, PriceError, attribution_from_sticker_record,
                                parse_price_text, resolve_item_price)


@pytest.mark.parametrize("text,amount,currency,qty", [
    ("$2.50", "2.50", "USD", 1),
    ("2,50 €", "2.50", "EUR", 1),
    ("£3", "3", "GBP", 1),
    ("C$3", "3", "CAD", 1),
    ("USD 4", "4", "USD", 1),
    ("4 USD", "4", "USD", 1),
    ("50c", "0.5", "USD", 1),
    ("25¢ each", "0.25", "USD", 1),
    ("3 for $1", "1", "USD", 3),
    ("10/$5", "5", "USD", 10),
])
def test_parse_price_text(text, amount, currency, qty):
    p = parse_price_text(text)
    assert p == {"amount": Decimal(amount), "currency": currency, "quantity": qty}


@pytest.mark.parametrize("text", ["", "12", "price 7.99", "#182", "HP 70"])
def test_bare_numbers_are_not_prices(text):
    assert parse_price_text(text) is None


def test_validation():
    with pytest.raises(PriceError):
        PriceAttribution("-1")
    with pytest.raises(PriceError):
        PriceAttribution("1", currency="XYZ")
    with pytest.raises(PriceError):
        PriceAttribution("1", source="guess")
    with pytest.raises(PriceError):
        PriceAttribution("1", scope="box")            # needs scope_key
    with pytest.raises(PriceError):
        PriceAttribution("1", quantity=0)


def test_resolution_prefers_specific_scope_then_entered_then_latest():
    sign = PriceAttribution("0.25", source="shelf_sign", scope="box", scope_key="b", observed_at=100)
    ocr = PriceAttribution("1.00", source="sticker", method="ocr", observed_at=300)
    typed = PriceAttribution("0.90", source="tag", observed_at=200)
    assert resolve_item_price([]) is None
    assert resolve_item_price([sign]).amount == Decimal("0.2500")
    r = resolve_item_price([sign, ocr, typed])
    assert r.amount == Decimal("0.9000") and r.attribution.source == "tag"
    later = PriceAttribution("0.80", source="tag", observed_at=400)
    assert resolve_item_price([typed, later]).amount == Decimal("0.8000")


def test_lot_price_split_per_item_and_kind_filter():
    lot = PriceAttribution("1", scope="lot", scope_key="l", quantity=3, source="tag")
    r = resolve_item_price([lot])
    assert r.amount == Decimal("0.3333") and "1 for 3" in r.basis
    paid = PriceAttribution("5", kind="paid", source="receipt")
    assert resolve_item_price([paid]) is None
    assert resolve_item_price([paid], kind="paid").amount == Decimal("5.0000")


def test_roundtrip_and_sticker_adapter():
    a = PriceAttribution("2.5", source="listing", venue="shop")
    assert PriceAttribution.from_dict(a.to_dict()) == a
    s = attribution_from_sticker_record({"price_usd": "1.99", "sticker_tier_label": "A"})
    assert s.source == "sticker" and s.method == "ocr" and s.amount == Decimal("1.99")
    assert attribution_from_sticker_record({"price_usd": None}) is None
