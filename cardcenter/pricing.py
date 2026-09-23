"""Price attribution: what a card cost, from whatever said so.

A binder price sticker is one way a price reaches an item. Shops and sellers
also use shelf or bin signs ("all singles 25c"), handwritten tags, per-page
prices, lot prices ("3 for $1"), receipts, online listings, or just say it.
This module records any of those as a ``PriceAttribution`` and resolves the
general price for one item from the attributions that cover it.

Resolution rule (``resolve_item_price``):

1. The most specific scope wins: ``item`` > ``page`` > ``box`` > ``lot`` >
   ``venue``. A sticker on the card beats the bin sign above it.
2. Within a scope, a human-entered price beats one read by OCR, and a later
   observation beats an earlier one.
3. A lot price with a known count becomes a per-item price (``3 for $1`` ->
   $0.3333 each) and says so in ``basis``.

No price is invented: with no attribution the result is ``None``.
Amounts are ``Decimal``; currency is an ISO 4217 code.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Iterable, Optional, Sequence

SOURCES = (
    "sticker",        # printed/price-gun sticker on the sleeve, toploader or pocket
    "tag",            # handwritten or printed tag attached to the item
    "shelf_sign",     # sign on a bin, box, case or table
    "page_label",     # one price for a binder page
    "receipt",        # what was paid, from a receipt
    "listing",        # an online or marketplace listing
    "verbal",         # quoted by the seller
    "manual",         # entered by the user with no other source
)
SCOPES = ("item", "page", "box", "lot", "venue")
KINDS = ("asking", "paid", "sold", "estimate")
METHODS = ("entered", "ocr")

_SCOPE_RANK = {s: i for i, s in enumerate(SCOPES)}
_CURRENCY_SYMBOLS = {"$": "USD", "US$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "C$": "CAD", "A$": "AUD"}
_CURRENCY_CODES = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD"}


class PriceError(ValueError):
    pass


def _money(value) -> Decimal:
    try:
        d = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, AttributeError):
        raise PriceError(f"not an amount: {value!r}")
    if d < 0 or not d.is_finite():
        raise PriceError(f"amount must be a non-negative number: {value!r}")
    return d


@dataclass(frozen=True)
class PriceAttribution:
    amount: Decimal
    currency: str = "USD"
    source: str = "manual"
    scope: str = "item"
    kind: str = "asking"
    method: str = "entered"
    quantity: int = 1               # items the amount covers (lot pricing)
    scope_key: str = ""             # which page / box / lot / venue
    venue: str = ""
    observed_at: float = field(default_factory=time.time)
    raw_text: str = ""
    evidence: str = ""              # sha256, scan id, identification id

    def __post_init__(self):
        object.__setattr__(self, "amount", _money(self.amount))
        cur = (self.currency or "").upper()
        if cur not in _CURRENCY_CODES:
            raise PriceError(f"unsupported currency {self.currency!r}")
        object.__setattr__(self, "currency", cur)
        for name, allowed in (("source", SOURCES), ("scope", SCOPES), ("kind", KINDS),
                              ("method", METHODS)):
            if getattr(self, name) not in allowed:
                raise PriceError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if int(self.quantity) < 1:
            raise PriceError("quantity must be >= 1")
        if self.scope != "item" and not self.scope_key:
            raise PriceError(f"a {self.scope}-scope price needs scope_key (which {self.scope})")

    @property
    def per_item(self) -> Decimal:
        return (self.amount / Decimal(int(self.quantity))).quantize(Decimal("0.0001"), ROUND_HALF_UP)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["amount"] = str(self.amount)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PriceAttribution":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class ResolvedPrice:
    amount: Decimal
    currency: str
    basis: str
    attribution: PriceAttribution
    considered: int

    def to_dict(self) -> dict:
        return {"amount": str(self.amount), "currency": self.currency, "basis": self.basis,
                "considered": self.considered, "attribution": self.attribution.to_dict()}


def resolve_item_price(attributions: Iterable[PriceAttribution], kind: str = "asking",
                       currency: Optional[str] = None) -> Optional[ResolvedPrice]:
    cands = [a for a in attributions if a.kind == kind and (currency is None or a.currency == currency.upper())]
    if not cands:
        return None
    best = min(cands, key=lambda a: (_SCOPE_RANK[a.scope], 0 if a.method == "entered" else 1, -a.observed_at))
    basis = f"{best.source} at {best.scope} scope"
    if best.quantity > 1:
        basis += f", {best.amount} for {best.quantity}"
    if best.method == "ocr":
        basis += " (read by OCR)"
    return ResolvedPrice(best.per_item, best.currency, basis, best, len(cands))


# --------------------------------------------------------------------------
# Reading a price from text (sticker OCR, a typed note, a sign)
# --------------------------------------------------------------------------

_LOT_RE = re.compile(
    r"(?P<n>\d{1,3})\s*(?:for|/|@)\s*(?P<cur>US\$|C\$|A\$|[$€£¥])?\s*(?P<amt>\d+(?:[.,]\d{1,2})?)",
    re.I)
_AMT_RE = re.compile(
    r"(?P<cur>US\$|C\$|A\$|[$€£¥])\s*(?P<amt>\d{1,6}(?:[.,]\d{1,2})?)"
    r"|(?P<amt2>\d{1,6}(?:[.,]\d{1,2})?)\s*(?P<code>USD|EUR|GBP|JPY|CAD|AUD|€|£)"
    r"|(?P<code2>USD|EUR|GBP|JPY|CAD|AUD)\s*(?P<amt3>\d{1,6}(?:[.,]\d{1,2})?)"
    r"|(?P<cents>\d{1,2})\s*(?:¢|c\b|cents?\b)",
    re.I)


def _dec(txt: str) -> Decimal:
    t = txt.strip()
    if "," in t and "." not in t and len(t.split(",")[-1]) in (1, 2):
        t = t.replace(",", ".")
    return _money(t)


def parse_price_text(text: str, default_currency: str = "USD") -> Optional[dict]:
    """Return ``{"amount", "currency", "quantity"}`` or None. Refuses rather
    than guessing: a bare number with no currency marker is not a price."""
    if not text:
        return None
    t = " ".join(text.split())
    m = _LOT_RE.search(t)
    if m and (m.group("cur") or "for" in m.group(0).lower()):
        cur = _CURRENCY_SYMBOLS.get(m.group("cur") or "", default_currency)
        return {"amount": _dec(m.group("amt")), "currency": cur, "quantity": int(m.group("n"))}
    m = _AMT_RE.search(t)
    if not m:
        return None
    if m.group("cents"):
        return {"amount": (Decimal(m.group("cents")) / 100), "currency": default_currency, "quantity": 1}
    if m.group("amt"):
        cur = _CURRENCY_SYMBOLS.get(m.group("cur").upper(), default_currency)
        return {"amount": _dec(m.group("amt")), "currency": cur, "quantity": 1}
    code = (m.group("code") or m.group("code2")).upper()
    amt = m.group("amt2") or m.group("amt3")
    return {"amount": _dec(amt), "currency": _CURRENCY_SYMBOLS.get(code, code), "quantity": 1}


def attribution_from_sticker_record(record: dict, evidence: str = "") -> Optional[PriceAttribution]:
    """Adapter for ``ingest.binder_ingest`` pocket records (``price_usd``)."""
    price = record.get("price_usd")
    if not price:
        return None
    return PriceAttribution(amount=price, currency="USD", source="sticker", scope="item",
                            method="ocr", raw_text=str(price), evidence=evidence,
                            venue=str(record.get("sticker_tier_label") or ""))
