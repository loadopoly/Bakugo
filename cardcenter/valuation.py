"""Turning a centering ceiling into an offer -- and refusing to when we can't.

WHAT A CEILING IS AND IS NOT
----------------------------
Centering gives an upper bound on the grade. It says nothing about where under
that bound the card lands, because corners, edges and surface decide that and
this tool cannot see them. So expected value is NOT price(ceiling). It is

    E[value] = sum over g of  P(grade = g) * price(g)      for g <= ceiling

and the P(g) has to come from somewhere real. There are exactly three honest
sources: the population report for that specific card, your own submission
history, or an explicit assumption you are willing to defend. This module will
not invent one. ``GradePrior.uninformative()`` exists, is uniform, is almost
certainly wrong, and says so in every result it touches.

WHY EXPECTED VALUE IS THE WRONG HEADLINE NUMBER
-----------------------------------------------
At a counter you are making a bet with a fat left tail: the card grades two
below the ceiling and the spread between grades is often 5-20x. A positive
expected value with a 40% chance of loss is a bad trade to make forty times in
an afternoon. So the headline outputs here are the loss probability and the
downside percentile, with expected value alongside rather than in front.

PRICES
------
There is no built-in price data and none is fabricated. Configure a real source
or the valuation refuses. The eBay source below hits the genuine Browse API
endpoint (api.ebay.com/buy/browse/v1) and needs a real OAuth token; note that
*sold* comparables come from the Marketplace Insights API, which is
restricted-access and which you must be approved for. Current asking prices are
not sold prices, and treating them as such inflates every valuation.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np

from .grading import GradeBand, load_standards
from .types import Measured


class PricingUnavailable(RuntimeError):
    """No trustworthy price data. Raised instead of guessing."""


# ---------------------------------------------------------------------------
# Grade priors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GradePrior:
    """P(grade) for a card of this type, before centering is considered."""

    weights: dict[str, float]
    source: str
    trustworthy: bool = True

    @staticmethod
    def from_population(counts: dict[str, int], source: str) -> "GradePrior":
        """Build from a population report. This is the good path.

        Population reports are published per card by PSA and BGS and are the
        only widely available empirical answer to 'what do cards like this
        actually grade'. They are biased -- people submit cards they think will
        grade well -- but that bias points the same way as your own selection at
        a counter, so it is closer to right than anything else on offer.
        """
        total = sum(counts.values())
        if total <= 0:
            raise ValueError("population counts are empty")
        return GradePrior(
            weights={k: v / total for k, v in counts.items()},
            source=source,
            trustworthy=True,
        )

    @staticmethod
    def uninformative(grades: Sequence[str]) -> "GradePrior":
        """Uniform over grades. A placeholder, not an estimate.

        Marked untrustworthy so every downstream result carries the warning.
        Real grade distributions are nothing like uniform.
        """
        w = 1.0 / len(grades)
        return GradePrior(
            weights={g: w for g in grades},
            source="uniform placeholder (NOT a real distribution)",
            trustworthy=False,
        )

    def truncated_at(self, ceiling_grade: str, grader: str) -> dict[str, float]:
        """Renormalise over grades at or below the centering ceiling."""
        order = [t["grade"] for t in load_standards()["graders"][grader]["tiers"]]
        if ceiling_grade not in order:
            raise KeyError(f"{ceiling_grade!r} is not a {grader} grade")
        allowed = order[order.index(ceiling_grade) :]
        kept = {g: self.weights.get(g, 0.0) for g in allowed}
        total = sum(kept.values())
        if total <= 0:
            raise PricingUnavailable(
                f"the prior has no mass at or below the {grader} {ceiling_grade} "
                "centering ceiling, so no expected value can be computed"
            )
        return {g: v / total for g, v in kept.items()}


# ---------------------------------------------------------------------------
# Price sources
# ---------------------------------------------------------------------------


class PriceSource(Protocol):
    def prices_by_grade(self, card_key: str, grader: str) -> dict[str, float]: ...
    @property
    def description(self) -> str: ...


@dataclass
class ManualPriceSource:
    """Prices you looked up yourself. Honest, offline, and usually the fastest
    thing to actually get working at a counter."""

    table: dict[str, dict[str, float]]
    note: str = "manually entered comparables"

    def prices_by_grade(self, card_key: str, grader: str) -> dict[str, float]:
        key = f"{card_key}|{grader}"
        if key in self.table:
            return dict(self.table[key])
        if card_key in self.table:
            return dict(self.table[card_key])
        raise PricingUnavailable(
            f"no comparables entered for {card_key!r} at {grader}. "
            "Add them, or do not make an offer on this card."
        )

    @property
    def description(self) -> str:
        return self.note


@dataclass
class CsvPriceSource:
    """Load sold comparables you exported yourself.

    Expected columns: card_key, grader, grade, price, sold_date.
    Rows without a sold_date are rejected: an asking price is not a sale, and
    quietly averaging the two is how a valuation ends up 30% high.
    """

    path: str
    _table: dict[str, dict[str, list[float]]] = field(default_factory=dict, init=False)
    _rejected: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        with open(self.path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if not row.get("sold_date"):
                    self._rejected += 1
                    continue
                key = f"{row['card_key']}|{row['grader']}"
                try:
                    price = float(row["price"])
                except (TypeError, ValueError):
                    self._rejected += 1
                    continue
                self._table.setdefault(key, {}).setdefault(row["grade"], []).append(price)

    def prices_by_grade(self, card_key: str, grader: str) -> dict[str, float]:
        key = f"{card_key}|{grader}"
        if key not in self._table:
            raise PricingUnavailable(f"no sold comparables in {self.path} for {key}")
        # Median, not mean: comp sets are small and contain the occasional
        # nonsense sale, and one outlier moves a mean of five sales a long way.
        return {g: float(np.median(v)) for g, v in self._table[key].items()}

    @property
    def description(self) -> str:
        extra = f", {self._rejected} rows rejected (no sold date)" if self._rejected else ""
        return f"sold comparables from {self.path}{extra}"


@dataclass
class EbayBrowsePriceSource:
    """Real eBay Browse API. Requires a real OAuth token.

    The correct host is api.ebay.com and the path is /buy/browse/v1/... . Note
    that Browse returns ACTIVE listings, which are asking prices. Sold
    comparables come from the Marketplace Insights API, which is
    restricted-access. This class refuses rather than substituting asking prices
    for sold prices, because that substitution silently inflates every offer.
    """

    oauth_token: Optional[str] = None
    base_url: str = "https://api.ebay.com/buy/browse/v1"
    allow_asking_prices: bool = False

    def prices_by_grade(self, card_key: str, grader: str) -> dict[str, float]:
        if not self.oauth_token:
            raise PricingUnavailable(
                "no eBay OAuth token configured. Get one from the eBay developer "
                "program; there is no offline fallback and no default price."
            )
        if not self.allow_asking_prices:
            raise PricingUnavailable(
                "eBay Browse returns active listings (asking prices), not sold "
                "prices. Sold comparables require the Marketplace Insights API, "
                "which needs separate approval. Set allow_asking_prices=True only "
                "if you understand that asking prices run above realised prices."
            )
        raise PricingUnavailable(
            "live eBay querying is not implemented here. Wire it to your approved "
            "credentials, or export sold comps and use CsvPriceSource."
        )

    @property
    def description(self) -> str:
        return "eBay Browse API (requires credentials)"


# ---------------------------------------------------------------------------
# Offer maths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DealCosts:
    """Everything between the price on the slab and the money in your hand."""

    grading_fee: float = 0.0
    shipping_each_way: float = 0.0
    marketplace_fee_frac: float = 0.13
    payment_fee_frac: float = 0.03
    grade_wait_months: float = 2.0
    monthly_discount_rate: float = 0.01

    def net_of_sale(self, gross: float) -> float:
        return gross * (1.0 - self.marketplace_fee_frac - self.payment_fee_frac)

    @property
    def submission_cost(self) -> float:
        return self.grading_fee + 2 * self.shipping_each_way

    @property
    def time_discount(self) -> float:
        return 1.0 / ((1.0 + self.monthly_discount_rate) ** self.grade_wait_months)


@dataclass(frozen=True)
class OfferAnalysis:
    card_key: str
    grader: str
    ceiling_grade: str
    grade_probabilities: dict[str, float]
    prices: dict[str, float]
    expected_net: float
    p10_net: float
    p50_net: float
    suggested_offer: float
    loss_probability: float
    expected_profit: float
    prior_source: str
    price_source: str
    warnings: tuple[str, ...]

    def describe(self) -> str:
        lines = [
            f"{self.card_key}  ({self.grader}, centering ceiling {self.ceiling_grade})",
            "",
            f"  suggested offer     ${self.suggested_offer:,.2f}",
            f"  expected profit     ${self.expected_profit:,.2f}",
            f"  probability of loss {self.loss_probability * 100:.0f}%",
            "",
            f"  net if median grade  ${self.p50_net:,.2f}",
            f"  net at 10th pct      ${self.p10_net:,.2f}   <- the bad case",
            f"  expected net         ${self.expected_net:,.2f}",
            "",
            "  grade distribution used:",
        ]
        for g, p in sorted(self.grade_probabilities.items(), key=lambda kv: -kv[1]):
            if p > 0.005:
                price = self.prices.get(g)
                px = f"${price:,.0f}" if price is not None else "no comp"
                lines.append(f"    {g:<5} {p * 100:5.1f}%   {px}")
        lines += [
            "",
            f"  prior : {self.prior_source}",
            f"  prices: {self.price_source}",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def analyse_offer(
    card_key: str,
    band: GradeBand,
    prior: GradePrior,
    price_source: PriceSource,
    costs: Optional[DealCosts] = None,
    target_margin: float = 0.35,
    submit_for_grading: bool = True,
) -> OfferAnalysis:
    """Work out what a card is worth to you, and how likely you are to lose.

    The ceiling used is the PESSIMISTIC end of the centering band. If our
    measurement and the published standards together allow anything from a 9 to
    a 10, offering as though it were a 10 is buying our own uncertainty at full
    price. Assume the 9.
    """
    costs = costs or DealCosts()
    warnings: list[str] = []

    ceiling = band.worst  # pessimistic end of the band
    probs = prior.truncated_at(ceiling, band.grader)
    prices = price_source.prices_by_grade(card_key, band.grader)

    missing = [g for g, p in probs.items() if p > 0.01 and g not in prices]
    if missing:
        warnings.append(
            f"no comparable price for grade(s) {', '.join(missing)} carrying "
            f"{sum(probs[g] for g in missing) * 100:.0f}% of the probability mass; "
            "those outcomes are valued at zero, so this is a floor, not an estimate"
        )
    if not prior.trustworthy:
        warnings.append(
            "the grade distribution is a uniform placeholder, not real data. "
            "Every number below inherits that. Pull the population report."
        )
    if band.grader_confidence == "low":
        warnings.append(
            f"{band.grader} centering thresholds are weakly sourced, so the "
            "ceiling itself is uncertain"
        )

    outcomes: list[tuple[float, float]] = []
    for g, p in probs.items():
        gross = prices.get(g, 0.0)
        net = costs.net_of_sale(gross)
        if submit_for_grading:
            net = net * costs.time_discount - costs.submission_cost
        outcomes.append((p, net))
    outcomes.sort(key=lambda x: x[1])

    expected_net = sum(p * v for p, v in outcomes)

    def percentile(q: float) -> float:
        acc = 0.0
        for p, v in outcomes:
            acc += p
            if acc >= q:
                return v
        return outcomes[-1][1]

    p10 = percentile(0.10)
    p50 = percentile(0.50)

    suggested = max(0.0, expected_net * (1.0 - target_margin))
    loss_prob = sum(p for p, v in outcomes if v < suggested)
    expected_profit = expected_net - suggested

    if loss_prob > 0.35:
        warnings.append(
            f"{loss_prob * 100:.0f}% of the grade distribution lands below this "
            "offer. Positive expected value with a tail this fat is a bad bet to "
            "repeat across a case."
        )
    if p10 < 0:
        warnings.append(
            "the tenth-percentile outcome is a net loss even before the purchase "
            "price -- grading fees exceed the value of a low-grade result"
        )

    return OfferAnalysis(
        card_key=card_key,
        grader=band.grader,
        ceiling_grade=ceiling,
        grade_probabilities=probs,
        prices=prices,
        expected_net=expected_net,
        p10_net=p10,
        p50_net=p50,
        suggested_offer=suggested,
        loss_probability=loss_prob,
        expected_profit=expected_profit,
        prior_source=prior.source,
        price_source=price_source.description,
        warnings=tuple(warnings),
    )
