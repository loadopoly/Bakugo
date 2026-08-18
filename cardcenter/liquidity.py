"""Flip period from trading volume.

I was wrong to file this under "not a software problem". Sold-comp timestamps
are volumetric data, and time-to-sale follows from them directly. What is not
computable is a *guaranteed* instant buyer at a known price -- that is a
market-making function requiring capital and inventory risk. Those are different
claims and the distinction is the whole content of this module:

    COMPUTABLE      a distribution over days-to-sale at a given asking price,
                    with an honest confidence interval, and the holding-cost
                    discount that follows from it
    NOT COMPUTABLE  "you can flip this immediately for $X"

THE MODEL
---------
Sales of a given card at a given grade arrive as a Poisson process with rate
lambda. Observing n sales over a window T gives the Gamma posterior

    lambda | data ~ Gamma(n + 1/2, T)          (Jeffreys prior)

and time-to-next-sale is Exponential(lambda), so the marginal wait is
Lomax-distributed with

    P(sold within t) = 1 - (1 + t/T)^-(n + 1/2)

That form matters more than it looks. With n=2 sales the naive point estimate
1/lambda = T/2 is wildly overconfident; the Lomax tail is heavy and honest about
it. The interval this produces is wide for thin markets because thin markets ARE
uncertain, not because the estimator is bad.

PRICE ELASTICITY IS THE WEAK LINK
----------------------------------
Listing below market clears faster. The rate is modelled as

    lambda(p) = lambda_0 * (median_price / p) ** elasticity

with elasticity defaulting to 2.0 for collectibles. This default is an
ASSUMPTION, not a measurement. It is fitted from the comp data when there are
enough sales spread across enough prices, and flagged as assumed when not. It is
the least trustworthy number in this file and everything downstream inherits
that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Sequence

import numpy as np

# Below this many observed sales, a point estimate of the flip period is not
# meaningful and the module says so instead of producing one.
THIN_MARKET_SALES = 5


@dataclass(frozen=True)
class Sale:
    price: float
    sold_at: datetime
    grade: Optional[str] = None


@dataclass
class SalesHistory:
    """Observed sold comparables for one card at one grade."""

    sales: list[Sale]
    window_days: Optional[float] = None
    label: str = ""

    def __post_init__(self) -> None:
        self.sales = sorted(self.sales, key=lambda s: s.sold_at)

    @property
    def n(self) -> int:
        return len(self.sales)

    @property
    def observed_days(self) -> float:
        """Length of the observation window.

        Defaults to the span between first and last sale, which UNDERSTATES the
        true window: you did not start watching at the first sale. That bias
        inflates the estimated rate and shortens the predicted flip. If you know
        how long you actually observed, pass window_days.
        """
        if self.window_days is not None:
            return max(1.0, float(self.window_days))
        if self.n < 2:
            return 1.0
        span = (self.sales[-1].sold_at - self.sales[0].sold_at).total_seconds() / 86400.0
        return max(1.0, span)

    @property
    def median_price(self) -> float:
        if not self.sales:
            raise ValueError("no sales")
        return float(np.median([s.price for s in self.sales]))

    @property
    def prices(self) -> np.ndarray:
        return np.array([s.price for s in self.sales])

    @property
    def sales_per_month(self) -> float:
        return self.n / self.observed_days * 30.0

    def fit_elasticity(self) -> tuple[float, bool]:
        """Estimate price elasticity of clearing rate, or admit we cannot.

        Returns (elasticity, was_fitted). Splits the comps at the median price
        and compares how quickly each half arrived. Needs enough sales on both
        sides and enough price spread to mean anything; otherwise falls back to
        the documented default.
        """
        if self.n < 8:
            return 2.0, False
        p = self.prices
        med = float(np.median(p))
        lo = [s for s in self.sales if s.price <= med]
        hi = [s for s in self.sales if s.price > med]
        if len(lo) < 3 or len(hi) < 3:
            return 2.0, False
        spread = float(np.max(p) / max(np.min(p), 1e-9))
        if spread < 1.5:
            return 2.0, False  # no lever arm; any fit is noise

        def rate(group: list[Sale]) -> float:
            if len(group) < 2:
                return len(group) / self.observed_days
            span = (group[-1].sold_at - group[0].sold_at).total_seconds() / 86400.0
            return len(group) / max(1.0, span)

        r_lo, r_hi = rate(lo), rate(hi)
        p_lo = float(np.median([s.price for s in lo]))
        p_hi = float(np.median([s.price for s in hi]))
        if r_lo <= 0 or r_hi <= 0 or p_hi <= p_lo:
            return 2.0, False
        eps = math.log(r_lo / r_hi) / math.log(p_hi / p_lo)
        # Clamp: a fit from a dozen sales should not produce an exotic exponent.
        return float(min(6.0, max(0.0, eps))), True


@dataclass(frozen=True)
class FlipEstimate:
    ask_price: float
    median_market: float
    expected_days: Optional[float]
    days_ci: Optional[tuple[float, float]]
    p_sold_30d: float
    p_sold_90d: float
    sales_per_month: float
    n_sales: int
    elasticity: float
    elasticity_fitted: bool
    thin_market: bool
    holding_discount: float
    warnings: tuple[str, ...]

    def describe(self) -> str:
        lines = [f"ask ${self.ask_price:,.2f} vs median ${self.median_market:,.2f}"]
        if self.thin_market:
            lines.append(
                f"  THIN MARKET: {self.n_sales} recorded sale(s). No flip period "
                "is reported because none would mean anything."
            )
        else:
            lines.append(
                f"  expected days to sell : {self.expected_days:.0f}"
                + (
                    f"  (80% CI {self.days_ci[0]:.0f}-{self.days_ci[1]:.0f})"
                    if self.days_ci
                    else ""
                )
            )
        lines += [
            f"  P(sold within 30 days): {self.p_sold_30d * 100:.0f}%",
            f"  P(sold within 90 days): {self.p_sold_90d * 100:.0f}%",
            f"  observed volume       : {self.sales_per_month:.1f} sales/month "
            f"from {self.n_sales} comp(s)",
            f"  holding discount      : {(1 - self.holding_discount) * 100:.1f}%",
            f"  price elasticity      : {self.elasticity:.1f} "
            + ("(fitted)" if self.elasticity_fitted else "(ASSUMED, not measured)"),
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def p_sold_within(days: float, n: int, window_days: float, rate_multiplier: float = 1.0) -> float:
    """Lomax CDF: P(sold within `days`), marginalising over the unknown rate.

    Uses the Gamma(n + 1/2, T) posterior rather than a point estimate, so a
    two-sale history produces a wide, heavy-tailed answer instead of a confident
    one.
    """
    shape = n + 0.5
    effective = max(1e-9, window_days / max(rate_multiplier, 1e-9))
    return 1.0 - (1.0 + days / effective) ** (-shape)


def _quantile_days(q: float, n: int, window_days: float, rate_multiplier: float) -> float:
    shape = n + 0.5
    effective = max(1e-9, window_days / max(rate_multiplier, 1e-9))
    return effective * ((1.0 - q) ** (-1.0 / shape) - 1.0)


def estimate_flip(
    history: SalesHistory,
    ask_price: Optional[float] = None,
    monthly_holding_cost_frac: float = 0.01,
    elasticity: Optional[float] = None,
) -> FlipEstimate:
    """Time-to-sale and the holding discount that follows from it."""
    if history.n == 0:
        raise ValueError("no sales history; nothing to estimate from")

    median = history.median_price
    ask = float(ask_price) if ask_price is not None else median

    if elasticity is None:
        eps, fitted = history.fit_elasticity()
    else:
        eps, fitted = float(elasticity), False

    # Asking above the median slows the sale; below speeds it.
    rate_multiplier = (median / max(ask, 1e-6)) ** eps
    rate_multiplier = float(min(20.0, max(0.05, rate_multiplier)))

    T = history.observed_days
    n = history.n
    thin = n < THIN_MARKET_SALES

    p30 = p_sold_within(30.0, n, T, rate_multiplier)
    p90 = p_sold_within(90.0, n, T, rate_multiplier)

    expected_days: Optional[float] = None
    ci: Optional[tuple[float, float]] = None
    if not thin:
        # Lomax mean exists only for shape > 1, which n>=5 guarantees.
        shape = n + 0.5
        effective = T / rate_multiplier
        expected_days = effective / (shape - 1.0)
        ci = (
            _quantile_days(0.10, n, T, rate_multiplier),
            _quantile_days(0.90, n, T, rate_multiplier),
        )

    horizon_months = (expected_days or 90.0) / 30.0
    holding_discount = 1.0 / ((1.0 + monthly_holding_cost_frac) ** horizon_months)

    warnings: list[str] = []
    if thin:
        warnings.append(
            f"only {n} recorded sale(s). Any flip period from this is a guess "
            "dressed as a number; treat the card as illiquid."
        )
    if not fitted and elasticity is None:
        warnings.append(
            f"price elasticity {eps:.1f} is the documented default, not fitted "
            "from your comps. It is the least reliable input here."
        )
    if history.window_days is None and n >= 2:
        warnings.append(
            "observation window inferred from first-to-last sale, which "
            "understates it and therefore overstates how fast this sells. Pass "
            "window_days for an unbiased rate."
        )
    if expected_days is not None and expected_days > 180:
        warnings.append(
            f"expected holding period is {expected_days / 30:.0f} months. Capital "
            "tied up that long is a real cost, not a rounding error."
        )
    if ask > median * 1.25:
        warnings.append(
            f"asking {ask / median:.0%} of median materially slows the sale under "
            "any elasticity assumption"
        )

    return FlipEstimate(
        ask_price=ask,
        median_market=median,
        expected_days=expected_days,
        days_ci=ci,
        p_sold_30d=p30,
        p_sold_90d=p90,
        sales_per_month=history.sales_per_month,
        n_sales=n,
        elasticity=eps,
        elasticity_fitted=fitted,
        thin_market=thin,
        holding_discount=holding_discount,
        warnings=tuple(warnings),
    )


def liquidity_adjusted_value(
    gross: float, flip: FlipEstimate, marketplace_fee_frac: float = 0.13
) -> float:
    """Gross resale value discounted for fees and for the wait to realise it."""
    return gross * (1.0 - marketplace_fee_frac) * flip.holding_discount


def sales_from_rows(rows: Sequence[dict], grade: Optional[str] = None) -> SalesHistory:
    """Build a history from CSV-ish rows with price and sold_date columns."""
    out: list[Sale] = []
    for r in rows:
        if grade is not None and r.get("grade") != grade:
            continue
        raw = r.get("sold_date")
        if not raw:
            continue  # an asking price is not a sale
        try:
            when = datetime.fromisoformat(str(raw))
            price = float(r["price"])
        except (ValueError, KeyError, TypeError):
            continue
        out.append(Sale(price=price, sold_at=when, grade=r.get("grade")))
    return SalesHistory(sales=out, label=grade or "all grades")
