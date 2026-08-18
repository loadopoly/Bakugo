"""Almgren-Chriss optimal liquidation, applied to card inventory.

WHAT THIS IS FOR
----------------
Not single-card flipping. For one card there is no schedule to optimise and
Almgren-Chriss is degenerate. It is for INVENTORY: you bought the case, you have
eleven copies of the same card, and dumping them all this week depresses the
price you get for all eleven while holding them for six months exposes you to
whatever the market does in six months. That tradeoff is exactly what
Almgren-Chriss solves, and it is a real tradeoff in card dealing.

THE MODEL (Almgren & Chriss, "Optimal execution of portfolio transactions", 2000)
---------------------------------------------------------------------------------
Liquidate X units over horizon T in N intervals of length tau. Selling exerts

    permanent impact   g(v) = gamma * v      moves the market for everyone, forever
    temporary impact   h(v) = eta * v        your own slippage this interval only

and the unsold remainder is exposed to volatility sigma. Minimising
E[cost] + lambda * Var[cost] gives the trajectory

    x(t) = X * sinh(kappa (T - t)) / sinh(kappa T)

with kappa fixed by  2(cosh(kappa tau) - 1) / tau^2 = lambda sigma^2 / eta_tilde,
where eta_tilde = eta - gamma tau / 2. The useful summary number is the half-life
1/kappa: sell over roughly that many days and you are near the frontier.

lambda -> 0 gives linear selling (pure cost minimisation, maximum risk).
lambda -> large gives an urgent front-loaded schedule.

WHERE IT BREAKS FOR CARDS, WHICH IS MOST PLACES
------------------------------------------------
Almgren-Chriss assumes a continuously traded asset with a diffusive price and
enough depth that you can sell any quantity in any interval. Card markets
violate all three:

  * Trades arrive as a Poisson process, not a continuum. If your card sells
    0.3 times a week, a schedule that says "sell 2 per day" is not a plan, it is
    arithmetic. ``check_feasibility`` compares the schedule against the observed
    arrival rate from the liquidity module and refuses when the schedule exceeds
    what the market has ever absorbed. This is the check that matters most.
  * Quantities are integers and usually small. X = 3 has essentially no
    interesting trajectory.
  * eta and gamma must be calibrated from data showing how your own selling moved
    the price. Almost nobody has that. They are required arguments here rather
    than defaulted, because a fabricated impact parameter produces a
    confident-looking schedule built on nothing.

Used within those limits the model earns its place. Used outside them it is
quantitative decoration on a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


class ExecutionNotApplicable(RuntimeError):
    """The problem is outside the range where this model means anything."""


@dataclass(frozen=True)
class ImpactParameters:
    """Market impact, in price units per unit sold per interval.

    These are NOT defaulted. Calibrate them from your own fills: sell a known
    quantity at a known rate, measure how far the realised price fell below the
    pre-trade midpoint, and split the part that recovered (temporary, eta) from
    the part that did not (permanent, gamma).
    """

    permanent_gamma: float
    temporary_eta: float
    daily_volatility: float
    calibration_note: str = ""
    calibrated: bool = False

    @staticmethod
    def assumed(
        median_price: float, daily_volatility: float, note: str = ""
    ) -> "ImpactParameters":
        """A stand-in so the machinery can be exercised, flagged as fictional.

        Anchors impact at roughly 2% of price per unit per day permanent and 5%
        temporary, which is a guess chosen to be in the right order of magnitude
        for a thin collectibles market. Every output derived from it inherits
        the guess, and ``calibrated`` stays False so the caller cannot forget.
        """
        return ImpactParameters(
            permanent_gamma=0.02 * median_price,
            temporary_eta=0.05 * median_price,
            daily_volatility=daily_volatility,
            calibration_note=note or "ASSUMED impact, not calibrated from fills",
            calibrated=False,
        )


@dataclass(frozen=True)
class LiquidationSchedule:
    units: int
    horizon_days: float
    n_intervals: int
    kappa: float
    half_life_days: float
    holdings: tuple[float, ...]
    trades: tuple[float, ...]
    expected_cost: float
    cost_variance: float
    risk_aversion: float
    impact_calibrated: bool
    feasible: Optional[bool]
    warnings: tuple[str, ...]

    @property
    def cost_stdev(self) -> float:
        return math.sqrt(max(0.0, self.cost_variance))

    def describe(self) -> str:
        lines = [
            f"liquidating {self.units} units over {self.horizon_days:.0f} days "
            f"in {self.n_intervals} intervals",
            f"  urgency half-life : {self.half_life_days:.1f} days",
            f"  expected impact cost: ${self.expected_cost:,.2f}",
            f"  cost std deviation  : ${self.cost_stdev:,.2f}",
            "  schedule (units remaining):",
        ]
        step = max(1, self.n_intervals // 8)
        for i in range(0, len(self.holdings), step):
            day = i * self.horizon_days / max(1, self.n_intervals)
            lines.append(f"    day {day:5.1f}  hold {self.holdings[i]:6.2f}")
        if self.feasible is False:
            lines.append("  INFEASIBLE against observed market volume")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def solve_kappa(
    risk_aversion: float, sigma: float, eta_tilde: float, tau: float
) -> float:
    """Solve 2(cosh(kappa tau) - 1)/tau^2 = lambda sigma^2 / eta_tilde."""
    if risk_aversion <= 0 or sigma <= 0:
        return 0.0  # risk-neutral: sell linearly
    if eta_tilde <= 0:
        raise ExecutionNotApplicable(
            "temporary impact net of permanent impact is non-positive; the "
            "interval is too long relative to eta for the model to hold"
        )
    target = risk_aversion * sigma**2 / eta_tilde
    # cosh(kappa tau) = 1 + target tau^2 / 2
    rhs = 1.0 + target * tau**2 / 2.0
    if rhs < 1.0:
        return 0.0
    return float(math.acosh(rhs) / tau)


def optimal_liquidation(
    units: int,
    horizon_days: float,
    impact: ImpactParameters,
    risk_aversion: float = 1e-6,
    n_intervals: int = 20,
) -> LiquidationSchedule:
    """Almgren-Chriss trajectory for selling `units` over `horizon_days`."""
    warnings: list[str] = []
    if units < 2:
        raise ExecutionNotApplicable(
            f"{units} unit(s): there is no execution schedule to optimise. "
            "Almgren-Chriss is about splitting an order; a single card is a "
            "single decision. Use the liquidity estimate instead."
        )
    if units < 5:
        warnings.append(
            f"{units} units is a very small order. The continuous trajectory "
            "below will be rounded to integers and most of its structure will "
            "disappear in the rounding."
        )
    if horizon_days <= 0:
        raise ValueError("horizon must be positive")

    tau = horizon_days / n_intervals
    eta_tilde = impact.temporary_eta - impact.permanent_gamma * tau / 2.0
    kappa = solve_kappa(risk_aversion, impact.daily_volatility, eta_tilde, tau)

    times = np.linspace(0.0, horizon_days, n_intervals + 1)
    if kappa <= 1e-12:
        holdings = units * (1.0 - times / horizon_days)
        warnings.append(
            "risk aversion is effectively zero, so the schedule is linear. That "
            "minimises impact cost and maximises exposure to price drift."
        )
    else:
        holdings = units * np.sinh(kappa * (horizon_days - times)) / math.sinh(
            kappa * horizon_days
        )
    trades = -np.diff(holdings)

    expected_cost = (
        0.5 * impact.permanent_gamma * units**2
        + (eta_tilde / tau) * float(np.sum(trades**2))
    )
    cost_variance = (
        impact.daily_volatility**2 * tau * float(np.sum(holdings[1:] ** 2))
    )

    if not impact.calibrated:
        warnings.append(
            "impact parameters are not calibrated from real fills. "
            + (impact.calibration_note or "")
            + " The SHAPE of this schedule is meaningful; the dollar figures are not."
        )

    return LiquidationSchedule(
        units=units,
        horizon_days=horizon_days,
        n_intervals=n_intervals,
        kappa=kappa,
        half_life_days=(1.0 / kappa) if kappa > 1e-12 else float("inf"),
        holdings=tuple(float(x) for x in holdings),
        trades=tuple(float(x) for x in trades),
        expected_cost=expected_cost,
        cost_variance=cost_variance,
        risk_aversion=risk_aversion,
        impact_calibrated=impact.calibrated,
        feasible=None,
        warnings=tuple(warnings),
    )


def check_feasibility(
    schedule: LiquidationSchedule, sales_per_month: float, tolerance: float = 1.5
) -> LiquidationSchedule:
    """Test the schedule against the volume the market has actually absorbed.

    This is the check that keeps Almgren-Chriss honest for collectibles. The
    model happily returns a smooth trajectory for any inputs; it has no concept
    of a market that trades 0.4 times a month. Comparing the peak required rate
    against the observed Poisson arrival rate is what distinguishes a plan from
    arithmetic.
    """
    warnings = list(schedule.warnings)
    tau_days = schedule.horizon_days / schedule.n_intervals
    peak_per_day = max(schedule.trades) / tau_days if schedule.trades else 0.0
    observed_per_day = sales_per_month / 30.0

    feasible = True
    if observed_per_day <= 0:
        feasible = False
        warnings.append(
            "no observed sales volume at all. No liquidation schedule is "
            "executable in a market with no recorded trades."
        )
    elif peak_per_day > tolerance * observed_per_day:
        feasible = False
        warnings.append(
            f"schedule peaks at {peak_per_day:.2f} sales/day but the market has "
            f"absorbed {observed_per_day:.2f}/day. The trajectory is not "
            f"executable; extend the horizon to at least "
            f"{schedule.units / max(observed_per_day, 1e-9):.0f} days or accept "
            "that you are the marginal seller."
        )
    elif peak_per_day > 0.6 * observed_per_day:
        warnings.append(
            f"schedule peaks at {peak_per_day:.2f} sales/day against an observed "
            f"{observed_per_day:.2f}/day. You would be a large fraction of this "
            "card's total volume, which is where the impact assumptions start "
            "to matter most and are least calibrated."
        )

    return LiquidationSchedule(
        **{**schedule.__dict__, "feasible": feasible, "warnings": tuple(warnings)}
    )


def efficient_frontier(
    units: int,
    horizon_days: float,
    impact: ImpactParameters,
    risk_aversions: Sequence[float] = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
    n_intervals: int = 20,
) -> list[tuple[float, float, float]]:
    """Trace (risk_aversion, expected_cost, cost_stdev) across urgency levels.

    The frontier is the actual output of the model: there is no single right
    schedule, only a curve of cost-versus-risk, and where you sit on it is a
    preference rather than a calculation.
    """
    out = []
    for lam in risk_aversions:
        s = optimal_liquidation(units, horizon_days, impact, lam, n_intervals)
        out.append((lam, s.expected_cost, s.cost_stdev))
    return out
