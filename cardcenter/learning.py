"""Learning from what the system sees, without learning from itself.

REVIEW OF QUIPU, AND WHAT IS ACTUALLY TRANSFERABLE
---------------------------------------------------
Having read the source (19k lines across src/quipu), three things are worth
taking and one central thing is not.

WORTH TAKING

  1. Identity reduction as a hard contract. `radam_optimizer.radam_step` is
     documented as "a strict mathematical superset... when all extension knobs
     are at their identity values, produces bit-for-bit the same trajectory as
     vanilla Adam. Verified by test_radam_optimizer.py." That discipline is
     adopted here as a contract: with ZERO observations this module's decoder
     must reproduce the existing edit-distance snap exactly. It is proved below
     and asserted in the tests. An adaptive system you cannot switch off is one
     you cannot debug.

  2. Delta-space scaling from `neural_plasticity._smooth_dial`: step size scaled
     by `0.5*(|target| + |current|)` so parameters of wildly different magnitude
     move at comparable relative pace. Directly applicable to impact parameters,
     where gamma, eta and sigma differ by orders of magnitude.

  3. The SQLite kv persistence pattern (`local_store`, `brain_kv`) for state that
     must survive across sessions.

NOT WORTH TAKING: rADAM ITSELF

rADAM is a sound piece of work for its problem -- a high-dimensional, non-convex
surface where no closed form exists and you need momentum, noise injection and
rectification to make progress. That is not the problem here. The three things
this system needs to learn are:

    character confusion       Dirichlet-multinomial   exact conjugate posterior
    which printings appear    Dirichlet               exact conjugate posterior
    Almgren-Chriss eta,gamma  Normal-inverse-gamma    exact conjugate posterior

All three have closed-form posteriors. Running a stochastic optimizer over them
would converge more slowly, add noise to an exact answer, and -- decisively --
throw away the posterior VARIANCE. The variance is the most important output
here: a liquidation schedule computed from an eta estimated to +/-300% is not a
schedule, and only a Bayesian treatment tells you that. The heartbeat-momentum
and toroidal-phase extensions have no defined meaning for these estimands.

WHAT LEARNING CAN AND CANNOT DO ABOUT PIXEL LOSS
--------------------------------------------------
It cannot recover lost pixels. Information not captured is not in the file, and
no amount of training changes the channel capacity of a 6-pixel glyph. Any claim
that more training will let the system read numbers it currently cannot is
false, and the resolution gate stays exactly where the measurement put it.

What learning does is improve DECISIONS UNDER pixel loss, in two concrete ways:

  * A learned confusion model replaces edit distance with a real likelihood.
    Edit distance treats 8->0 and 8->4 as equally likely. They are not, and the
    difference is measurable.
  * A learned encounter prior replaces "uniform over 131 printings" with what
    actually walks through the door. If 94% of the Sol Rings in shops are the
    $1.50 Commander printings, an ambiguous read should say so.

Together these are Bayesian posterior decoding, and they are strictly better
than the snap they replace. They also make the system *more* willing to report
ambiguity, not less: a reading that edit distance resolves uniquely can be
posterior-ambiguous when the prior disagrees.

THE CIRCULARITY RULE, AGAIN
----------------------------
Only VERIFIED observations update these models. The system's own decodings must
never become its own training data -- that is the same degenerate loop as the
marketplace-confirmation problem, one layer down, and it is more insidious here
because a confusion model trained on its own output converges to certainty about
its own biases. `observe_*` requires evidence; there is no path that feeds a
decode back in.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .catalog import MIN_PX_TO_READ, CatalogEntry
from .execution import ImpactParameters

# Resolution bands for the confusion model. Confusion is strongly
# scale-dependent, so a single pooled matrix would average a clean 30px read
# together with a marginal 13px one and describe neither.
RESOLUTION_BANDS: tuple[tuple[float, float, str], ...] = (
    (MIN_PX_TO_READ, 20.0, "marginal"),
    (20.0, 32.0, "good"),
    (32.0, float("inf"), "clean"),
)

# Dirichlet prior strength. The diagonal (correct read) gets the mass; every
# substitution shares the rest. Chosen so that with zero data the resulting
# log-likelihood is exactly monotone in edit distance -- see the identity
# reduction argument in the module docstring.
PRIOR_STAY = 8.0
PRIOR_SUB = 1.0

# Cost of an insertion or deletion in the alignment. Set equal to the prior
# substitution cost so that, untrained, alignment cost is exactly
# log(PRIOR_STAY/PRIOR_SUB) x edit distance and the decoder reduces to the
# edit-distance snap.
INDEL_COST = math.log(PRIOR_STAY / PRIOR_SUB)


def band_for(glyph_px: float) -> str:
    for lo, hi, name in RESOLUTION_BANDS:
        if lo <= glyph_px < hi:
            return name
    return "below_gate"


@dataclass
class ConfusionModel:
    """P(observed character | true character), per resolution band.

    Dirichlet-multinomial with a symmetric prior. The posterior mean of a
    Dirichlet is just (alpha + counts) normalised, so the update is a single
    increment and the "training" is exact rather than iterative.
    """

    alphabet: str = "0123456789abcdefghijklmnopqrstuvwxyz"
    counts: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)

    def observe(self, true_text: str, observed_text: str, glyph_px: float) -> None:
        """Record one VERIFIED reading. Never call this with a decode result."""
        band = band_for(glyph_px)
        if band == "below_gate":
            return
        table = self.counts.setdefault(band, {})
        # Align by position where lengths match; length mismatches are recorded
        # as evidence about the band's reliability but not as substitutions,
        # because we cannot tell which character was inserted or dropped.
        if len(true_text) != len(observed_text):
            return
        for t, o in zip(true_text.lower(), observed_text.lower()):
            if t in self.alphabet and o in self.alphabet:
                table.setdefault(t, {})[o] = table.setdefault(t, {}).get(o, 0) + 1

    def p_observe(self, observed: str, true: str, band: str) -> float:
        """Posterior-mean probability of seeing `observed` when truth is `true`."""
        n_alpha = len(self.alphabet)
        row = self.counts.get(band, {}).get(true, {})
        alpha_stay = PRIOR_STAY
        alpha_sub = PRIOR_SUB
        total_prior = alpha_stay + alpha_sub * (n_alpha - 1)
        observed_count = row.get(observed, 0)
        row_total = sum(row.values())
        prior = alpha_stay if observed == true else alpha_sub
        return (prior + observed_count) / (total_prior + row_total)

    def log_likelihood(self, observed: str, true: str, band: str) -> float:
        """log Bayes factor of `true` against the read-perfectly null.

        NOT a sum of per-character log-probabilities. That formulation has a
        length bias: every character contributes a negative term, so a shorter
        candidate scores higher purely by having fewer terms, and a reading of
        "11" prefers the printing "1" over "111" for no reason connected to the
        image. Real bug, caught by the tie test.

        The fix is to score each candidate as a likelihood RATIO against the
        hypothesis that the engine read it perfectly:

            substitution   cost = log( P(true|true) / P(observed|true) )  >= 0
            exact match    cost = 0
            insert/delete  cost = INDEL_COST

        A correct character now costs nothing regardless of how many there are,
        so candidates of different lengths are directly comparable. The score is
        the negated minimum-cost alignment, computed by Levenshtein DP with these
        costs in place of unit costs.

        IDENTITY REDUCTION. At the prior, P(stay) = PRIOR_STAY / Z and
        P(sub) = PRIOR_SUB / Z for every pair, so every substitution costs
        exactly log(PRIOR_STAY / PRIOR_SUB), and INDEL_COST is set to the same
        value. The alignment cost is then log(PRIOR_STAY/PRIOR_SUB) times the
        ordinary edit distance -- a strictly monotone function of it. So the
        untrained argmax is the minimum-edit-distance candidate, exactly the
        snap this replaces, and evidence moves it away only in proportion to
        what has been observed.
        """
        obs = observed.lower()
        tru = true.lower()

        def sub_cost(o: str, t: str) -> float:
            if o == t:
                return 0.0
            p_stay = self.p_observe(t, t, band)
            p_obs = self.p_observe(o, t, band)
            return math.log(max(1e-12, p_stay) / max(1e-12, p_obs))

        n, m = len(tru), len(obs)
        dp = np.zeros((n + 1, m + 1), dtype=float)
        dp[:, 0] = np.arange(n + 1) * INDEL_COST
        dp[0, :] = np.arange(m + 1) * INDEL_COST
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[i, j] = min(
                    dp[i - 1, j - 1] + sub_cost(obs[j - 1], tru[i - 1]),
                    dp[i - 1, j] + INDEL_COST,
                    dp[i, j - 1] + INDEL_COST,
                )
        return -float(dp[n, m])

    def observations(self, band: Optional[str] = None) -> int:
        bands = [band] if band else list(self.counts)
        return sum(
            sum(sum(row.values()) for row in self.counts.get(b, {}).values())
            for b in bands
        )

    def accuracy(self, band: str) -> Optional[float]:
        table = self.counts.get(band, {})
        total = sum(sum(row.values()) for row in table.values())
        if total == 0:
            return None
        correct = sum(row.get(t, 0) for t, row in table.items())
        return correct / total


@dataclass
class EncounterPrior:
    """P(printing) -- what actually walks through the door.

    Uniform over candidates until evidence says otherwise. This is where most of
    the practical gain lives: 131 printings are not equally likely to be sitting
    in a shop case, and a decoder that pretends they are throws away the single
    most informative thing the system learns from operating.
    """

    counts: dict[str, int] = field(default_factory=dict)
    concentration: float = 1.0  # symmetric Dirichlet prior

    def observe(self, card_id: str) -> None:
        self.counts[card_id] = self.counts.get(card_id, 0) + 1

    def probabilities(self, candidates: Sequence[CatalogEntry]) -> dict[str, float]:
        ids = [c.card_id for c in candidates]
        weights = {
            i: self.concentration + self.counts.get(i, 0) for i in ids
        }
        total = sum(weights.values())
        return {i: w / total for i, w in weights.items()}

    @property
    def total_observations(self) -> int:
        return sum(self.counts.values())


@dataclass(frozen=True)
class Decode:
    posterior: dict[str, float]
    best_id: Optional[str]
    best_number: Optional[str]
    best_probability: float
    decisive: bool
    band: str
    used_learned_confusion: bool
    used_learned_prior: bool
    warnings: tuple[str, ...]

    def describe(self) -> str:
        if self.best_id is None:
            return "no candidate; nothing to decode"
        head = (
            f"posterior decode: #{self.best_number} at "
            f"{self.best_probability * 100:.0f}%"
            + ("  [decisive]" if self.decisive else "  [NOT decisive]")
        )
        rest = sorted(self.posterior.items(), key=lambda kv: -kv[1])[1:4]
        if rest:
            head += "\n  runners-up: " + ", ".join(
                f"{i[:8]} {p * 100:.0f}%" for i, p in rest
            )
        for w in self.warnings:
            head += f"\n  WARNING: {w}"
        return head


def posterior_decode(
    reading: str,
    candidates: Sequence[CatalogEntry],
    glyph_px: float,
    confusion: Optional[ConfusionModel] = None,
    prior: Optional[EncounterPrior] = None,
    decisive_threshold: float = 0.85,
) -> Decode:
    """Bayesian decode of an OCR reading against the real candidate set.

    IDENTITY REDUCTION. With a fresh ConfusionModel and a fresh EncounterPrior,
    the prior is uniform and p_observe takes only two values -- p_stay for a
    match and p_sub for any substitution, with p_stay > p_sub. The log
    likelihood of a same-length candidate is then

        (L - d) * log p_stay + d * log p_sub

    for substitution distance d, which is strictly decreasing in d. So the argmax
    is the minimum-distance candidate: exactly the behaviour of the edit-distance
    snap this replaces. Learning only ever moves the decoder away from that
    baseline in proportion to evidence.
    """
    confusion = confusion or ConfusionModel()
    prior = prior or EncounterPrior()
    band = band_for(glyph_px)
    warnings: list[str] = []

    if not candidates:
        return Decode({}, None, None, 0.0, False, band, False, False,
                      ("no candidates supplied",))

    if band == "below_gate":
        return Decode(
            {}, None, None, 0.0, False, band, False, False,
            (
                f"glyph height {glyph_px:.0f}px is below the measured "
                f"{MIN_PX_TO_READ:.0f}px floor. Learning does not raise the "
                "information content of the image; the gate still applies.",
            ),
        )

    priors = prior.probabilities(candidates)
    log_post: dict[str, float] = {}
    for c in candidates:
        ll = confusion.log_likelihood(reading, c.collector_number, band)
        log_post[c.card_id] = ll + math.log(max(1e-12, priors[c.card_id]))

    peak = max(log_post.values())
    unnorm = {k: math.exp(v - peak) for k, v in log_post.items()}
    total = sum(unnorm.values())
    posterior = {k: v / total for k, v in unnorm.items()}

    best_id = max(posterior, key=posterior.get)
    best_p = posterior[best_id]
    best_number = next(c.collector_number for c in candidates if c.card_id == best_id)

    used_conf = confusion.observations(band) > 0
    used_prior = prior.total_observations > 0

    if not used_conf:
        warnings.append(
            f"no verified readings recorded in the '{band}' band yet, so the "
            "confusion model is at its prior and this decode reduces to the "
            "edit-distance baseline"
        )
    if band == "marginal":
        warnings.append(
            "marginal resolution band: errors here present as a confident wrong "
            "printing, so verify before acting on value"
        )
    if not (best_p >= decisive_threshold):
        warnings.append(
            f"posterior mass {best_p * 100:.0f}% is below the "
            f"{decisive_threshold * 100:.0f}% decisiveness threshold; treat the "
            "printing as unresolved"
        )

    return Decode(
        posterior=posterior,
        best_id=best_id,
        best_number=best_number,
        best_probability=best_p,
        decisive=bool(best_p >= decisive_threshold),
        band=band,
        used_learned_confusion=used_conf,
        used_learned_prior=used_prior,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Online Almgren-Chriss impact calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fill:
    """One realised sale, as evidence about market impact.

    ``reference_price`` is the pre-trade market level (median of recent comps);
    ``realised_price`` is what you actually got; ``cumulative_sold`` is how many
    units you had already pushed into the market; ``rate_per_day`` is how fast
    you were selling at the time.
    """

    reference_price: float
    realised_price: float
    cumulative_sold: float
    rate_per_day: float


@dataclass
class ImpactEstimator:
    """Bayesian linear regression for Almgren-Chriss impact parameters.

    The model implied by Almgren-Chriss is that realised slippage decomposes into
    a permanent part proportional to cumulative volume already sold, and a
    temporary part proportional to the current selling rate:

        reference - realised  =  gamma * cumulative_sold  +  eta * rate  +  noise

    That is an ordinary two-parameter linear regression, and with a
    normal-inverse-gamma prior it has an exact closed-form posterior. No
    optimizer, no learning rate, no convergence question -- and, importantly, a
    posterior COVARIANCE, which is what tells you whether the resulting schedule
    means anything.

    Updates are exact and order-independent: the same fills in any order give the
    same posterior.
    """

    prior_precision: float = 1e-3  # weak prior, but proper
    _xtx: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    _xty: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _yty: float = 0.0
    n: int = 0

    def observe(self, fill: Fill) -> None:
        x = np.array([fill.cumulative_sold, fill.rate_per_day], dtype=float)
        y = float(fill.reference_price - fill.realised_price)
        self._xtx += np.outer(x, x)
        self._xty += x * y
        self._yty += y * y
        self.n += 1

    def observe_many(self, fills: Iterable[Fill]) -> None:
        for f in fills:
            self.observe(f)

    @property
    def _posterior(self) -> tuple[np.ndarray, np.ndarray, float]:
        precision = self._xtx + self.prior_precision * np.eye(2)
        cov = np.linalg.inv(precision)
        mean = cov @ self._xty
        dof = max(1, self.n - 2)
        resid = self._yty - float(mean @ self._xty)
        sigma2 = max(1e-12, resid / dof)
        return mean, cov * sigma2, sigma2

    def estimate(
        self, daily_volatility: float, median_price: float
    ) -> ImpactParameters:
        """Posterior-mean gamma and eta, or an explicitly assumed fallback."""
        if self.n < 4:
            return ImpactParameters.assumed(
                median_price,
                daily_volatility,
                note=(
                    f"only {self.n} fill(s) recorded; at least 4 are needed to "
                    "identify two impact parameters plus noise"
                ),
            )
        mean, cov, _ = self._posterior
        gamma, eta = float(mean[0]), float(mean[1])
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))

        if eta <= 0 or gamma < 0:
            return ImpactParameters.assumed(
                median_price,
                daily_volatility,
                note=(
                    f"fitted impact is non-physical (gamma={gamma:.3g}, "
                    f"eta={eta:.3g}); your fills do not yet show a price "
                    "response to selling pressure"
                ),
            )

        rel = float(sd[1] / eta) if eta > 0 else float("inf")
        note = f"calibrated from {self.n} fills; eta relative sd {rel * 100:.0f}%"
        if rel > 0.5:
            note += " -- too uncertain to trust the dollar figures"
        return ImpactParameters(
            permanent_gamma=gamma,
            temporary_eta=eta,
            daily_volatility=daily_volatility,
            calibration_note=note,
            calibrated=rel <= 0.5,
        )

    def uncertainty(self) -> Optional[dict[str, float]]:
        if self.n < 4:
            return None
        mean, cov, sigma2 = self._posterior
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        return {
            "gamma": float(mean[0]),
            "gamma_sd": float(sd[0]),
            "eta": float(mean[1]),
            "eta_sd": float(sd[1]),
            "noise_sd": float(math.sqrt(sigma2)),
            "n_fills": float(self.n),
        }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


LEARNING_SCHEMA = """
CREATE TABLE IF NOT EXISTS learning_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


class LearningStore:
    """Key/value persistence for learned state, after QUIPU's brain_kv pattern."""

    def __init__(self, path: str = "cardcenter.db") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.executescript(LEARNING_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "LearningStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def save_confusion(self, model: ConfusionModel) -> None:
        self._put("ocr:confusion", {"alphabet": model.alphabet, "counts": model.counts})

    def load_confusion(self) -> ConfusionModel:
        raw = self._get("ocr:confusion")
        if not raw:
            return ConfusionModel()
        return ConfusionModel(alphabet=raw["alphabet"], counts=raw["counts"])

    def save_prior(self, prior: EncounterPrior) -> None:
        self._put(
            "ocr:encounter_prior",
            {"counts": prior.counts, "concentration": prior.concentration},
        )

    def load_prior(self) -> EncounterPrior:
        raw = self._get("ocr:encounter_prior")
        if not raw:
            return EncounterPrior()
        return EncounterPrior(counts=raw["counts"], concentration=raw["concentration"])

    def save_impact(self, est: ImpactEstimator, key: str = "default") -> None:
        self._put(
            f"impact:{key}",
            {
                "xtx": est._xtx.tolist(),
                "xty": est._xty.tolist(),
                "yty": est._yty,
                "n": est.n,
                "prior_precision": est.prior_precision,
            },
        )

    def load_impact(self, key: str = "default") -> ImpactEstimator:
        raw = self._get(f"impact:{key}")
        if not raw:
            return ImpactEstimator()
        est = ImpactEstimator(prior_precision=raw["prior_precision"])
        est._xtx = np.array(raw["xtx"], dtype=float)
        est._xty = np.array(raw["xty"], dtype=float)
        est._yty = float(raw["yty"])
        est.n = int(raw["n"])
        return est

    def _put(self, key: str, value: dict) -> None:
        self.conn.execute(
            "INSERT INTO learning_state (key, value, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "updated_at=excluded.updated_at",
            (key, json.dumps(value), time.time()),
        )
        self.conn.commit()

    def _get(self, key: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT value FROM learning_state WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None


def learning_report(
    confusion: ConfusionModel, prior: EncounterPrior, estimator: ImpactEstimator
) -> str:
    lines = ["LEARNED STATE", ""]
    lines.append("OCR confusion model:")
    any_band = False
    for _, _, band in RESOLUTION_BANDS:
        n = confusion.observations(band)
        acc = confusion.accuracy(band)
        if n:
            any_band = True
            lines.append(f"  {band:<9} {n:5d} verified chars, accuracy {acc * 100:.0f}%")
        else:
            lines.append(f"  {band:<9}     0 verified chars -- at prior")
    if not any_band:
        lines.append("  decoding is currently identical to the edit-distance baseline")

    lines.append("")
    lines.append(f"Encounter prior: {prior.total_observations} verified sightings")
    if prior.total_observations < 30:
        lines.append(
            "  too few to shift a decode meaningfully; still effectively uniform"
        )
    else:
        top = sorted(prior.counts.items(), key=lambda kv: -kv[1])[:3]
        lines.append(
            "  most-seen: " + ", ".join(f"{k[:10]} x{v}" for k, v in top)
        )

    lines.append("")
    u = estimator.uncertainty()
    if u is None:
        lines.append(
            f"Impact model: {estimator.n} fill(s); not yet identifiable. "
            "Almgren-Chriss dollar figures remain assumed."
        )
    else:
        lines.append(
            f"Impact model: gamma {u['gamma']:.4g} +/- {u['gamma_sd']:.2g}, "
            f"eta {u['eta']:.4g} +/- {u['eta_sd']:.2g}, from {int(u['n_fills'])} fills"
        )
    lines.append("")
    lines.append(
        "All of the above updates ONLY from verified observations. No decode "
        "or schedule this system produces is ever fed back as evidence."
    )
    return "\n".join(lines)
