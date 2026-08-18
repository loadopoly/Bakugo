"""Identifying which card this is -- and which PRINT of it.

The catalogs exist and they are good. Scryfall covers Magic completely and free,
carries every printing with collector numbers, finishes and market prices, and
is queried live below. Pokemon, Yu-Gi-Oh and the sports catalogs have their own
equivalents. Getting card data was never the hard part and it was wrong of me to
file this under "not built".

THE HARD PART IS THE PRINT, NOT THE CARD
-----------------------------------------
Scryfall lists 131 printings of Sol Ring. Naming the card is easy: the art and
the title are large, high-contrast, and match robustly under ORB features even
through glass at an angle. Naming the *printing* is what determines the price,
and the printings differ by:

    collector number     ~1.5 mm of printed text
    set symbol           ~3.5 mm glyph
    foil vs non-foil     a specular property, not a spatial one
    border colour        whole-card, trivially resolvable
    full-art / textless  whole-card, trivially resolvable

Cross-referenced against the resolution available at a display case:

    arm's length, 1x      4 px/mm    collector number =  6 px tall  -> hopeless
    leaning, 1x           8 px/mm    collector number = 12 px tall  -> unreliable
    leaning, 2.5x        18 px/mm    collector number = 27 px tall  -> workable

So at 1x across a counter you can identify the card and not the print. For a
card whose printings span 100x in price, an identification that confidently
picks one print is worse than one that reports the ambiguity, because it
produces a specific wrong number instead of a visible question.

This module therefore returns an ``Identification`` that is explicitly either
RESOLVED or AMBIGUOUS, and when ambiguous it reports the full price spread
across the candidate printings and refuses to collapse it to a point estimate.

Foil detection deserves its own warning: through display glass, foil and glare
look alike, and the whole reason for the quality gate is that glare is
everywhere. Finish is never inferred from a through-glass frame.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np

USER_AGENT = "cardcenter/0.3 (card centering measurement; contact: local use)"

# Approximate printed sizes of the features that distinguish one printing from
# another, in millimetres. These drive the resolution gate below.
FEATURE_SIZE_MM: dict[str, float] = {
    "collector_number": 1.5,
    "set_symbol": 3.5,
    "artist_line": 1.5,
    "border_color": 60.0,
    "full_art": 60.0,
    "textless": 40.0,
    "frame_effects": 40.0,
    "finish": 0.0,  # not a spatial feature at all
}

# Glyph height in pixels before a printed feature can be read. This value is
# MEASURED, not assumed. Rendering collector numbers at a range of glyph heights
# with through-glass-like blur and noise, then reading them with Tesseract and
# snapping to the catalog, gives:
#
#     glyph px |  8 candidates | 30 candidates | 131 candidates
#         6    |      38%      |      42%      |      17%
#         9    |      75%      |      83%      |      75%
#        12    |     100%      |     100%      |      92%
#        18    |      62%*     |      92%      |     100%
#
#   (* small-sample artefact: with only 8 candidates they are all single digits,
#      which collide heavily under edit distance.)
#
# The number that sets the floor is not the accuracy but the ERROR MODE. Below
# 12 px a failed read does not come back empty -- it snaps to a real but wrong
# collector number, i.e. a confident selection of the wrong printing. At 9 px
# roughly one reading in five is confidently wrong. That is the failure this
# project exists to prevent, so the gate sits at 12 with a marginal band above.
MIN_PX_TO_READ = 12.0

# Between MIN_PX_TO_READ and this, readings are attempted but flagged for human
# verification: accuracy is high but not high enough to bet a high-value card on.
MARGINAL_PX_TO_READ = 20.0


class CatalogUnavailable(RuntimeError):
    """Catalog could not be reached or returned nothing usable."""


@dataclass(frozen=True)
class CatalogEntry:
    """One specific printing of one card."""

    card_id: str
    name: str
    set_name: str
    collector_number: str
    image_url: Optional[str] = None
    prices: dict[str, float] = field(default_factory=dict)
    attributes: dict[str, object] = field(default_factory=dict)

    @property
    def best_price(self) -> Optional[float]:
        for k in ("usd", "usd_foil", "usd_etched", "eur"):
            v = self.prices.get(k)
            if v:
                return float(v)
        return None

    def label(self) -> str:
        return f"{self.name} [{self.set_name} #{self.collector_number}]"


class CardCatalog(Protocol):
    def find_by_name(self, name: str) -> list[CatalogEntry]: ...
    @property
    def description(self) -> str: ...


@dataclass
class ScryfallCatalog:
    """Live Scryfall lookup. Real endpoint, no key required, be polite.

    Scryfall asks for a descriptive User-Agent and roughly 100ms between
    requests; both are honoured here. Prices come from Scryfall's aggregation
    and are indicative market prices, NOT sold comparables for a graded copy --
    a graded PSA 9 does not trade at the raw market price, so these feed
    identification and ambiguity spread, not the offer maths.
    """

    cache_dir: Optional[str] = None
    min_interval_s: float = 0.12
    _last_call: float = field(default=0.0, init=False)

    def _get(self, url: str) -> dict:
        cache_path = None
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            key = str(abs(hash(url)))
            cache_path = Path(self.cache_dir) / f"{key}.json"
            if cache_path.exists():
                return json.loads(cache_path.read_text())

        wait = self.min_interval_s - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            raise CatalogUnavailable(f"Scryfall request failed: {exc}") from exc
        finally:
            self._last_call = time.time()

        if cache_path is not None:
            cache_path.write_text(json.dumps(payload))
        return payload

    def find_by_name(self, name: str) -> list[CatalogEntry]:
        q = urllib.parse.quote(f'!"{name}"')
        payload = self._get(
            f"https://api.scryfall.com/cards/search?q={q}&unique=prints"
        )
        out: list[CatalogEntry] = []
        for c in payload.get("data", []):
            prices = {
                k: float(v)
                for k, v in (c.get("prices") or {}).items()
                if v not in (None, "")
            }
            imgs = c.get("image_uris") or {}
            out.append(
                CatalogEntry(
                    card_id=c["id"],
                    name=c.get("name", name),
                    set_name=c.get("set_name", "?"),
                    collector_number=str(c.get("collector_number", "?")),
                    image_url=imgs.get("normal") or imgs.get("small"),
                    prices=prices,
                    attributes={
                        "border_color": c.get("border_color"),
                        "finishes": c.get("finishes"),
                        "full_art": c.get("full_art"),
                        "textless": c.get("textless"),
                        "frame_effects": c.get("frame_effects"),
                        "promo": c.get("promo"),
                    },
                )
            )
        if not out:
            raise CatalogUnavailable(f"no printings found for {name!r}")
        return out

    @property
    def description(self) -> str:
        return "Scryfall (live, prices are raw-market indicative)"


@dataclass
class LocalCatalog:
    """Offline catalog from a JSON file. Same shape as the live one."""

    entries: list[CatalogEntry]
    note: str = "local catalog"

    @staticmethod
    def from_json(path: str) -> "LocalCatalog":
        data = json.loads(Path(path).read_text())
        return LocalCatalog(
            entries=[CatalogEntry(**e) for e in data], note=f"local catalog {path}"
        )

    def find_by_name(self, name: str) -> list[CatalogEntry]:
        hits = [e for e in self.entries if e.name.lower() == name.lower()]
        if not hits:
            raise CatalogUnavailable(f"{name!r} not in {self.note}")
        return hits

    @property
    def description(self) -> str:
        return self.note


# ---------------------------------------------------------------------------
# Visual matching
# ---------------------------------------------------------------------------


def orb_signature(image: np.ndarray, n_features: int = 900):
    """ORB keypoints and descriptors from a rectified card.

    ORB rather than a perceptual hash because identification has to survive what
    a display case does to a photograph: a different exposure, a different white
    balance, a few degrees of residual rotation and a glare patch across one
    corner. A hash compares whole images and fails on all four. Local features
    survive partial occlusion, which is precisely the glare case.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.equalizeHist(gray)
    orb = cv2.ORB_create(nfeatures=n_features)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des


def match_score(
    query: tuple, reference: tuple, ratio: float = 0.75, min_inliers: int = 12
) -> tuple[int, float]:
    """Geometrically verified match count between two ORB signatures.

    Returns (inliers, inlier_fraction). Raw descriptor matches are not enough:
    two cards from the same set share frames, mana symbols and typography, and
    will produce dozens of spurious matches. Requiring the matches to agree on a
    single homography throws those out, because a real match is a consistent
    geometric transform and a coincidental one is not.
    """
    kp1, des1 = query
    kp2, des2 = reference
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return 0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    if len(good) < min_inliers:
        return 0, 0.0

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return 0, 0.0
    inliers = int(mask.sum())
    return inliers, inliers / max(1, len(good))


# ---------------------------------------------------------------------------
# Variant resolution
# ---------------------------------------------------------------------------


def resolvable_features(px_per_mm: float) -> dict[str, bool]:
    """Which distinguishing features are physically readable at this scale."""
    out = {}
    for feat, size_mm in FEATURE_SIZE_MM.items():
        if feat == "finish":
            out[feat] = False  # never spatial; never inferred through glass
        else:
            out[feat] = (size_mm * px_per_mm) >= MIN_PX_TO_READ
    return out


def distinguishing_features(entries: Sequence[CatalogEntry]) -> set[str]:
    """Which attributes actually differ across this set of printings."""
    if len(entries) < 2:
        return set()
    diff: set[str] = set()
    if len({e.collector_number for e in entries}) > 1:
        diff.add("collector_number")
    if len({e.set_name for e in entries}) > 1:
        diff.add("set_symbol")
    for key in ("border_color", "full_art", "textless", "frame_effects"):
        vals = {json.dumps(e.attributes.get(key), sort_keys=True) for e in entries}
        if len(vals) > 1:
            diff.add(key if key in FEATURE_SIZE_MM else "frame_effects")
    finishes = {json.dumps(e.attributes.get("finishes"), sort_keys=True) for e in entries}
    if len(finishes) > 1:
        diff.add("finish")
    return diff


@dataclass(frozen=True)
class Identification:
    name: str
    candidates: list[CatalogEntry]
    resolved: Optional[CatalogEntry]
    px_per_mm: float
    unresolvable_features: tuple[str, ...]
    match_inliers: int
    catalog: str
    warnings: tuple[str, ...]

    @property
    def is_ambiguous(self) -> bool:
        return self.resolved is None and len(self.candidates) > 1

    @property
    def price_spread(self) -> Optional[tuple[float, float]]:
        prices = [e.best_price for e in self.candidates if e.best_price]
        if not prices:
            return None
        return (min(prices), max(prices))

    @property
    def spread_ratio(self) -> Optional[float]:
        s = self.price_spread
        if not s or s[0] <= 0:
            return None
        return s[1] / s[0]

    def describe(self) -> str:
        lines = [f"identified: {self.name}  ({self.match_inliers} verified features)"]
        if self.resolved is not None:
            lines.append(f"  printing: {self.resolved.label()}")
            p = self.resolved.best_price
            if p:
                lines.append(f"  raw market: ${p:,.2f}")
        else:
            lines.append(
                f"  PRINTING AMBIGUOUS: {len(self.candidates)} candidate printings"
            )
            spread = self.price_spread
            if spread:
                lines.append(
                    f"  raw market spans ${spread[0]:,.2f} - ${spread[1]:,.2f}"
                    + (f"  ({self.spread_ratio:.0f}x)" if self.spread_ratio else "")
                )
            if self.unresolvable_features:
                lines.append(
                    "  cannot read at "
                    f"{self.px_per_mm:.1f} px/mm: "
                    + ", ".join(self.unresolvable_features)
                )
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def identify(
    name: str,
    catalog: CardCatalog,
    px_per_mm: float,
    match_inliers: int = 0,
    known_set: Optional[str] = None,
    known_collector_number: Optional[str] = None,
) -> Identification:
    """Resolve a card name to a specific printing, or report that we cannot.

    ``known_set`` / ``known_collector_number`` let a caller supply what a human
    read off the card, which is the realistic path at 1x: the operator squints at
    the collector number, types it, and the ambiguity collapses.
    """
    entries = catalog.find_by_name(name)
    warnings: list[str] = []

    # A supplied collector number that matches nothing must NOT fall through to
    # the unfiltered list. Silently ignoring it and then resolving to whatever
    # the set filter leaves behind returns a confident identification of the
    # wrong printing -- which is worse than reporting ambiguity, because the
    # operator believes they disambiguated it.
    if known_collector_number:
        matched = [e for e in entries if e.collector_number == known_collector_number]
        if not matched:
            raise CatalogUnavailable(
                f"no printing of {name!r} has collector number "
                f"{known_collector_number!r}. Check the number rather than "
                "proceeding on an unverified identification."
            )
        entries = matched
    if known_set:
        narrowed = [e for e in entries if e.set_name.lower() == known_set.lower()]
        if not narrowed:
            warnings_pre = (
                f"no candidate printing is from set {known_set!r}; the set hint "
                "was ignored"
            )
        else:
            entries = narrowed
            warnings_pre = None
    else:
        warnings_pre = None

    if warnings_pre:
        warnings.append(warnings_pre)

    diff = distinguishing_features(entries)
    readable = resolvable_features(px_per_mm)
    unresolvable = tuple(sorted(f for f in diff if not readable.get(f, False)))

    resolved: Optional[CatalogEntry] = None
    if len(entries) == 1:
        resolved = entries[0]
    elif not unresolvable and diff:
        # Everything that separates these printings is large enough to see, but
        # actually reading it needs OCR and symbol classification, which is not
        # implemented. Report the honest state rather than guessing.
        warnings.append(
            "the distinguishing features are large enough to resolve at this "
            "scale, but reading them requires OCR and set-symbol classification "
            "which is not implemented. Supply the collector number to disambiguate."
        )

    if "finish" in diff:
        warnings.append(
            "these printings differ by finish (foil / etched). Foil cannot be "
            "identified from a through-glass frame: foil and glass glare look "
            "alike, and foil often carries the larger price."
        )

    ident = Identification(
        name=name,
        candidates=entries,
        resolved=resolved,
        px_per_mm=px_per_mm,
        unresolvable_features=unresolvable,
        match_inliers=match_inliers,
        catalog=catalog.description,
        warnings=tuple(warnings),
    )

    if ident.is_ambiguous and (ident.spread_ratio or 1) > 3.0:
        warnings = list(ident.warnings) + [
            f"candidate printings differ by {ident.spread_ratio:.0f}x in price. "
            "Do not make an offer on this card until the printing is pinned down."
        ]
        ident = Identification(**{**ident.__dict__, "warnings": tuple(warnings)})
    return ident
