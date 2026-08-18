"""Deriving card-type labels from catalogs, and checking them against reality.

WHY THIS IS NOT DISTILLATION
-----------------------------
Distillation transfers what a teacher knows. For card *type* no teacher is
needed: the answer is a database field. PokemonTCG.io publishes `supertype`,
`subtypes` and `rarity`; Scryfall publishes `full_art`, `border_color` and
`frame_effects`. Those are exact labels for hundreds of thousands of cards,
available now, at no annotation cost.

Approximating a large VLM's guess at something a catalog states outright would
be strictly worse: it converts a fact into an estimate. Distillation belongs
later, and only if a trained model is too heavy for the phone.

WHY THE LABEL MATTERS
---------------------
Measured on 57 real photographs, 44% of cards reach only GEOMETRY_ONLY because
they are full-bleed -- no printed border exists to measure. The tool currently
discovers this by attempting border detection and failing. Knowing the type in
advance routes the card to the right measurement immediately, and turns a
refusal into an explanation.

THE FAILURE MODE THIS DESIGN ACCEPTS
-------------------------------------
The classifier's output is a ROUTING hint, never a measurement. `capability.assess`
still determines empirically what is measurable. So a wrong type costs a wasted
attempt, not a wrong number. That is the right place for a learned component:
somewhere it can be wrong without corrupting the answer.

VOCABULARY, SAMPLED FROM THE LIVE API
--------------------------------------
Across 593 cards in three sets the rarity field takes these values, and the
mapping below is built from what is actually present rather than from memory:

    Common, Uncommon, Rare, Rare Holo            -> bordered
    Illustration Rare, Special Illustration Rare -> full_bleed
    Hyper Rare, Rare Secret, Rare Ultra          -> full_bleed
    Ultra Rare, Rare Holo V/VMAX/VSTAR           -> full_bleed (usually full-art)
    Double Rare                                  -> bordered (ex cards keep a frame)
    Radiant Rare                                 -> bordered
    ACE SPEC Rare                                -> bordered
    supertype == Energy                          -> energy
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

USER_AGENT = "cardcenter/1.6 (card-type labelling; local research use)"


class CardType(str, Enum):
    """Routing classes, chosen by what each implies for measurement."""

    BORDERED = "bordered"          # printed frame on all four sides
    FULL_BLEED = "full_bleed"      # art to the edge: full-art, Hyper/Illustration Rare
    ENERGY = "energy"              # basic Energy: full-bleed, minimal print
    TEXTURED = "textured"          # textured/etched holo: border may exist but is noisy
    UNKNOWN = "unknown"

    @property
    def expects_border(self) -> bool:
        return self in (CardType.BORDERED, CardType.TEXTURED)

    @property
    def expected_capability(self) -> str:
        """What `capability.assess` should be able to reach for this type."""
        if self is CardType.BORDERED:
            return "full"
        if self is CardType.TEXTURED:
            return "partial"
        return "geometry_only"


# Rarity -> type. Built from the live vocabulary, not from memory.
_RARITY_FULL_BLEED = {
    "illustration rare",
    "special illustration rare",
    "hyper rare",
    "mega hyper rare",
    "rare secret",
    "rare rainbow",
    "rare ultra",
    "ultra rare",
    "rare shiny",
    "shiny rare",
    "shiny ultra rare",
    "amazing rare",
    "rare holo v",
    "rare holo vmax",
    "rare holo vstar",
    "rare holo ex",
    "rare holo gx",
    "rare holo lv.x",
    "legend",
    "black white rare",
    "classic collection",
}
_RARITY_BORDERED = {
    "common",
    "uncommon",
    "rare",
    "rare holo",
    "double rare",
    "radiant rare",
    "ace spec rare",
    "rare ace",
    "promo",
    "rare break",
    "rare prime",
}
# Subtypes that imply full-art regardless of rarity wording.
_SUBTYPE_FULL_BLEED = {"v", "vmax", "vstar", "v-union", "radiant", "tera"}


@dataclass(frozen=True)
class TypeLabel:
    card_id: str
    name: str
    card_type: CardType
    rarity: Optional[str]
    supertype: Optional[str]
    subtypes: tuple[str, ...]
    image_url: Optional[str]
    confidence: str  # "exact" | "inferred" | "guess"
    rule: str

    def as_row(self) -> dict:
        return {
            "card_id": self.card_id,
            "name": self.name,
            "label": self.card_type.value,
            "rarity": self.rarity,
            "supertype": self.supertype,
            "subtypes": list(self.subtypes),
            "image_url": self.image_url,
            "confidence": self.confidence,
            "rule": self.rule,
        }


def classify_pokemon(card: dict) -> TypeLabel:
    """Map one PokemonTCG.io card record to a routing type.

    Ordered most-specific first, and each branch records WHICH rule fired so a
    disagreement with reality can be traced to a rule rather than to the whole
    mapping.
    """
    name = card.get("name", "?")
    cid = card.get("id", "?")
    rarity = card.get("rarity")
    supertype = card.get("supertype")
    subtypes = tuple(card.get("subtypes") or ())
    images = card.get("images") or {}
    url = images.get("large") or images.get("small")
    r = (rarity or "").strip().lower()
    subs = {s.strip().lower() for s in subtypes}

    if supertype == "Energy" and "special" not in subs:
        return TypeLabel(cid, name, CardType.ENERGY, rarity, supertype, subtypes, url,
                         "exact", "supertype == Energy")

    if r in _RARITY_FULL_BLEED:
        return TypeLabel(cid, name, CardType.FULL_BLEED, rarity, supertype, subtypes, url,
                         "exact", f"rarity '{rarity}' is a full-art tier")

    # A V/VMAX/VSTAR at an unusual rarity is still normally full-art.
    if subs & _SUBTYPE_FULL_BLEED and "ex" not in subs:
        return TypeLabel(cid, name, CardType.FULL_BLEED, rarity, supertype, subtypes, url,
                         "inferred", f"subtype {sorted(subs & _SUBTYPE_FULL_BLEED)} implies full-art")

    if r in _RARITY_BORDERED:
        return TypeLabel(cid, name, CardType.BORDERED, rarity, supertype, subtypes, url,
                         "exact", f"rarity '{rarity}' is a bordered tier")

    if not rarity:
        return TypeLabel(cid, name, CardType.UNKNOWN, rarity, supertype, subtypes, url,
                         "guess", "no rarity field")

    # An unrecognised rarity is usually a new premium tier, which is usually
    # full-art -- but mark it a guess so it can be reviewed rather than trusted.
    return TypeLabel(cid, name, CardType.UNKNOWN, rarity, supertype, subtypes, url,
                     "guess", f"rarity '{rarity}' not in the mapping")


def classify_scryfall(card: dict) -> TypeLabel:
    """Map one Scryfall card record. Scryfall states full_art directly."""
    name = card.get("name", "?")
    cid = card.get("id", "?")
    imgs = card.get("image_uris") or {}
    url = imgs.get("normal") or imgs.get("small")
    subs: tuple[str, ...] = tuple(card.get("frame_effects") or ())

    if card.get("full_art"):
        return TypeLabel(cid, name, CardType.FULL_BLEED, card.get("rarity"), "magic", subs, url,
                         "exact", "full_art flag set")
    if card.get("textless"):
        return TypeLabel(cid, name, CardType.FULL_BLEED, card.get("rarity"), "magic", subs, url,
                         "exact", "textless")
    if card.get("border_color") == "borderless":
        return TypeLabel(cid, name, CardType.FULL_BLEED, card.get("rarity"), "magic", subs, url,
                         "exact", "border_color == borderless")
    return TypeLabel(cid, name, CardType.BORDERED, card.get("rarity"), "magic", subs, url,
                     "exact", f"border_color == {card.get('border_color')}")


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _get(url: str, tries: int = 3, pause: float = 1.0) -> dict:
    """GET with backoff. Both APIs rate-limit and return 500 under load."""
    last: Optional[Exception] = None
    for i in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read())
            time.sleep(pause)
            return out
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(2.5 * (i + 1))
    raise RuntimeError(f"request failed after {tries} tries: {last}")


def fetch_pokemon_set(set_id: str, page_size: int = 250) -> list[TypeLabel]:
    """Label an entire Pokemon set. One request per 250 cards."""
    out: list[TypeLabel] = []
    page = 1
    while True:
        d = _get(
            "https://api.pokemontcg.io/v2/cards"
            f"?q=set.id:{urllib.parse.quote(set_id)}&pageSize={page_size}&page={page}"
        )
        cards = d.get("data") or []
        out.extend(classify_pokemon(c) for c in cards)
        if len(cards) < page_size:
            break
        page += 1
    return out


def build_training_manifest(
    set_ids: Iterable[str], path: str = "card_types.json"
) -> dict:
    """Produce a labelled manifest: image URL -> type, ready for training.

    Images are referenced by URL rather than downloaded, so the manifest is small
    and re-runnable. Catalogue scans are flat, evenly lit and cropped -- which is
    the DOMAIN GAP this manifest carries: a model trained on them has not seen an
    oblique handheld photograph with a caliper in frame. Train with aggressive
    perspective, lighting and blur augmentation, then evaluate on real photos.
    """
    labels: list[TypeLabel] = []
    failures: list[str] = []
    for sid in set_ids:
        try:
            labels.extend(fetch_pokemon_set(sid))
        except RuntimeError as exc:
            failures.append(f"{sid}: {exc}")

    counts: dict[str, int] = {}
    conf: dict[str, int] = {}
    for t in labels:
        counts[t.card_type.value] = counts.get(t.card_type.value, 0) + 1
        conf[t.confidence] = conf.get(t.confidence, 0) + 1

    manifest = {
        "generated_at": time.time(),
        "n": len(labels),
        "counts": counts,
        "confidence": conf,
        "sets_failed": failures,
        "domain_note": (
            "Labels are exact catalogue metadata. Images are clean scans; real "
            "captures are oblique, unevenly lit and may contain a caliper. Train "
            "with heavy augmentation and evaluate on real photographs."
        ),
        "rows": [t.as_row() for t in labels],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


# ---------------------------------------------------------------------------
# Verification -- the step that decides whether the mapping is usable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Agreement:
    n: int
    agree: int
    disagree: list[tuple[str, str, str]]  # (card, predicted, observed)

    @property
    def rate(self) -> float:
        return self.agree / self.n if self.n else 0.0

    def describe(self) -> str:
        lines = [f"catalogue label vs measured capability: {self.agree}/{self.n} agree "
                 f"({100 * self.rate:.0f}%)"]
        if self.disagree:
            lines.append("disagreements (each is a rule to fix OR a detector bug):")
            for name, pred, obs in self.disagree[:12]:
                lines.append(f"  {name:<28} catalogue={pred:<12} measured={obs}")
        return "\n".join(lines)


def verify_against_capability(
    pairs: list[tuple[str, CardType, str]]
) -> Agreement:
    """Check catalogue labels against what the pipeline actually measured.

    ``pairs`` is (card_name, catalogue_type, observed_capability). A bordered
    card that measures GEOMETRY_ONLY means either the rarity mapping is wrong or
    border detection failed -- and you want to know which BEFORE training on
    100k catalogue labels, because a systematic mapping error would be learned
    faithfully and silently.
    """
    agree = 0
    bad: list[tuple[str, str, str]] = []
    for name, ctype, observed in pairs:
        expected = ctype.expected_capability
        ok = (
            observed == expected
            or (expected == "full" and observed in ("full", "single_axis", "partial"))
            or (expected == "partial" and observed in ("partial", "geometry_only"))
        )
        if ok:
            agree += 1
        else:
            bad.append((name, ctype.value, observed))
    return Agreement(len(pairs), agree, bad)
