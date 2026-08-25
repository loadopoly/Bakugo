"""Identify WHICH card a photograph shows, when it cannot be measured.

WHY THIS EXISTS, AND WHY IT IS NOT PART OF THE MEASUREMENT PATH
---------------------------------------------------------------
Centering metrology needs the card's cut edge located to a fraction of a
millimetre, because the quantity being reported is a ratio of borders a few
millimetres wide. Measured on 162 handheld photographs of sleeved cards over a
bulk bin, that precision is simply not in the data: the located border edge
wanders about 1.3 mm from column to column through a glossy sleeve, so the
confidence gate refuses -- correctly -- on nearly all of them.

Identification is a different problem with a different error budget, and
conflating the two is what made those photographs look worthless. To read a
name you do not need the boundary to a fraction of a millimetre; you need the
glyphs to survive, and at 20-50 px/mm they do. A rectification good enough to
be useless for metrology is comfortably good enough to read "Sableye" off.

So this module deliberately does NOT depend on a precise boundary. It runs
sparse OCR over the whole rectified card and keeps whatever survives, rather
than cropping a fixed fraction of the card and hoping the layout lines up --
which is what a halo boundary breaks. Measured on the same corpus, fixed-crop
name bands landed on sleeve and background; whole-card sparse text returned
real names.

THE CLOSED VOCABULARY, WHICH IS THE WHOLE SAFETY ARGUMENT
----------------------------------------------------------
Sparse OCR over artwork returns mostly garbage -- "wes 2G i (y", "Sco A Oo a".
Accepting the best-looking token would produce a confident wrong
identification on every card, which is worse than no identification at all,
because a wrong name routes to a wrong price.

The defence is the one ``read_collector_number`` already uses: the answer must
be a member of a known set, and it is only accepted when it is closer to
exactly ONE member than to any other. Garbage tokens are far from every
species name and get dropped; a genuine name survives a few OCR errors and
snaps home. A token that sits between two real names resolves to neither.

A species name is also cross-checkable, which a collector number is not:
modern cards print the National Pokedex number ("NO. 0706") next to the
species. When both are read they must agree, and disagreement is reported
rather than silently resolved in favour of one.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .ocr import OcrUnavailable, levenshtein

_SPECIES_DATA = Path(__file__).parent / "data" / "species.json"

# Tokens shorter than this are not worth matching: at three characters almost
# any garbage IS some real name, so the closed vocabulary stops discriminating.
#
# Four rather than five, because it was measured: with the zero-edit rule in
# _allowed_edits applied to short names, dropping the floor from 5 to 4 changed
# nothing at all on the 162-photo corpus -- same 17 identifications, same 11
# species, no new false positives. The edit budget was doing the safety work,
# not the length floor, so the floor stays where it costs no coverage. Species
# with four-letter names (Seel, Abra, Onix, Mew) can therefore still be
# identified, but only from an exact reading.
MIN_TOKEN_LEN = 4

# Words that appear on essentially every card of the game. They are real words
# and would otherwise soak up edit-distance budget against species names.
_BOILERPLATE = {
    "basic", "stage", "evolves", "from", "pokemon", "pokémon", "trainer",
    "energy", "ability", "weakness", "resistance", "retreat", "damage",
    "attack", "attacks", "opponent", "opponents", "active", "bench",
    "benched", "discard", "draw", "card", "cards", "deck", "hand", "turn",
    "coin", "flip", "heads", "tails", "prize", "knocked", "illus", "nintendo",
    "creatures", "game", "freak", "supporter", "item", "tool", "special",
    "your", "this", "each", "that", "with", "when", "have", "into", "play",
    "more", "than", "then", "does", "must", "such",
}


@dataclass(frozen=True)
class Species:
    name: str
    dex: Optional[int] = None


@lru_cache(maxsize=1)
def load_species() -> tuple[Species, ...]:
    """The closed vocabulary. Replaceable -- see data/species.json."""
    if not _SPECIES_DATA.exists():
        return ()
    raw = json.loads(_SPECIES_DATA.read_text(encoding="utf-8"))
    out = []
    for item in raw.get("species", []):
        if isinstance(item, str):
            out.append(Species(name=item))
        else:
            out.append(Species(name=item["name"], dex=item.get("dex")))
    return tuple(out)


@dataclass
class SparseTextEngine:
    """Tesseract in sparse-text mode, over a whole card rather than a crop.

    ``psm 11`` finds text anywhere in the image without assuming a page layout,
    which is what a card is: a name, a number, rules text and flavour text
    scattered over artwork at several sizes.
    """

    psm: int = 11
    timeout_s: int = 40

    @property
    def name(self) -> str:
        return f"tesseract-psm{self.psm}"

    def read_text(self, image: np.ndarray) -> str:
        """Raw sparse text. Tokenising is the caller's job so the dex number,
        which needs the surrounding punctuation, survives."""
        if shutil.which("tesseract") is None:
            raise OcrUnavailable("tesseract is not installed")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "card.png"
            cv2.imwrite(str(path), image)
            cmd = [
                "tesseract", str(path), "stdout",
                "--psm", str(self.psm),
                "-c", "debug_file=/dev/null",
            ]
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=self.timeout_s, check=False,
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                raise OcrUnavailable(f"tesseract failed: {exc}") from exc
        return out.stdout


def tokenise(text: str) -> list[str]:
    """Split raw OCR output into candidate word tokens."""
    return [t for t in re.split(r"[^A-Za-zéÉ]+", text) if t]


def prepare_for_text(rect_bgr: np.ndarray, max_long_side: int = 1800) -> np.ndarray:
    """Contrast-normalise a rectified card for sparse OCR.

    Local contrast equalisation, not thresholding: a card carries text over
    artwork at wildly different local brightness -- black on silver foil, white
    on dark art -- and any single global threshold erases one of them. CLAHE
    keeps both legible without deciding which is foreground.
    """
    img = rect_bgr
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    longest = max(img.shape[:2])
    if longest > max_long_side:
        s = max_long_side / longest
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)


def read_dex_number(text: str) -> Optional[int]:
    """Pull a National Pokedex number out of raw OCR text.

    Printed as "NO. 0706" on modern cards. OCR mangles the separator freely, so
    match loosely on the digits and require the value to be a plausible dex
    entry rather than any number on the card (HP, damage, set totals).
    """
    for m in re.finditer(r"\bN[O0Q][^0-9A-Za-z]{0,3}(\d{3,4})\b", text, re.IGNORECASE):
        n = int(m.group(1))
        if 1 <= n <= 1025:
            return n
    return None


def _allowed_edits(token: str) -> int:
    """Edit budget scaled to token length -- and deliberately mean on short ones.

    The budget must shrink with length, because the number of garbage strings
    within one edit of a target does not. Sparse OCR over artwork emits a
    torrent of four-letter junk, and a four-letter species name with a one-edit
    budget accepts a huge share of it: measured on this corpus, an earlier
    version reported "Seel" 22 times from readings of 'Sees', 'seal' and
    'peel', out of 58 identifications. None of those cards was a Seel.

    Every identification that proved sound was a long name at zero edits --
    Goodra, Sableye, Genesect, Thundurus, Zacian -- so that is what the budget
    is now shaped to admit. Short names must be read exactly; only a name long
    enough that a coincidence is implausible earns any tolerance at all.
    """
    n = len(token)
    if n <= 5:
        return 0
    if n <= 8:
        return 1
    return 2


@dataclass(frozen=True)
class Recognition:
    name: Optional[str]
    dex: Optional[int]
    matched_token: Optional[str]
    edits: Optional[int]
    alternatives: tuple[str, ...]
    tokens_considered: int
    engine: str
    warnings: tuple[str, ...] = field(default=())
    # True only when a dex number was READ OFF THE CARD and agreed with the
    # name. ``dex`` alone cannot carry this: it falls back to the vocabulary's
    # number for the matched species, so a caller inspecting it would report
    # independent corroboration that never happened.
    corroborated: bool = False
    orientation: str = "upright"

    @property
    def resolved(self) -> bool:
        return self.name is not None

    def describe(self) -> str:
        if not self.resolved:
            why = self.warnings[0] if self.warnings else "no token matched a known species"
            return f"UNIDENTIFIED -- {why}"
        bits = [f"{self.name}"]
        if self.dex is not None:
            bits.append(f"(dex {self.dex})")
        bits.append(f"from OCR {self.matched_token!r} at {self.edits} edit(s)")
        line = "  ".join(bits)
        for w in self.warnings:
            line += f"\n  WARNING: {w}"
        return line


def match_species(
    tokens: Sequence[str], vocabulary: Sequence[Species]
) -> tuple[Optional[Species], Optional[str], Optional[int], tuple[str, ...]]:
    """Snap OCR tokens to at most one species name.

    Returns (species, token, edits, alternatives). A token that is equally
    close to two different species resolves to neither: that is the same
    "closer to exactly one than to any other" rule the collector-number reader
    uses, and it is what keeps garbage from becoming a confident answer.
    """
    best: Optional[tuple[int, int, Species, str]] = None  # (edits, -len, sp, token)
    ties: set[str] = set()

    for token in tokens:
        low = token.lower()
        if len(low) < MIN_TOKEN_LEN or low in _BOILERPLATE:
            continue
        budget = _allowed_edits(low)
        for sp in vocabulary:
            d = levenshtein(low, sp.name.lower())
            if d > budget:
                continue
            key = (d, -len(low), sp, token)
            if best is None or (d, -len(low)) < (best[0], best[1]):
                best = key
                ties = {sp.name}
            elif (d, -len(low)) == (best[0], best[1]) and sp.name != best[2].name:
                ties.add(sp.name)

    if best is None:
        return None, None, None, ()
    if len(ties) > 1:
        return None, best[3], best[0], tuple(sorted(ties))
    return best[2], best[3], best[0], ()


def recognise_card(
    rect_bgr: np.ndarray,
    vocabulary: Optional[Sequence[Species]] = None,
    engine: Optional[SparseTextEngine] = None,
) -> Recognition:
    """Read a rectified card and name the species, or refuse.

    Deliberately tolerant of an imprecise boundary: see the module docstring.
    """
    vocab = list(vocabulary) if vocabulary is not None else list(load_species())
    engine = engine or SparseTextEngine()
    if not vocab:
        return Recognition(
            name=None, dex=None, matched_token=None, edits=None,
            alternatives=(), tokens_considered=0, engine=engine.name,
            warnings=(
                "the species vocabulary is empty, so nothing can be "
                "identified. Populate cardcenter/data/species.json.",
            ),
        )

    prepared = prepare_for_text(rect_bgr)

    # A card photographed upside down still has a perfectly readable name; the
    # engine just cannot read it. enforce_portrait fixes a 90-degree rotation
    # but says nothing about which end is up, and on the corpus several cards
    # resolved only after a half turn. Try upright first and rotate only if
    # that fails, so the common case costs nothing.
    attempts = [("upright", prepared)]
    orientation = "upright"
    raw_text = ""
    try:
        raw_text = engine.read_text(prepared)
        if match_species(tokenise(raw_text), vocab)[0] is None:
            flipped = cv2.rotate(prepared, cv2.ROTATE_180)
            flipped_text = engine.read_text(flipped)
            if match_species(tokenise(flipped_text), vocab)[0] is not None:
                raw_text, orientation = flipped_text, "rotated 180"
    except OcrUnavailable as exc:
        return Recognition(
            name=None, dex=None, matched_token=None, edits=None,
            alternatives=(), tokens_considered=0, engine="unavailable",
            warnings=(str(exc),),
        )

    tokens = tokenise(raw_text)
    dex = read_dex_number(raw_text)
    sp, token, edits, alts = match_species(tokens, vocab)

    warnings: list[str] = []
    if sp is None and alts:
        warnings.append(
            f"OCR token {token!r} is equally close to {', '.join(alts)}; "
            "refusing rather than picking one."
        )
    elif sp is None:
        warnings.append(
            "no OCR token was close enough to a known species name. The card "
            "may be outside the vocabulary, face-down, or too glared to read."
        )

    # Cross-check: a printed dex number and a read name must agree.
    if sp is not None and dex is not None and sp.dex is not None and sp.dex != dex:
        warnings.append(
            f"the name reads {sp.name!r} (dex {sp.dex}) but the printed dex "
            f"number reads {dex}. These disagree, so the identification is "
            "not trustworthy -- verify by eye before pricing."
        )
        return Recognition(
            name=None, dex=dex, matched_token=token, edits=edits,
            alternatives=(sp.name,), tokens_considered=len(tokens),
            engine=engine.name, warnings=tuple(warnings),
            corroborated=False, orientation=orientation,
        )

    corroborated = (
        sp is not None and dex is not None and sp.dex is not None and sp.dex == dex
    )
    if sp is not None and not corroborated:
        warnings.append(
            "no printed dex number was read, so the name is uncorroborated."
        )

    return Recognition(
        name=sp.name if sp else None,
        dex=dex if dex is not None else (sp.dex if sp else None),
        matched_token=token,
        edits=edits,
        alternatives=alts,
        tokens_considered=len(tokens),
        engine=engine.name,
        warnings=tuple(warnings),
        corroborated=corroborated,
        orientation=orientation,
    )
