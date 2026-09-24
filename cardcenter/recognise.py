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
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .ocr import OcrUnavailable, background_run_kwargs, levenshtein

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
                    **background_run_kwargs(),
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                raise OcrUnavailable(f"tesseract failed: {exc}") from exc
        return out.stdout

    def read_words(self, image: np.ndarray) -> list["Word"]:
        """The same sparse read, with where each word sits (tesseract's TSV).
        One tesseract run, like ``read_text``."""
        if shutil.which("tesseract") is None:
            raise OcrUnavailable("tesseract is not installed")
        h, w = image.shape[:2]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "card.png"
            cv2.imwrite(str(path), image)
            cmd = [
                "tesseract", str(path), "stdout",
                "--psm", str(self.psm),
                "-c", "debug_file=/dev/null", "tsv",
            ]
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=self.timeout_s, check=False,
                    **background_run_kwargs(),
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                raise OcrUnavailable(f"tesseract failed: {exc}") from exc
        return parse_tsv(out.stdout, h)


@dataclass(frozen=True)
class Word:
    """One OCR word and where it sits: ``top``/``height`` as fractions of the
    card's height, ``line`` an id shared by the words on one text line."""

    text: str
    top: float = float("nan")
    height: float = float("nan")
    line: tuple = ()


def parse_tsv(tsv: str, image_h: int) -> list[Word]:
    """Words from tesseract's TSV output, in reading order."""
    words: list[Word] = []
    lines = tsv.splitlines()
    if not lines:
        return words
    head = lines[0].split("\t")
    try:
        col = {k: head.index(k) for k in
               ("block_num", "par_num", "line_num", "top", "height", "text")}
    except ValueError:
        return words
    h = float(max(image_h, 1))
    for row in lines[1:]:
        f = row.split("\t")
        if len(f) < len(head):
            continue
        text = f[col["text"]].strip()
        if not text:
            continue
        try:
            words.append(Word(
                text, int(f[col["top"]]) / h, int(f[col["height"]]) / h,
                (int(f[col["block_num"]]), int(f[col["par_num"]]), int(f[col["line_num"]])),
            ))
        except ValueError:
            continue
    return words


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


# The name is printed at the top of the card, in the largest type on it: the
# top 16% (14 mm of 88) holds the name bar and the "Evolves from" line under
# it on every modern layout.
NAME_BAND = 0.16


def species_key(name: str) -> str:
    """What a name is matched on: its letters and digits, lower case, accents
    dropped -- "Mr. Mime" is mrmime, "Farfetch'd" farfetchd, "Flabébé"
    flabebe. OCR splits the first two into two words; see match_words."""
    flat = unicodedata.normalize("NFKD", name)
    return "".join(c for c in flat.lower() if c.isalnum() and c.isascii())


@lru_cache(maxsize=4)
def _keyed(vocabulary: tuple) -> tuple:
    return tuple((species_key(sp.name), sp) for sp in vocabulary)


def _best(cands, vocabulary):
    """cands: (rank, token, key) -> (species, token, edits, alternatives).
    A lower rank wins before edits and length do, and a rank above 0 must
    read exactly; the tie rule is unchanged."""
    best = None   # (rank, edits, -len, species, token)
    ties: set[str] = set()
    for rank, token, key in cands:
        if len(key) < MIN_TOKEN_LEN or key in _BOILERPLATE:
            continue
        budget = _allowed_edits(key)
        for sk, sp in _keyed(tuple(vocabulary)):
            d = levenshtein(key, sk)
            if d > budget or (rank > 0 and d > 0):
                continue
            k = (rank, d, -len(key))
            if best is None or k < best[:3]:
                best = (*k, sp, token)
                ties = {sp.name}
            elif k == best[:3] and sp.name != best[3].name:
                ties.add(sp.name)
    if best is None:
        return None, None, None, ()
    if len(ties) > 1:
        return None, best[4], best[1], tuple(sorted(ties))
    return best[3], best[4], best[1], ()


def match_species(
    tokens: Sequence[str], vocabulary: Sequence[Species]
) -> tuple[Optional[Species], Optional[str], Optional[int], tuple[str, ...]]:
    """Snap OCR tokens to at most one species name.

    Returns (species, token, edits, alternatives). A token that is equally
    close to two different species resolves to neither: that is the same
    "closer to exactly one than to any other" rule the collector-number reader
    uses, and it is what keeps garbage from becoming a confident answer.
    Two neighbouring tokens are also tried joined ("Mr" "Mime").
    """
    cands = [(0, t, species_key(t)) for t in tokens]
    cands += [(0, f"{a} {b}", species_key(a + b)) for a, b in zip(tokens, tokens[1:])
              if len(species_key(a)) >= 1 and len(species_key(b)) >= 1]
    return _best(cands, vocabulary)


def match_words(
    words: Sequence[Word], vocabulary: Sequence[Species],
    name_band: tuple[float, float] = (0.0, NAME_BAND),
) -> tuple[Optional[Species], Optional[str], Optional[int], tuple[str, ...]]:
    """match_species with where each word sits on the card.

    The name is the word in the name bar; the rest of the card is rules and
    flavour text, full of ordinary words one letter from a species. A
    Meganium (2.20.0, the name not yet in the vocabulary) was named Seaking
    from "soaking" in its flavour text, at one edit. So a word below the name
    bar names the card only when it reads EXACTLY as a species (an attack that
    names one is rare, a misread that lands on one is not), a word in the bar
    beats any word below it, and the word after "from" ("Evolves from
    Bayleef") is the previous stage, never the card.

    ``name_band``: where the name bar is, as fractions of the image height --
    (0, NAME_BAND) for a card rectified edge to edge; a caller that
    rectifies with a margin (see serve._identify_payload) says so."""
    lo, hi = name_band
    cands = []
    for i, w in enumerate(words):
        prev = [x for x in words[max(0, i - 2):i] if x.line == w.line]
        if any(species_key(x.text) == "from" for x in prev):
            continue
        in_band = np.isfinite(w.top) and lo <= w.top + 0.5 * w.height <= hi
        seq = [(w.text, species_key(w.text))]
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is not None and nxt.line == w.line:
            seq.append((f"{w.text} {nxt.text}", species_key(w.text + nxt.text)))
        for token, key in seq:
            if in_band or not np.isfinite(w.top):
                cands.append((0, token, key))
            else:
                cands.append((1, token, key))
    return _best(cands, vocabulary)


def recognise_card(
    rect_bgr: np.ndarray,
    vocabulary: Optional[Sequence[Species]] = None,
    engine: Optional[SparseTextEngine] = None,
    name_band: tuple[float, float] = (0.0, NAME_BAND),
) -> Recognition:
    """Read a rectified card and name the species, or refuse.

    Deliberately tolerant of an imprecise boundary: see the module docstring.
    ``name_band`` as in match_words.
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
    def read(img) -> tuple[str, list[Word]]:
        if hasattr(engine, "read_words"):
            words = engine.read_words(img)
            return " ".join(w.text for w in words), words
        text = engine.read_text(img)
        return text, [Word(t) for t in tokenise(text)]

    def read_band(img) -> list[Word]:
        # The name bar alone, enlarged: a whole-card sparse read of a holo
        # card in a sleeve misses the name about one time in three (field
        # stills, 2.20.0) where the bar by itself at twice the size reads.
        h = img.shape[0]
        band = img[: max(8, int(round(name_band[1] * h)))]
        f = min(2.0, 2400.0 / max(band.shape[1], 1))
        band = cv2.resize(band, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        return [Word(w.text, name_band[0], 0.0, ("band",) + tuple(w.line))
                for w in engine.read_words(band)]

    orientation = "upright"
    try:
        found = None
        for label, img in (("upright", prepared),
                           ("rotated 180", cv2.rotate(prepared, cv2.ROTATE_180))):
            text, words = read(img)
            if found is None:
                found = (text, words, label)
            if match_words(words, vocab, name_band)[0] is not None:
                found = (text, words, label)
                break
            if hasattr(engine, "read_words"):
                bw = read_band(img)
                if match_words(bw, vocab, name_band)[0] is not None:
                    found = (text, bw + words, label)
                    break
        raw_text, words, orientation = found
    except OcrUnavailable as exc:
        return Recognition(
            name=None, dex=None, matched_token=None, edits=None,
            alternatives=(), tokens_considered=0, engine="unavailable",
            warnings=(str(exc),),
        )

    tokens = tokenise(raw_text)
    dex = read_dex_number(raw_text)
    sp, token, edits, alts = match_words(words, vocab, name_band)

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
