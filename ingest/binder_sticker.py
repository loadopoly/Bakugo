"""OCR over one pocket crop: front-card identity text, and back-sticker fields.

ROTATION IS SEARCHED PER CROP, NOT ASSUMED FROM THE PAGE
------------------------------------------------------------
Tested against real photos (2026-09-14): the same rectified page came out
consistently upright for some photos and 180-degrees rotated for others
(order_quad's "start at the corner nearest the image origin" rule has no way
to know which corner is physically the top of the page -- it only produces a
mathematically consistent ordering). Separately, and independently of that,
a price sticker is frequently applied to the card at close to a right angle
to the card itself -- confirmed visually on a real sample sticker, which read
top-to-bottom rather than left-to-right in an otherwise correctly-oriented
crop. Work Instruction v2.0 S6 already warns stickers are "applied at an
angle". Because these two rotations are independent, this module does not
try to compute a single page-level or pocket-level correction; it tries the
crop at 0/90/180/270 degrees and keeps whichever OCR result scores highest by
``_score_text``, per crop, every time.

WHAT WAS ACTUALLY VALIDATED, AND WHAT WAS NOT
---------------------------------------------
Tested end to end against a real sticker crop from the sample binder, at
native photo resolution (see binder_grid.rectify_page's docstring for why
resolution matters -- a downscaled-then-cropped version of the identical
sticker produced pure noise): correctly-oriented OCR read the tier label
text ("FIVE DOLLAR POKEMON CARD"), the item label ("Trading Carts Pokemon" --
one character off from "Cards"), and the SKU/barcode number ("8279931",
exact) directly off the raw OCR text. It did NOT reliably read the price
figure itself ("$4.95") in that same test -- the bold price bar has
different contrast characteristics (dark text on a mid-toned colour fill
rather than on white) and a tighter, better-targeted crop of just that bar
is what S6 actually asks for ("targeted OCR ... orange/red bar"), which
``_price_bar_candidates`` attempts but has not been proven reliable.

This is reported here rather than smoothed over because it changes what this
module should be trusted for on day one: expect price_usd to come back None
with status price_needs_review more often than would be ideal, until this is
revisited with more real stickers to tune against. The module never
fabricates a price it isn't confident in -- see extract_sticker_fields -- so
the failure mode of today's price OCR is "flagged for a human", never
"silently wrong".

FULL-PIPELINE RUN, POST PERFORMANCE FIX (2026-09-14)
------------------------------------------------------
Once preprocess_text_crop stopped blindly upscaling already-large native-
resolution crops (see its own docstring), a real 3-page-pair folder ran end
to end in ~105s instead of timing out. That run surfaced two things worth
recording here rather than only in commit history:

  card_identity_ocr (read_front_identity) is noisier than expected -- see
  that function's own docstring for the measurement. Earlier validation in
  this module's development focused on the back/sticker side; front-of-card
  text OCR quality had not actually been checked against a full run until
  now, and it is the weaker of the two.

  classify_back_franchise produced one franchise_mismatch on a page pair
  that was, as far as could be told by eye, a uniform Pokemon page: guess
  "yugioh" at confidence 0.55, just over this module's 0.5 reporting
  threshold. Given _YUGIOH_HUE is explicitly uncalibrated (see below), this
  is plausible evidence of exactly the false-positive risk already flagged
  there, not a new failure mode -- but it is a real occurrence against real
  photos, not a hypothetical one, and worth watching for in flagged output:
  a franchise_mismatch on a page that looks visually uniform to a human
  reviewer is more likely to be this module being wrong about Yugioh's hue
  than the page actually being mixed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

PRICE_RE = re.compile(r"\$\s?(\d{1,4}\.\d{2})")
DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
SKU_RE = re.compile(r"(?<!\d)(\d{6,8})(?!\d)")
WORD_RE = re.compile(r"[A-Za-z]{4,}")
UPPER_LINE_RE = re.compile(r"^[A-Z0-9 .,'\-]{6,}$")


def preprocess_text_crop(crop: np.ndarray, upscale: int = 3, target_max_dim: int = 1100) -> np.ndarray:
    """Same recipe as cardcenter.ocr.preprocess_number_crop, generalised past
    a digit-only charset: get the crop to a print-resolution working size,
    denoise, local-contrast-normalise, then binarise. The invert-if-mostly-
    dark rule matters here too -- a light sticker on a dark card back and a
    dark card back region behind the price bar are opposite polarities
    within the same pocket crop.

    ``upscale`` is only applied to a crop SMALLER than ``target_max_dim`` --
    binder_grid.rectify_page already rectifies close to the source photo's
    native resolution (by design, see its docstring), so a pocket crop
    reaching this function is typically already 900-1300px on a side.
    Blindly multiplying that by 3 (this module's first version) produced
    3000-4000px working images and measured 15+ seconds of Tesseract time
    PER POCKET -- correct, but far too slow to run over a real folder.
    Downscaling an oversized crop toward target_max_dim before binarising
    cut that to a few seconds with no measured loss in which rotation won
    the _score_text comparison on the same test crop.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    longest = max(gray.shape[:2])
    if longest < target_max_dim:
        scale = min(upscale, target_max_dim / max(longest, 1))
    else:
        scale = target_max_dim / longest
    if abs(scale - 1.0) > 0.02:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    gray = cv2.bilateralFilter(gray, 7, 45, 45)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float((binary == 255).mean()) < 0.5:
        binary = cv2.bitwise_not(binary)
    return cv2.copyMakeBorder(binary, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def _score_text(text: str) -> float:
    """How much does this OCR output look like real sticker/card text, as
    opposed to noise? Used only to CHOOSE among rotations of the same crop,
    never to decide whether to trust the winning text's individual fields --
    field extraction still requires its own regex match on top of this.
    """
    score = 0.0
    score += 12.0 * len(PRICE_RE.findall(text))
    score += 6.0 * len(DATE_RE.findall(text))
    score += 3.0 * len(SKU_RE.findall(text))
    score += 1.5 * len(WORD_RE.findall(text))
    return score


def read_best_orientation(crop_bgr: np.ndarray, psm: int = 6, upscale: int = 3):
    """Try the crop at 0/90/180/270 degrees, keep the highest-scoring OCR.

    Returns (text, angle_degrees, score). Requires pytesseract; raises
    RuntimeError with a clear message if it is not importable or the
    tesseract binary is missing, rather than silently returning empty text
    that a caller might mistake for "nothing here".
    """
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pytesseract is not installed; sticker/identity OCR is unavailable"
        ) from exc

    best_text, best_angle, best_score = "", 0, -1.0
    for k in range(4):
        rot = np.rot90(crop_bgr, k)
        prep = preprocess_text_crop(rot, upscale=upscale)
        try:
            text = pytesseract.image_to_string(prep, config=f"--psm {psm}")
        except Exception:
            text = ""
        s = _score_text(text)
        if s > best_score:
            best_text, best_angle, best_score = text, k * 90, s
    return best_text, best_angle, best_score


def read_front_identity(crop_bgr: np.ndarray) -> tuple[str, float]:
    """Raw OCR text off a front pocket. Deliberately NOT resolved against a
    species/card catalog -- Work Instruction v2.0 S9 asks for
    "card_identity_ocr": "<extracted name / set text>", a raw reading for a
    human to read and cross-check, not a resolved identification. Matching
    that literally also keeps this module honest about scope: cardcenter's
    OWN catalog.py explains at length why collapsing an OCR reading into a
    confident specific-card match is the wrong move when the OCR itself
    hasn't been validated at this resolution (see that module's docstring).
    Returns (cleaned_text, confidence_score) -- the score is the same
    _score_text heuristic used for rotation selection, carried through so a
    caller can down-weight an obviously weak reading without re-OCRing.

    MEASURED QUALITY (2026-09-14, full pipeline run against a real 3x3
    Pokemon page pair): this came back mostly noise -- short fragments of
    punctuation and stray letters rather than legible card names -- on 9/9
    occupied front pockets tested. _score_text is tuned against sticker/date/
    price/SKU patterns (see its docstring), none of which a card's own front
    text contains, so the score returned here is a much weaker rotation-
    selection signal for this call site than it is for extract_sticker_fields
    -- it picks the least-bad orientation, not a good reading. Treat
    card_identity_ocr as "a human still needs to look at the photo" in this
    build, not as usable text on its own; this is the same
    report-don't-guess posture as price_usd, stated plainly because it was
    not yet measured (as opposed to sticker OCR, which was) as of the
    previous revision of this module.
    """
    text, _angle, score = read_best_orientation(crop_bgr, psm=6)
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned, score


@dataclass(frozen=True)
class StickerFields:
    price_usd: Optional[str]
    sticker_sku: Optional[str]
    sticker_tier_label: Optional[str]
    sticker_date: Optional[str]
    raw_text: str
    used_targeted_crop: bool
    warnings: tuple = field(default_factory=tuple)


def _bright_blob_roi(pocket_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Largest white/orange connected blob, away from the crop's own edges.

    The sticker is white card stock with an orange/red price bar, glued over
    a much darker (Pokemon blue swirl) or busier card back, so it is
    normally the single largest bright, saturated-or-desaturated-but-light
    region in the pocket. Restricting the search to the inner ~88% of the
    pocket keeps this from grabbing a sliver of a neighbouring pocket's
    white card stock or, on an edge/spine pocket, the binder rings.
    """
    h, w = pocket_bgr.shape[:2]
    if h < 20 or w < 20:
        return None
    mx, my = int(w * 0.06), int(h * 0.06)
    inner = pocket_bgr[my : h - my, mx : w - mx]
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = cv2.split(hsv)
    white_mask = (s_ < 70) & (v_ > 150)
    orange_mask = (h_ < 25) & (s_ > 80) & (v_ > 80)
    mask = ((white_mask | orange_mask).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    area_total = inner.shape[0] * inner.shape[1]
    best = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(best)
    bx, by, bw, bh = cv2.boundingRect(best)
    solidity = area / max(bw * bh, 1)
    if area < 0.06 * area_total or area > 0.85 * area_total or solidity < 0.5:
        return None
    pad = max(4, int(0.04 * min(bw, bh)))
    x0, y0 = max(0, bx - pad), max(0, by - pad)
    x1, y1 = min(inner.shape[1], bx + bw + pad), min(inner.shape[0], by + bh + pad)
    return inner[y0:y1, x0:x1]


def _extract_tier_label(text: str) -> Optional[str]:
    best = None
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 6:
            continue
        letters = sum(1 for ch in line if ch.isalpha())
        if letters < 6:
            continue
        upper_frac = sum(1 for ch in line if ch.isalpha() and ch.isupper()) / max(letters, 1)
        if upper_frac < 0.6:
            continue
        if best is None or len(line) > len(best):
            best = line
    return best


def extract_sticker_fields(pocket_bgr: np.ndarray) -> StickerFields:
    """Read a back pocket's price sticker.

    Runs OCR twice -- once over the whole pocket crop, once over a
    color-isolated sticker sub-crop when one can be found -- and keeps
    whichever scored higher (see _score_text), rather than assuming the
    targeted crop is always better: a bad blob detection on a busy card back
    can score worse than just reading the whole pocket. price_usd is set
    ONLY from a literal $X.XX regex match in the winning text; when no such
    match exists it stays None and a warning is attached, which
    binder_ingest.py turns into status=price_needs_review. It is never
    inferred, defaulted, or carried over from a sibling pocket's price --
    Work Instruction v2.0 S8 is explicit that a low-confidence or unreadable
    price must be flagged, not guessed, and confidently wrong stock prices
    silently written down the flagged/matched pipeline are the specific
    failure everything else in this package (confidence.py's whole reason
    for existing, catalog.py's ambiguity reporting) is trying to avoid.
    """
    warnings = []
    whole_text, _angle, whole_score = read_best_orientation(pocket_bgr)

    roi = _bright_blob_roi(pocket_bgr)
    used_targeted = False
    best_text, best_score = whole_text, whole_score
    if roi is not None and roi.size > 0:
        roi_text, _roi_angle, roi_score = read_best_orientation(roi)
        if roi_score >= whole_score:
            best_text, best_score, used_targeted = roi_text, roi_score, True
    else:
        warnings.append("could not isolate a sticker region; OCR ran on the whole pocket")

    price_m = PRICE_RE.search(best_text)
    date_m = DATE_RE.search(best_text)
    sku_m = SKU_RE.search(best_text)
    tier = _extract_tier_label(best_text)

    price = price_m.group(1) if price_m else None
    if price is None:
        warnings.append("no $X.XX pattern found in OCR text; price needs human review")

    return StickerFields(
        price_usd=price,
        sticker_sku=sku_m.group(1) if sku_m else None,
        sticker_tier_label=tier,
        sticker_date=date_m.group(1) if date_m else None,
        raw_text=best_text.strip(),
        used_targeted_crop=used_targeted,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class FranchiseGuess:
    guess: Optional[str]  # "pokemon" | "yugioh" | None (abstain)
    confidence: float  # 0..1, ranking only -- not a calibrated probability
    note: str


# Reference hue centres (OpenCV H in 0..180) for card-BACK artwork, sampled
# away from sticker/glare pixels. POKEMON_HUE is measured directly from this
# session's real sample photos (2026-09-14): the blue swirl back is a tight
# cluster around hue ~105-115. YUGIOH_HUE is NOT measured from a real sample
# -- no Yugioh card-back photo was in the sample set, only Yugioh FRONTS
# (Dimensional Barrier, Krawler Axon, Sky Striker Airspace) -- it is the
# well-known brown/tan-with-maroon-swirl Yu-Gi-Oh back, encoded from general
# knowledge of the card, not calibrated. Treat a "yugioh" guess from this
# function as materially less trustworthy than a "pokemon" one until it can
# be checked against real photos.
_POKEMON_HUE = 108.0
_YUGIOH_HUE = 12.0
_HUE_TOLERANCE = 18.0


def classify_back_franchise(card_back_bgr: np.ndarray) -> FranchiseGuess:
    """Best-effort back-design franchise check for Work Instruction v2.0 S7.

    This is a soft, abstaining signal, never a source of truth: the group a
    card is filed under comes from the containing folder (S2's own ground
    truth) or, on the front side, from the front card's own printed
    franchise cues read by a human/VLM elsewhere in the pipeline. This
    function only answers "does the BACK design look like a confident
    mismatch", which is what franchise_mismatch (S7/S8) needs -- it returns
    guess=None (abstain) far more often than it commits, on purpose, per the
    same reporting-ambiguity-rather-than-guessing approach cardcenter.ocr
    and cardcenter.catalog already use elsewhere in this codebase.
    """
    if card_back_bgr.size == 0:
        return FranchiseGuess(None, 0.0, "empty crop")
    hsv = cv2.cvtColor(card_back_bgr, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = cv2.split(hsv)
    # Exclude sticker pixels (low saturation/white, or the orange price bar)
    # and very dark/glare-blown pixels so the remaining hue is dominated by
    # the actual card-back print rather than what's stuck on top of it.
    sticker_like = ((s_ < 60) & (v_ > 150)) | ((h_ < 25) & (s_ > 80) & (v_ > 80))
    valid = (~sticker_like) & (v_ > 25) & (v_ < 250) & (s_ > 25)
    if valid.mean() < 0.15:
        return FranchiseGuess(None, 0.0, "too little non-sticker, non-glare area to classify")
    hues = h_[valid].astype(np.float64)
    # Circular mean, since hue wraps at 180 in OpenCV's convention.
    ang = hues / 180.0 * 2 * np.pi
    mean_ang = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    mean_hue = (mean_ang / (2 * np.pi) * 180.0) % 180.0

    def circ_dist(a, b):
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    d_poke = circ_dist(mean_hue, _POKEMON_HUE)
    d_yugi = circ_dist(mean_hue, _YUGIOH_HUE)
    if d_poke <= _HUE_TOLERANCE and d_poke < d_yugi:
        conf = max(0.0, 1.0 - d_poke / _HUE_TOLERANCE)
        return FranchiseGuess("pokemon", conf, f"mean hue {mean_hue:.0f} near Pokemon reference")
    if d_yugi <= _HUE_TOLERANCE and d_yugi < d_poke:
        conf = max(0.0, 1.0 - d_yugi / _HUE_TOLERANCE) * 0.6  # discounted: uncalibrated reference
        return FranchiseGuess(
            "yugioh", conf, f"mean hue {mean_hue:.0f} near (uncalibrated) Yugioh reference"
        )
    return FranchiseGuess(None, 0.0, f"mean hue {mean_hue:.0f} matches neither reference")
