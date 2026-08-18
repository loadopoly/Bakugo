"""Reading the collector number -- when the pixels actually contain it.

WHY A GENERIC OCR PIPELINE IS NOT THE ANSWER HERE
--------------------------------------------------
The GeoGraph OCR node reads historical documents by sending images to Gemini
2.5 Flash. That is a good design for its problem: archival scans where text is
50-100 px tall, where the answer is open-vocabulary, and where a plausible
reading beats no reading.

This problem is the opposite on all three counts.

  1. RESOLUTION. A collector number is ~1.5 mm of print. Through display glass
     at 1x it is 6-12 px tall. No OCR engine reads text that is not in the
     pixels; asking a vision model to try produces a *confident plausible*
     answer rather than a refusal, and a confident wrong printing is exactly
     the failure this project exists to avoid. Being able to call a better OCR
     does not raise the information content of the image.

  2. VOCABULARY. We are not reading arbitrary text. We are choosing among the N
     collector numbers that actually exist for this card. That is a closed
     vocabulary, and it is a much stronger constraint than any engine's
     language model -- it is the "relationality" that makes this tractable.

  3. FAILURE COST. In archival digitisation a wrong transcription is a data
     quality issue. Here it selects a $1.20 printing instead of a $3,521 one.

So this module does three things in order, and the first two matter more than
the third:

  GATE      refuse outright when px/mm cannot support the glyph height
  READ      Tesseract, restricted to digits and a card-number charset
  CONSTRAIN require the reading to match a real catalog candidate, uniquely,
            within a small edit distance -- otherwise report ambiguity

The constraint step is what converts a fallible reading into a safe one. An OCR
output of "2S3" is not a number; matched against the candidate list it resolves
to "263" unambiguously. An output of "1" against candidates {1, 11, 111} is
ambiguous and stays ambiguous.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import cv2
import numpy as np

from .catalog import FEATURE_SIZE_MM, MARGINAL_PX_TO_READ, MIN_PX_TO_READ, CatalogEntry

# Tesseract's own confidence below this is not worth constraining against.
MIN_ENGINE_CONFIDENCE = 40.0
# Maximum edit distance allowed when snapping a reading to a catalog entry.
MAX_SNAP_DISTANCE = 1
# A QUIPU prior must be at least this many times the runner-up to break a tie.
QUIPU_TIE_DOMINANCE = 2.0


def _quipu_tie_break(tied: Sequence[str]) -> Optional[str]:
    """Pick among tied catalog numbers using QUIPU cross-corpus priors.

    Returns the dominant candidate, or None when the Observer is absent,
    silent about these numbers, or not clearly decisive.
    """
    try:
        from .quipu_client import enabled, number_priors

        if not enabled():
            return None
        priors = number_priors()
        if not priors:
            return None
        weighted = sorted(
            ((priors.get(n.lower(), 0.0), n) for n in tied), reverse=True
        )
        best_w, best_n = weighted[0]
        runner_w = weighted[1][0] if len(weighted) > 1 else 0.0
        if best_w > 0.0 and best_w >= QUIPU_TIE_DOMINANCE * max(runner_w, 1e-9):
            return best_n
    except Exception:  # pragma: no cover - the Observer is always optional
        pass
    return None


def _quipu_report(resolved: str, raw: str, corrected: bool) -> None:
    """Feed a resolved reading back to the Observer (fire-and-forget)."""
    try:
        from .quipu_client import enabled, feedback_async, observe_async

        if not enabled():
            return
        observe_async(f"collector number {resolved}")
        if corrected:
            feedback_async(expected=resolved, observed=raw)
    except Exception:  # pragma: no cover - the Observer is always optional
        pass


class OcrUnavailable(RuntimeError):
    """No OCR engine, or the image cannot support one."""


@dataclass(frozen=True)
class OcrReading:
    text: str
    confidence: float
    engine: str


class OcrEngine(Protocol):
    def read(self, image: np.ndarray, charset: str) -> OcrReading: ...
    @property
    def name(self) -> str: ...


@dataclass
class TesseractEngine:
    """Local Tesseract. No network, no API key, no hallucination.

    Tesseract fails by returning garbage or nothing, which the catalog
    constraint then rejects. A vision LLM fails by returning something
    plausible, which the constraint may well accept. For this task the
    dumber engine is the safer one.
    """

    psm: int = 7  # treat the crop as a single text line

    @property
    def name(self) -> str:
        return "tesseract"

    def read(self, image: np.ndarray, charset: str = "0123456789") -> OcrReading:
        if shutil.which("tesseract") is None:
            raise OcrUnavailable("tesseract is not installed")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "crop.png"
            cv2.imwrite(str(path), image)
            cmd = [
                "tesseract",
                str(path),
                "stdout",
                "--psm",
                str(self.psm),
                "-c",
                f"tessedit_char_whitelist={charset}",
                "-c",
                "debug_file=/dev/null",
            ]
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=25, check=False
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                raise OcrUnavailable(f"tesseract failed: {exc}") from exc
        text = "".join(out.stdout.split())
        return OcrReading(text=text, confidence=100.0 if text else 0.0, engine=self.name)


def preprocess_number_crop(crop: np.ndarray, upscale: int = 4) -> np.ndarray:
    """Prepare a small text crop for OCR.

    Upscaling does not add information -- it cannot, the information is bounded
    by the capture -- but Tesseract is trained on print-resolution glyphs and
    performs materially better when the input is in that regime. The resolution
    gate, not this function, is what decides whether the reading is meaningful.
    """
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    gray = cv2.resize(
        gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.bilateralFilter(gray, 7, 45, 45)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Card numbers are printed light-on-dark about as often as dark-on-light.
    if float((binary == 255).mean()) < 0.5:
        binary = cv2.bitwise_not(binary)
    return cv2.copyMakeBorder(binary, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class NumberResult:
    """Outcome of trying to read a collector number."""

    reading: Optional[str]
    snapped_to: Optional[str]
    candidates_considered: int
    px_per_mm: float
    glyph_px: float
    gated_out: bool
    ambiguous_matches: tuple[str, ...]
    engine: str
    warnings: tuple[str, ...]

    @property
    def resolved(self) -> bool:
        return self.snapped_to is not None

    def describe(self) -> str:
        if self.gated_out:
            return (
                f"collector number NOT ATTEMPTED: {self.glyph_px:.0f} px glyph "
                f"height at {self.px_per_mm:.1f} px/mm is below the "
                f"{MIN_PX_TO_READ:.0f} px floor. The digits are not in the image."
            )
        if self.resolved:
            return (
                f"collector number: {self.snapped_to}"
                + (
                    f"  (read '{self.reading}', snapped to catalog)"
                    if self.reading != self.snapped_to
                    else ""
                )
            )
        if self.ambiguous_matches:
            return (
                f"read '{self.reading}' but it matches "
                f"{len(self.ambiguous_matches)} catalog entries "
                f"({', '.join(self.ambiguous_matches[:5])}); left unresolved"
            )
        return f"read '{self.reading or ''}' which matches no catalog entry; left unresolved"


def read_collector_number(
    number_crop: np.ndarray,
    px_per_mm: float,
    candidates: Sequence[CatalogEntry],
    engine: Optional[OcrEngine] = None,
    max_snap_distance: int = MAX_SNAP_DISTANCE,
) -> NumberResult:
    """Read a collector number and snap it to a real catalog entry.

    Returns unresolved rather than guessing whenever the gate fails, the engine
    returns nothing usable, or the reading is compatible with more than one
    real printing.
    """
    glyph_px = FEATURE_SIZE_MM["collector_number"] * px_per_mm
    warnings: list[str] = []

    if glyph_px < MIN_PX_TO_READ:
        return NumberResult(
            reading=None,
            snapped_to=None,
            candidates_considered=len(candidates),
            px_per_mm=px_per_mm,
            glyph_px=glyph_px,
            gated_out=True,
            ambiguous_matches=(),
            engine="none (gated)",
            warnings=(
                "OCR was not attempted. Running it anyway would return a "
                "plausible number rather than a refusal, which is worse than "
                "reporting ambiguity.",
            ),
        )

    engine = engine or TesseractEngine()
    prepared = preprocess_number_crop(number_crop)
    # Constrain the engine's alphabet to the characters that actually appear in
    # this card's real collector numbers. If every candidate is numeric there is
    # no reason to let the engine consider letters, and every letter it can emit
    # is another way to produce a wrong reading. This is the same closed-
    # vocabulary idea as the snap step, applied one stage earlier.
    charset = "".join(sorted({c for e in candidates for c in e.collector_number}))
    if not charset:
        charset = "0123456789"
    try:
        reading = engine.read(prepared, charset=charset)
    except OcrUnavailable as exc:
        return NumberResult(
            reading=None,
            snapped_to=None,
            candidates_considered=len(candidates),
            px_per_mm=px_per_mm,
            glyph_px=glyph_px,
            gated_out=False,
            ambiguous_matches=(),
            engine="unavailable",
            warnings=(str(exc),),
        )

    raw = re.sub(r"[^0-9a-zA-Z]", "", reading.text).lower()
    if not raw:
        return NumberResult(
            reading=None,
            snapped_to=None,
            candidates_considered=len(candidates),
            px_per_mm=px_per_mm,
            glyph_px=glyph_px,
            gated_out=False,
            ambiguous_matches=(),
            engine=reading.engine,
            warnings=("the engine returned nothing legible",),
        )

    # The closed-vocabulary constraint. The answer must be one of the collector
    # numbers this card actually has, so a reading is only accepted when it is
    # closer to exactly one of them than to any other.
    numbers = sorted({e.collector_number.lower() for e in candidates})
    if not numbers:
        return NumberResult(
            reading=raw,
            snapped_to=None,
            candidates_considered=0,
            px_per_mm=px_per_mm,
            glyph_px=glyph_px,
            gated_out=False,
            ambiguous_matches=(),
            engine=reading.engine,
            warnings=("no catalog candidates to constrain the reading against",),
        )

    scored = sorted((levenshtein(raw, n), n) for n in numbers)
    best_distance = scored[0][0]
    tied = tuple(n for d, n in scored if d == best_distance)

    if best_distance > max_snap_distance:
        warnings.append(
            f"closest catalog number is '{tied[0]}' at edit distance "
            f"{best_distance}, beyond the snap limit. Treating as unread rather "
            "than forcing a match."
        )
        return NumberResult(
            reading=raw,
            snapped_to=None,
            candidates_considered=len(numbers),
            px_per_mm=px_per_mm,
            glyph_px=glyph_px,
            gated_out=False,
            ambiguous_matches=(),
            engine=reading.engine,
            warnings=tuple(warnings),
        )

    if len(tied) > 1:
        # QUIPU Observer tie-break: the mesh's cross-corpus numeric priors
        # (numbers seen by Loadopoly-OCR's unstructured scans and by prior
        # Bakugo sessions) can separate printings the pixels alone cannot.
        # The closed catalog vocabulary still bounds the answer; the prior
        # only chooses AMONG real candidates, and only when it clearly
        # dominates. Absent or flat priors leave the reading ambiguous.
        quipu_pick = _quipu_tie_break(tied)
        if quipu_pick is not None:
            warnings.append(
                f"reading '{raw}' tied between {', '.join(tied)}; resolved to "
                f"'{quipu_pick}' by QUIPU cross-corpus prior. Verify before "
                "trusting it on a high-value card."
            )
            tied = (quipu_pick,)
        else:
            warnings.append(
                "the reading is equally close to several real printings, so it does "
                "not disambiguate them"
            )
            return NumberResult(
                reading=raw,
                snapped_to=None,
                candidates_considered=len(numbers),
                px_per_mm=px_per_mm,
                glyph_px=glyph_px,
                gated_out=False,
                ambiguous_matches=tied,
                engine=reading.engine,
                warnings=tuple(warnings),
            )

    if best_distance > 0:
        warnings.append(
            f"reading '{raw}' corrected to '{tied[0]}' using the catalog. Verify "
            "before trusting it on a high-value card."
        )
    if glyph_px < MARGINAL_PX_TO_READ:
        warnings.append(
            f"glyph height {glyph_px:.0f}px is in the marginal band "
            f"({MIN_PX_TO_READ:.0f}-{MARGINAL_PX_TO_READ:.0f}px). Measured "
            "accuracy here is high but not certain, and errors present as a "
            "confident wrong printing. Verify on anything valuable."
        )

    _quipu_report(resolved=tied[0], raw=raw, corrected=best_distance > 0)

    return NumberResult(
        reading=raw,
        snapped_to=tied[0],
        candidates_considered=len(numbers),
        px_per_mm=px_per_mm,
        glyph_px=glyph_px,
        gated_out=False,
        ambiguous_matches=(),
        engine=reading.engine,
        warnings=tuple(warnings),
    )


def number_region(rect_bgr: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Crop the bottom-left corner, where collector numbers usually sit.

    A heuristic, and a fragile one across games and eras. It is deliberately
    generous, because a crop that misses the number costs a refusal while a crop
    that includes neighbouring text costs a wrong reading.
    """
    h, w = rect_bgr.shape[:2]
    y0 = int(h * 0.90)
    y1 = h
    x0 = int(w * 0.04)
    x1 = int(w * 0.46)
    return rect_bgr[y0:y1, x0:x1]
