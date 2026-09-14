"""Standalone CLI: turn a folder of binder-page photos into card+price records.

Implements "Work Instruction: OCR/VLM Ingestion -- Trading Card Binder Photo
Set" v2.0 (Adam Gard, 2026-09-14) end to end:

  S3  sort files by the numeric tail of their PXL_YYYYMMDD_xxxxxxxxx name
  S4  pair each front-page photo with the NEXT photo in that sorted order
  S5  segment each page into its pocket grid and map front (row,col) to back
      (row, (cols+1)-col) -- the page flips about its left/spine edge
  S6  read the price sticker in the mapped back pocket, price as the primary
      field, SKU/tier-label/date as optional secondary fields
  S7  confirm each pocket's franchise from the card art/back design rather
      than trusting the containing folder blindly
  S8  flag edge cases (empty pockets, unpaired fronts, unreadable prices,
      grid disagreement, franchise disagreement) instead of guessing
  S9  emit one JSON record per pocket with the documented schema
  S10 the full processing checklist above, plus a separate flagged/unmatched
      log

WHAT THIS MODULE DOES NOT DO, ON PURPOSE
-----------------------------------------
It does not resolve card_identity_ocr against a species/card catalog (see
binder_sticker.read_front_identity's docstring) and it does not treat a
successful price OCR as ground truth without the regex match that backs it
(see binder_sticker.extract_sticker_fields). Both mirror decisions already
made elsewhere in cardcenter (ocr.py, catalog.py): report what was actually
read, flag what wasn't, never paper over a gap with a plausible guess. Per
Adam's own closing note on the Work Instruction, every price/identity/group
value this module emits is meant to be spot-checked by a human before being
treated as authoritative -- that is a property of the source photos (sticker
legibility, grid segmentation, mixed-franchise pages), not something this
code claims to have solved.

REQUIRES
--------
``pip install pytesseract`` plus the Tesseract OCR binary itself on PATH
(the separate system install -- pytesseract is just a wrapper around it).
Checked on this machine while building this module: neither was present in
the plain Windows dev checkout (only numpy/opencv, cardcenter's own runtime
deps, were). Page/grid detection and pairing (binder_grid.py) have no OCR
dependency and work without either; identity/sticker text (binder_sticker.py)
raises a clear RuntimeError, not a silent empty result, if pytesseract can't
be imported when a caller actually reaches an OCR call.

USAGE
-----
    python -m ingest.binder_ingest --root /path/to/capture_folder --out out/

``--root`` may be a parent folder containing the four group subfolders named
in Work Instruction v2.0 S2 (Pokemon, Animal_Crossing, Magic_the_Gathering,
Yugioh -- matched case- and accent-insensitively, so an accented spelling of
Pokemon's name also matches), or a single folder of photos treated as one
group via ``--group``. Writes, per group, ``<group>_records.jsonl`` (every
pocket processed, matched or flagged) and ``<group>_flagged.jsonl``
(everything with status != matched), plus a plain-text ``summary.txt``
across all groups.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2

from . import binder_grid as grid
from . import binder_sticker as sticker

FILENAME_RE = re.compile(r"^(?P<prefix>[A-Za-z]+)_(?P<date>\d{8})_(?P<seq>\d+)")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

DEFAULT_GROUPS = ("Pokemon", "Animal_Crossing", "Magic_the_Gathering", "Yugioh")


def _norm_group_name(name: str) -> str:
    """Case/accent-insensitive folder-name key (NFKD-normalised), so an
    accented and an unaccented spelling of the same franchise name match."""
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFKD", name) if not unicodedata.combining(ch)
    )
    return stripped.strip().lower().replace(" ", "_")


def sort_key(path: Path):
    """Work Instruction S3: sort by (date, numeric tail) ascending. A name
    that doesn't match the expected pattern sorts after everything that
    does, stably by filename, rather than crashing the whole batch on one
    oddly-named file.
    """
    m = FILENAME_RE.match(path.stem)
    if not m:
        return (1, path.name, "")
    return (0, m.group("date"), m.group("seq").rjust(12, "0"))


def discover_groups(root: Path, explicit_group: Optional[str]) -> dict:
    """Return {group_name: [sorted image paths]}.

    If ``root`` contains one or more of the recognised group subfolders,
    each becomes its own group (S2). Otherwise ``root`` itself is treated as
    a single group, named by ``explicit_group`` if given or else the
    folder's own name -- with a printed warning, since an unnamed group
    means franchise routing has nothing to default to.
    """
    subdirs = {p.name: p for p in root.iterdir() if p.is_dir()} if root.is_dir() else {}
    matched = {}
    for name, path in subdirs.items():
        key = _norm_group_name(name)
        if key in {_norm_group_name(g) for g in DEFAULT_GROUPS}:
            matched[name] = path

    groups: dict[str, list[Path]] = {}
    if matched:
        for name, path in matched.items():
            files = sorted(
                (p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=sort_key
            )
            groups[name] = files
    else:
        group_name = explicit_group or root.name
        files = sorted(
            (p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=sort_key
        )
        groups[group_name] = files
    return groups


@dataclass
class PagePair:
    front: Path
    back: Optional[Path]  # None => unpaired trailing front (S8 pending_price)


def pair_pages(files: list[Path]) -> list[PagePair]:
    """S4/S10: strict front,back,front,back,... alternation by capture order.

    This assumes the physical capture sequence the Work Instruction
    describes (photograph front, flip the page, photograph back, turn to
    the next page) with no skipped or reordered shots. It does not detect a
    corrupted sequence (e.g. two fronts photographed in a row) -- there is
    no reliable signal in a single image alone that says "this is a front"
    vs "this is a back"; that would need a cross-page heuristic (card-back
    uniformity) not implemented here. A folder whose capture order was
    disrupted will silently mispair past that point, which is why every
    record this module emits is meant for human spot-checking, not blind
    trust (see the module docstring).
    """
    pairs = []
    i = 0
    while i < len(files):
        if i + 1 < len(files):
            pairs.append(PagePair(files[i], files[i + 1]))
            i += 2
        else:
            pairs.append(PagePair(files[i], None))
            i += 1
    return pairs


@dataclass
class PocketRecord:
    group: str
    front_image: str
    back_image: Optional[str]
    front_pocket: dict
    back_pocket: Optional[dict]
    card_identity_ocr: str
    price_usd: Optional[str]
    sticker_sku: Optional[str]
    sticker_tier_label: Optional[str]
    sticker_date: Optional[str]
    pairing_confidence: str
    status: str
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _pairing_confidence(front_score: float, back_seam_conf: str, sticker_targeted: bool) -> str:
    if front_score >= 12.0 and back_seam_conf == "supported" and sticker_targeted:
        return "high"
    if front_score >= 3.0:
        return "medium"
    return "low"


def process_page_pair(
    pair: PagePair, group: str, cfg: grid.GridConfig, expected_group_key: str
) -> list[PocketRecord]:
    front_img = cv2.imread(str(pair.front))
    if front_img is None:
        return [
            PocketRecord(
                group, pair.front.name, pair.back.name if pair.back else None,
                {}, None, "", None, None, None, None, "low", "no_card_detected",
                notes=[f"could not read image file {pair.front}"],
            )
        ]

    try:
        front_quad = grid.find_page_quad(front_img, cfg.min_area_frac)
        front_rect, _ = grid.rectify_page(front_img, front_quad)
        front_grid = grid.build_grid(front_rect, cfg)
    except grid.GridDetectionError as exc:
        return [
            PocketRecord(
                group, pair.front.name, pair.back.name if pair.back else None,
                {}, None, "", None, None, None, None, "low", "no_card_detected",
                notes=[f"front page: {exc}"],
            )
        ]

    if pair.back is None:
        # S8: unpaired front page. Still worth recording WHICH pockets look
        # occupied, so a human knows what needs a price rather than having
        # to re-open the photo.
        records = []
        for (r, c), pocket in sorted(front_grid.pockets.items()):
            crop = grid.crop_pocket(front_rect, pocket, cfg.pocket_margin_frac)
            occupied = grid.has_content(crop)
            identity, score = sticker.read_front_identity(crop) if occupied else ("", 0.0)
            records.append(
                PocketRecord(
                    group, pair.front.name, None,
                    {"row": r, "col": c}, None,
                    identity, None, None, None, None,
                    "low" if occupied else "low",
                    "pending_price" if occupied else "no_card_detected",
                    notes=["no back-page photo followed this one in capture order"],
                )
            )
        return records

    back_img = cv2.imread(str(pair.back))
    if back_img is None:
        return [
            PocketRecord(
                group, pair.front.name, pair.back.name, {}, None, "", None, None, None, None,
                "low", "no_card_detected", notes=[f"could not read image file {pair.back}"],
            )
        ]
    try:
        back_quad = grid.find_page_quad(back_img, cfg.min_area_frac)
        back_rect, _ = grid.rectify_page(back_img, back_quad)
        back_grid = grid.build_grid(back_rect, cfg)
    except grid.GridDetectionError as exc:
        return [
            PocketRecord(
                group, pair.front.name, pair.back.name, {}, None, "", None, None, None, None,
                "low", "no_card_detected", notes=[f"back page: {exc}"],
            )
        ]

    # Whole-page grid_mismatch check (S8): the two pages should be the SAME
    # physical sheet, so their rectified aspect ratios should closely agree.
    # A real per-pocket count mismatch (spec's own "9 front, 8 back" example)
    # can't arise here since both sides use the same configured (rows,cols)
    # -- this is the closest available proxy for "these two photos are
    # probably not really a front/back pair of one page", and it is checked
    # BEFORE trusting any pocket-level pairing on this page, per S8.
    fh, fw = front_grid.rect_shape
    bh, bw = back_grid.rect_shape
    f_aspect, b_aspect = fh / max(fw, 1), bh / max(bw, 1)
    if abs(f_aspect - b_aspect) / max(f_aspect, 1e-6) > 0.15:
        note = (
            f"front/back rectified aspect differs ({f_aspect:.2f} vs {b_aspect:.2f}); "
            "these may not be the same physical page -- verify pairing before trusting it"
        )
        records = []
        for (r, c) in sorted(front_grid.pockets):
            records.append(
                PocketRecord(
                    group, pair.front.name, pair.back.name,
                    {"row": r, "col": c}, None, "", None, None, None, None,
                    "low", "grid_mismatch", notes=[note],
                )
            )
        return records

    records: list[PocketRecord] = []
    for (r, c), front_pocket in sorted(front_grid.pockets.items()):
        back_col = grid.mirror_column(c, front_grid.cols)
        back_pocket = back_grid.pockets.get((r, back_col))
        front_crop = grid.crop_pocket(front_rect, front_pocket, cfg.pocket_margin_frac)
        front_occupied = grid.has_content(front_crop)

        if back_pocket is None:
            records.append(
                PocketRecord(
                    group, pair.front.name, pair.back.name,
                    {"row": r, "col": c}, None, "", None, None, None, None,
                    "low", "grid_mismatch",
                    notes=[f"mirrored back column {back_col} is outside the back page's grid"],
                )
            )
            continue

        back_crop = grid.crop_pocket(back_rect, back_pocket, cfg.pocket_margin_frac)
        back_occupied = grid.has_content(back_crop)
        back_pocket_dict = {"row": back_pocket.row, "col": back_pocket.col}

        if not front_occupied and not back_occupied:
            records.append(
                PocketRecord(
                    group, pair.front.name, pair.back.name,
                    {"row": r, "col": c}, back_pocket_dict, "", None, None, None, None,
                    "low", "no_card_detected", notes=[],
                )
            )
            continue

        if front_occupied != back_occupied:
            side = "front" if front_occupied else "back"
            records.append(
                PocketRecord(
                    group, pair.front.name, pair.back.name,
                    {"row": r, "col": c}, back_pocket_dict, "", None, None, None, None,
                    "low", "pocket_mismatch",
                    notes=[f"{side} pocket has a card but the mirrored pocket looks empty"],
                )
            )
            continue

        # Both sides occupied: read identity, sticker, and check franchise.
        identity, id_score = sticker.read_front_identity(front_crop)
        fields = sticker.extract_sticker_fields(back_crop)
        franchise = sticker.classify_back_franchise(back_crop)

        notes = list(fields.warnings)
        status = "matched"

        expected = _norm_group_name(expected_group_key)
        if franchise.guess is not None and franchise.confidence >= 0.5:
            if _norm_group_name(franchise.guess) != expected:
                status = "franchise_mismatch"
                notes.append(
                    f"back design looks like {franchise.guess} (confidence "
                    f"{franchise.confidence:.2f}) but this pocket is filed under "
                    f"'{expected_group_key}'"
                )

        if status == "matched" and fields.price_usd is None:
            status = "price_needs_review"

        conf = _pairing_confidence(id_score, back_grid.seam_confidence, fields.used_targeted_crop)

        records.append(
            PocketRecord(
                group=group,
                front_image=pair.front.name,
                back_image=pair.back.name,
                front_pocket={"row": r, "col": c},
                back_pocket=back_pocket_dict,
                card_identity_ocr=identity,
                price_usd=fields.price_usd,
                sticker_sku=fields.sticker_sku,
                sticker_tier_label=fields.sticker_tier_label,
                sticker_date=fields.sticker_date,
                pairing_confidence=conf,
                status=status,
                notes=notes,
            )
        )
    return records


@dataclass
class GroupSummary:
    group: str
    n_pages: int
    n_pairs: int
    n_pockets: int
    n_matched: int
    n_flagged: int
    by_status: dict


def run(root: Path, out_dir: Path, cfg: grid.GridConfig, explicit_group: Optional[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = discover_groups(root, explicit_group)
    summaries: list[GroupSummary] = []

    for group_name, files in groups.items():
        pairs = pair_pages(files)
        all_records: list[PocketRecord] = []
        for pair in pairs:
            all_records.extend(process_page_pair(pair, group_name, cfg, group_name))

        by_status: dict[str, int] = {}
        for r in all_records:
            by_status[r.status] = by_status.get(r.status, 0) + 1

        records_path = out_dir / f"{_norm_group_name(group_name)}_records.jsonl"
        flagged_path = out_dir / f"{_norm_group_name(group_name)}_flagged.jsonl"
        with records_path.open("w", encoding="utf-8") as fh:
            for r in all_records:
                fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
        with flagged_path.open("w", encoding="utf-8") as fh:
            for r in all_records:
                if r.status != "matched":
                    fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

        summaries.append(
            GroupSummary(
                group=group_name,
                n_pages=len(files),
                n_pairs=len(pairs),
                n_pockets=len(all_records),
                n_matched=by_status.get("matched", 0),
                n_flagged=len(all_records) - by_status.get("matched", 0),
                by_status=by_status,
            )
        )

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as fh:
        for s in summaries:
            fh.write(f"{s.group}: {s.n_pages} photos, {s.n_pairs} page-pairs, "
                      f"{s.n_pockets} pockets processed, {s.n_matched} matched, "
                      f"{s.n_flagged} flagged\n")
            for status, n in sorted(s.by_status.items()):
                fh.write(f"    {status}: {n}\n")
    return summaries


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="capture folder (parent of group subfolders, or one group's own folder)")
    ap.add_argument("--out", required=True, type=Path, help="output directory for *_records.jsonl / *_flagged.jsonl / summary.txt")
    ap.add_argument("--group", default=None, help="group name to use when --root has no recognised group subfolders")
    ap.add_argument("--rows", type=int, default=3, help="pocket grid rows per page (default 3)")
    ap.add_argument("--cols", type=int, default=3, help="pocket grid columns per page (default 3)")
    ap.add_argument("--min-page-area-frac", type=float, default=0.25, help="min fraction of frame the page must occupy to be detected")
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        print(f"--root {args.root} is not a directory", file=sys.stderr)
        return 2

    cfg = grid.GridConfig(rows=args.rows, cols=args.cols, min_area_frac=args.min_page_area_frac)
    summaries = run(args.root, args.out, cfg, args.group)
    for s in summaries:
        print(f"{s.group}: {s.n_matched} matched / {s.n_pockets} pockets "
              f"({s.n_flagged} flagged) -- {s.by_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
