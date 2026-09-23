"""Steps 2-3: pseudo-label new files with the current system, then queue the
items a human should look at.

Pseudo-labels go to ``pseudo_labels`` only. They are never copied into
``files.label`` and never count as confirmed.

What gets queued:

* every unconfirmed item on the **test** split, regardless of confidence --
  a test set built only from items the model found hard would measure the
  model on a skewed sample, so the test set is confirmed exhaustively;
* train items where the current system did not resolve, was ambiguous, fell
  below ``min_confidence``, or disagreed with the weak filename label or with
  another predictor;
* clutter captures (bulk piles, sleeved) with no confirmed card outline, for
  the quad-detection regression set.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

from .manifest import resolve_path
from .paths import Layout


@dataclass(frozen=True)
class Pseudo:
    label: Optional[str]
    number: Optional[int]
    confidence: float        # ranking only: 1 corroborated, 0.5 name only, 0 unresolved
    resolved: bool
    ambiguous: bool
    detail: str = ""


class Predictor(Protocol):
    model_id: str

    def predict(self, path: Path) -> Pseudo: ...


class CardcenterPredictor:
    """The current production reader: Tesseract + species snap, after
    detection and rectification. No network (QUIPU is disabled in the trainer)."""

    def __init__(self, detect: bool = True):
        from cardcenter import __version__
        from cardcenter.recognise import SparseTextEngine, load_species

        self.model_id = f"cardcenter-{__version__}"
        self._vocab = list(load_species())
        self._engine = SparseTextEngine()
        self._detect = detect

    def predict(self, path: Path) -> Pseudo:
        import cv2
        from cardcenter.recognise import recognise_card

        img = cv2.imread(str(path))
        if img is None:
            return Pseudo(None, None, 0.0, False, False, "image unreadable")
        note = ""
        if self._detect:
            try:
                from cardcenter.geometry import find_card_quad
                from cardcenter.multicard import _crop_quad

                quad, _, _ = find_card_quad(img)
                img, _ = _crop_quad(img, quad)
            except Exception as exc:
                note = f"detect failed ({exc}); full frame used"
        try:
            rec = recognise_card(img, vocabulary=self._vocab, engine=self._engine)
        except Exception as exc:
            return Pseudo(None, None, 0.0, False, False, f"recognise raised: {exc}")
        conf = 1.0 if rec.corroborated else (0.5 if rec.resolved else 0.0)
        detail = "; ".join(filter(None, [note, *rec.warnings]))
        return Pseudo(rec.name, rec.dex, conf, bool(rec.resolved), bool(rec.alternatives), detail)


def pseudo_label(con, layout: Layout, predictors: Sequence[Predictor],
                 shas: Optional[Sequence[str]] = None) -> int:
    """Label every present, unconfirmed trainable item each predictor has not
    seen yet (or only ``shas``). Returns rows written."""
    now = _dt.datetime.now().replace(microsecond=0)
    written = 0
    for pred in predictors:
        sql = ("SELECT f.sha256, f.rel_path FROM files f WHERE f.present "
               "AND f.kind IN ('image','pdf_page') "
               "AND COALESCE(f.label_strength, '') <> 'confirmed' "
               "AND NOT EXISTS (SELECT 1 FROM pseudo_labels p "
               "                WHERE p.sha256 = f.sha256 AND p.model_id = ?)")
        rows = con.execute(sql, [pred.model_id]).fetchall()
        if shas is not None:
            wanted = set(shas)
            rows = [r for r in rows if r[0] in wanted]
        for sha, rel in rows:
            p = pred.predict(resolve_path(layout, rel))
            con.execute(
                "INSERT OR REPLACE INTO pseudo_labels VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [sha, pred.model_id, p.label, p.number, p.confidence,
                 p.resolved, p.ambiguous, p.detail, now],
            )
            written += 1
    return written


def _same(a: Optional[str], b: Optional[str]) -> bool:
    return (a or "").strip().lower() == (b or "").strip().lower()


def build_queue(con, layout: Layout, min_confidence: float = 1.0) -> list[dict]:
    rows = con.execute(
        """SELECT sha256, rel_path, split, label, label_strength, source_type, franchise
           FROM files WHERE present AND kind IN ('image','pdf_page')
           AND COALESCE(label_strength, '') <> 'confirmed'
           ORDER BY split DESC, sha256"""
    ).fetchall()
    pseudo: dict[str, list] = {}
    for sha, model, label, number, conf, resolved, amb, detail in con.execute(
            "SELECT sha256, model_id, label, number, confidence, resolved, ambiguous, detail "
            "FROM pseudo_labels").fetchall():
        pseudo.setdefault(sha, []).append(
            {"model_id": model, "label": label, "number": number, "confidence": conf,
             "resolved": resolved, "ambiguous": amb, "detail": detail})
    queue = []
    for sha, rel, split, weak, strength, source, franchise in rows:
        ps = pseudo.get(sha, [])
        reasons = []
        if split == "test":
            reasons.append("test split: every test item is confirmed")
        if not ps:
            reasons.append("not yet pseudo-labelled")
        for p in ps:
            if not p["resolved"]:
                reasons.append(f"{p['model_id']}: unresolved")
            elif p["ambiguous"]:
                reasons.append(f"{p['model_id']}: ambiguous")
            elif (p["confidence"] or 0.0) < min_confidence:
                reasons.append(f"{p['model_id']}: confidence {p['confidence']:.2f}")
            if weak and p["label"] and not _same(weak, p["label"]):
                reasons.append(f"{p['model_id']}: disagrees with filename label {weak!r}")
        names = {(p["label"] or "").lower() for p in ps if p["resolved"]}
        if len(names) > 1:
            reasons.append("predictors disagree")
        if reasons:
            queue.append({"task": "identity", "sha256": sha, "path": str(resolve_path(layout, rel)),
                          "split": split, "franchise": franchise, "source_type": source,
                          "weak_label": weak, "pseudo": ps, "reasons": reasons})
    for sha, rel, split, source in con.execute(
            """SELECT f.sha256, f.rel_path, f.split, f.source_type FROM files f
               WHERE f.present AND f.kind IN ('image','pdf_page')
               AND f.source_type IN ('bulk_pile', 'sleeved')
               AND NOT EXISTS (SELECT 1 FROM quad_labels q WHERE q.sha256 = f.sha256)
               ORDER BY f.sha256""").fetchall():
        queue.append({"task": "quad", "sha256": sha, "path": str(resolve_path(layout, rel)),
                      "split": split, "source_type": source,
                      "reasons": ["clutter capture without a confirmed card outline"]})
    return queue


def export_queue(layout: Layout, queue: list[dict]) -> Path:
    layout.queue.mkdir(parents=True, exist_ok=True)
    out = layout.queue / f"review-{_dt.date.today().isoformat()}.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for entry in queue:
            fh.write(json.dumps(entry, default=str) + "\n")
    return out
