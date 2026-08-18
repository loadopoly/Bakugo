"""Storing scans and grade labels, with provenance that cannot be washed off.

THE CIRCULARITY PROBLEM
-----------------------
The tempting design is: the model estimates a grade, the marketplace shows that
estimate to participants, participants "confirm" it, confirmations become
training labels, the retrained model sets prices. Every arrow in that loop is
reasonable on its own and the whole is degenerate, because there is no external
anchor anywhere in it. A model trained on opinions it originally seeded will
converge on internal consistency rather than on truth, and it will drift in
whichever direction is most profitable to the people voting. When those same
numbers set prices, the drift is the product rather than a bug in it.

The only label that breaks the loop is a grade issued by an actual grading
company on that actual physical card, identified by cert number. It is slow and
it costs money, which is exactly why the shortcut is tempting.

So this store records WHERE every label came from and never lets the kinds mix
silently:

    CERTIFIED        a real grader graded this physical card; cert number
                     required. The only kind that trains anything by default.
    SELF_REPORTED    the owner says it graded X. Usually true, sometimes
                     aspirational, never verifiable. Evaluation only.
    MARKETPLACE_VOTE crowd opinion from a listing or a confirmatory-grading
                     poll. This is sentiment data. It is not a grade.
    MODEL_PREDICTED  our own output. Training on this is training on nothing.

``export_training_set`` returns CERTIFIED rows only unless explicitly overridden,
and the override is recorded in the manifest so an audit can see it happened.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Optional

from .types import CenteringResult


class LabelKind(str, Enum):
    CERTIFIED = "certified"
    SELF_REPORTED = "self_reported"
    MARKETPLACE_VOTE = "marketplace_vote"
    MODEL_PREDICTED = "model_predicted"

    @property
    def is_ground_truth(self) -> bool:
        return self is LabelKind.CERTIFIED


@dataclass
class ScanRecord:
    card_key: str
    holder: str
    worst_ratio_pct: float
    worst_ratio_sigma: float
    worst_axis: str
    h_ratio_pct: float
    v_ratio_pct: float
    left_mm: float
    right_mm: float
    top_mm: float
    bottom_mm: float
    px_per_mm: float
    inner_confidence: float
    refraction_applied: bool
    warnings: str
    phash: int = 0
    source: str = ""
    created_at: float = 0.0


SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_key TEXT NOT NULL,
    holder TEXT NOT NULL,
    worst_ratio_pct REAL NOT NULL,
    worst_ratio_sigma REAL NOT NULL,
    worst_axis TEXT NOT NULL,
    h_ratio_pct REAL NOT NULL,
    v_ratio_pct REAL NOT NULL,
    left_mm REAL, right_mm REAL, top_mm REAL, bottom_mm REAL,
    px_per_mm REAL,
    inner_confidence REAL,
    refraction_applied INTEGER,
    warnings TEXT,
    phash INTEGER,
    source TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id INTEGER NOT NULL REFERENCES scans(id),
    grader TEXT NOT NULL,
    grade TEXT NOT NULL,
    centering_subgrade TEXT,
    kind TEXT NOT NULL,
    cert_number TEXT,
    attributed_to TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_labels_scan ON labels(scan_id);
CREATE INDEX IF NOT EXISTS idx_labels_kind ON labels(kind);
CREATE INDEX IF NOT EXISTS idx_scans_card ON scans(card_key);
CREATE TABLE IF NOT EXISTS _schema_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


class ScanStore:
    def __init__(self, path: str = "cardcenter.db") -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "INSERT OR IGNORE INTO _schema_metadata (key, value, updated_at) VALUES ('schema_version', 'cardcenter/2', ?)",
            (time.time(),),
        )
        self.conn.commit()
        self._ensure_sync_columns()


    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "ScanStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writing ----------------------------------------------------------

    def add_scan(
        self,
        card_key: str,
        result: CenteringResult,
        phash: int = 0,
        source: str = "",
    ) -> int:
        w = result.worst_ratio
        rec = ScanRecord(
            card_key=card_key,
            holder=result.slab.name,
            worst_ratio_pct=w.value,
            worst_ratio_sigma=w.sigma,
            worst_axis=result.worst_axis.axis,
            h_ratio_pct=result.horizontal.ratio_pct.value,
            v_ratio_pct=result.vertical.ratio_pct.value,
            left_mm=result.horizontal.low_mm.value,
            right_mm=result.horizontal.high_mm.value,
            top_mm=result.vertical.low_mm.value,
            bottom_mm=result.vertical.high_mm.value,
            px_per_mm=result.px_per_mm,
            inner_confidence=result.quality.inner_confidence,
            refraction_applied=result.quality.refraction_applied,
            warnings=" | ".join(result.quality.warnings),
            phash=phash,
            source=source,
            created_at=time.time(),
        )
        return self.add_scan_record(asdict(rec))

    def add_scan_record(self, rec: dict) -> int:
        """Insert a scan from a mapping (CLI, serve, or web measure payload)."""
        d = dict(rec)
        d.setdefault("created_at", time.time())
        d["refraction_applied"] = int(bool(d.get("refraction_applied")))
        cols = ", ".join(d)
        marks = ", ".join("?" for _ in d)
        cur = self.conn.execute(
            f"INSERT INTO scans ({cols}) VALUES ({marks})", tuple(d.values())
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_scan_from_measure(self, payload: dict, source: str = "measure") -> int:
        """Persist a successful ``_measure_payload`` / CLI result dict."""
        borders = payload.get("borders") or {}
        return self.add_scan_record(
            {
                "card_key": payload.get("card_key") or "unidentified",
                "holder": payload.get("holder") or "raw",
                "worst_ratio_pct": float(payload.get("ratio") or payload.get("worst_ratio_pct") or 0),
                "worst_ratio_sigma": float(
                    payload.get("worst_ratio_sigma")
                    or ((payload.get("ratio_hi") or 0) - (payload.get("ratio_lo") or 0)) / 3.92
                    or 0
                ),
                "worst_axis": payload.get("axis") or payload.get("worst_axis") or "",
                "h_ratio_pct": float(payload.get("h_ratio_pct") or payload.get("ratio") or 0),
                "v_ratio_pct": float(payload.get("v_ratio_pct") or payload.get("ratio") or 0),
                "left_mm": borders.get("left"),
                "right_mm": borders.get("right"),
                "top_mm": borders.get("top"),
                "bottom_mm": borders.get("bottom"),
                "px_per_mm": payload.get("px_per_mm"),
                "inner_confidence": payload.get("inner_confidence"),
                "refraction_applied": int(bool(payload.get("refraction"))),
                "warnings": " | ".join(payload.get("warnings") or []),
                "phash": int(payload.get("phash") or 0),
                "source": source,
            }
        )

    def get_scan(self, scan_id: int) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
        return dict(row) if row else None

    def labels_for_scan(self, scan_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM labels WHERE scan_id = ? ORDER BY id", (scan_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def unsynced_scans(self) -> list[dict]:
        self._ensure_sync_columns()
        rows = self.conn.execute(
            "SELECT * FROM scans WHERE COALESCE(synced_at, 0) = 0 ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_synced(self, scan_id: int, when: Optional[float] = None) -> None:
        self._ensure_sync_columns()
        self.conn.execute(
            "UPDATE scans SET synced_at = ? WHERE id = ?",
            (when if when is not None else time.time(), scan_id),
        )
        self.conn.commit()

    def _ensure_sync_columns(self) -> None:
        cols = {
            r[1]
            for r in self.conn.execute("PRAGMA table_info(scans)").fetchall()
        }
        if "synced_at" not in cols:
            self.conn.execute("ALTER TABLE scans ADD COLUMN synced_at REAL DEFAULT 0.0")
            self.conn.commit()

    def add_label(
        self,
        scan_id: int,
        grader: str,
        grade: str,
        kind: LabelKind,
        centering_subgrade: Optional[str] = None,
        cert_number: Optional[str] = None,
        attributed_to: Optional[str] = None,
    ) -> int:
        """Record a grade label. Certified labels must carry a cert number.

        The cert number is what makes a certified label checkable by someone who
        does not trust us. Without it the claim is indistinguishable from
        self-reporting, so it is rejected rather than downgraded silently.
        """
        if kind is LabelKind.CERTIFIED and not cert_number:
            raise ValueError(
                "a certified label requires a cert number. Without one it is "
                "self-reported, and mislabelling it as certified poisons every "
                "training set built from this store."
            )
        cur = self.conn.execute(
            "INSERT INTO labels (scan_id, grader, grade, centering_subgrade, kind, "
            "cert_number, attributed_to, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                scan_id,
                grader,
                grade,
                centering_subgrade,
                kind.value,
                cert_number,
                attributed_to,
                time.time(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    # -- reading ----------------------------------------------------------

    def label_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT kind, COUNT(*) AS n FROM labels GROUP BY kind"
        ).fetchall()
        return {r["kind"]: r["n"] for r in rows}

    def scan_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0])

    def export_training_set(
        self,
        include_kinds: Iterable[LabelKind] = (LabelKind.CERTIFIED,),
        acknowledge_contamination: bool = False,
    ) -> dict:
        """Export labelled scans for model training.

        Defaults to certified labels only. Anything else requires
        ``acknowledge_contamination=True`` and is stamped into the manifest, so
        that a model trained on crowd votes cannot later be described as having
        been trained on grades.
        """
        kinds = tuple(include_kinds)
        non_truth = [k for k in kinds if not k.is_ground_truth]
        if non_truth and not acknowledge_contamination:
            raise ValueError(
                "refusing to export "
                + ", ".join(k.value for k in non_truth)
                + " as training labels. These are opinions, not grades; a model "
                "trained on them learns to reproduce the sentiment that produced "
                "them. If you have a specific reason, pass "
                "acknowledge_contamination=True and it will be recorded in the "
                "manifest."
            )

        placeholders = ", ".join("?" for _ in kinds)
        rows = self.conn.execute(
            f"""SELECT s.*, l.grader, l.grade, l.centering_subgrade, l.kind,
                       l.cert_number
                FROM labels l JOIN scans s ON s.id = l.scan_id
                WHERE l.kind IN ({placeholders})""",
            tuple(k.value for k in kinds),
        ).fetchall()

        manifest = {
            "exported_at": time.time(),
            "n_examples": len(rows),
            "kinds_included": [k.value for k in kinds],
            "ground_truth_only": not non_truth,
            "contamination_acknowledged": bool(non_truth),
            "database": str(Path(self.path).resolve()),
        }
        if non_truth:
            manifest["contamination_warning"] = (
                "This training set contains labels that are not grader-issued "
                "grades: " + ", ".join(k.value for k in non_truth) + ". A model "
                "fitted to these has no external anchor and must not be "
                "described as predicting grades."
            )
        return {"manifest": manifest, "examples": [dict(r) for r in rows]}

    def circularity_report(self) -> str:
        """Say plainly how much of the label pool is actually independent."""
        counts = self.label_counts()
        total = sum(counts.values())
        if total == 0:
            return "no labels recorded yet; nothing can be trained"
        certified = counts.get(LabelKind.CERTIFIED.value, 0)
        model = counts.get(LabelKind.MODEL_PREDICTED.value, 0)
        vote = counts.get(LabelKind.MARKETPLACE_VOTE.value, 0)
        lines = [
            f"labels total          : {total}",
            f"  certified (usable)  : {certified} ({100 * certified / total:.0f}%)",
            f"  self-reported       : {counts.get(LabelKind.SELF_REPORTED.value, 0)}",
            f"  marketplace votes   : {vote}",
            f"  model predictions   : {model}",
            "",
        ]
        if certified == 0:
            lines.append(
                "NO independent labels. Nothing here can train or evaluate a "
                "grading model; every label traces back to an opinion or to this "
                "tool's own output."
            )
        elif certified < 0.2 * total:
            lines.append(
                f"Only {100 * certified / total:.0f}% of labels are independent. A "
                "model fitted to this pool is mostly fitting its own reflection."
            )
        else:
            lines.append(
                f"{certified} independent labels available for training and "
                "evaluation."
            )
        if vote or model:
            lines.append(
                "Marketplace votes and model predictions are retained for "
                "analysis but are excluded from training exports by default."
            )
        return "\n".join(lines)


def write_manifest(path: str, payload: dict) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path
