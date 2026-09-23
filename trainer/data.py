"""The only readers the train and evaluate steps are allowed to use.

``training_items`` returns confirmed labels on the train split. It has no
parameter that widens it to weak or pseudo labels, or to the test split.
``frozen_test_items`` returns confirmed labels on the frozen test split and records
the read in ``test_uses``, so the number of times the frozen set has been
consulted is visible (every consultation spends some of its independence).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .manifest import resolve_path
from .paths import Layout


@dataclass(frozen=True)
class Item:
    sha256: str
    path: Path
    label: Optional[str]
    number: Optional[int]
    franchise: Optional[str]
    source_type: str
    px_per_mm: Optional[float]
    card_uid: Optional[str]
    split: str


_COLS = ("sha256, rel_path, label, label_number, franchise, source_type, "
         "px_per_mm, card_uid, split")


def _items(layout: Layout, rows) -> list[Item]:
    return [Item(r[0], resolve_path(layout, r[1]), r[2], r[3], r[4], r[5], r[6], r[7], r[8])
            for r in rows]


def training_items(con, layout: Layout, franchise: Optional[str] = None,
                   source_type: Optional[str] = None) -> list[Item]:
    sql = (f"SELECT {_COLS} FROM files WHERE split = 'train' "
           "AND label_strength = 'confirmed' AND present "
           "AND kind IN ('image', 'pdf_page')")
    args = []
    if franchise:
        sql += " AND franchise = ?"
        args.append(franchise)
    if source_type:
        sql += " AND source_type = ?"
        args.append(source_type)
    items = _items(layout, con.execute(sql + " ORDER BY sha256", args).fetchall())
    assert all(i.split == "train" for i in items)
    return items


def frozen_test_items(con, layout: Layout, *, model_id: str, task: str,
               franchise: Optional[str] = None, source_type: Optional[str] = None) -> list[Item]:
    sql = (f"SELECT {_COLS} FROM files f WHERE split = 'test' "
           "AND label_strength = 'confirmed' AND present "
           "AND kind IN ('image', 'pdf_page') "
           "AND EXISTS (SELECT 1 FROM frozen_test z WHERE z.sha256 = f.sha256)")
    args = []
    if franchise:
        sql += " AND franchise = ?"
        args.append(franchise)
    if source_type:
        sql += " AND source_type = ?"
        args.append(source_type)
    items = _items(layout, con.execute(sql + " ORDER BY sha256", args).fetchall())
    con.execute("INSERT INTO test_uses (model_id, task, n_items, used_at) VALUES (?, ?, ?, ?)",
                [model_id, task, len(items), _dt.datetime.now().replace(microsecond=0)])
    return items


def quad_items(con, layout: Layout, split: str):
    """Confirmed card-outline labels, grouped per image, for one split."""
    import json

    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    rows = con.execute(
        """SELECT f.sha256, f.rel_path, f.px_per_mm, q.card_index, q.corners
           FROM quad_labels q JOIN files f USING (sha256)
           WHERE f.split = ? AND f.present ORDER BY f.sha256, q.card_index""",
        [split],
    ).fetchall()
    out: dict[str, dict] = {}
    for sha, rel, ppm, _idx, corners in rows:
        entry = out.setdefault(sha, {"sha256": sha, "path": resolve_path(layout, rel),
                                     "px_per_mm": ppm, "quads": []})
        entry["quads"].append(json.loads(corners))
    return list(out.values())
