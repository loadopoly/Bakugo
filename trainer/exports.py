"""Labelled sets for evaluators that already exist in the repo.

* ``recognition_manifest`` writes the frozen, confirmed test set in the JSONL
  format ``eval/recognition_eval.py`` reads (``--detect`` for raw photos).
* ``price_manifest`` writes each frozen test item's resolved general price
  and where it came from (sticker, tag, sign, page, lot, receipt, ...). That
  set is collection content; it is written to the private tier only.
"""

from __future__ import annotations

import json
from pathlib import Path

from .data import frozen_test_items
from .paths import Layout


def recognition_manifest(con, layout: Layout, model_id: str = "recognition-eval-export") -> Path:
    items = [i for i in frozen_test_items(con, layout, model_id=model_id, task="recognition")
             if i.label]
    out = layout.reports / "recognition_test.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for i in items:
            fh.write(json.dumps({"image": str(i.path), "name": i.label, "number": i.number,
                                 "px_per_mm": i.px_per_mm, "sha256": i.sha256}) + "\n")
    return out


def price_manifest(con, layout: Layout, kind: str = "asking") -> Path:
    """Confirmed test items with their resolved general price, whatever the
    source. Evaluate the sticker/tag OCR on the rows whose ``source`` is one
    it reads. Collection content: private tier only."""
    from .manifest import item_price, resolve_path

    rows = con.execute(
        "SELECT f.sha256, f.rel_path FROM files f WHERE f.split = 'test' AND f.present "
        "AND EXISTS (SELECT 1 FROM frozen_test z WHERE z.sha256 = f.sha256) ORDER BY f.sha256"
    ).fetchall()
    out = layout.reports / "price_test.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for sha, rel in rows:
            price = item_price(con, sha, kind=kind)
            if price is None:
                continue
            a = price.attribution
            fh.write(json.dumps({"image": str(resolve_path(layout, rel)), "sha256": sha,
                                 "amount": str(price.amount), "currency": price.currency,
                                 "source": a.source, "scope": a.scope, "method": a.method,
                                 "raw_text": a.raw_text, "basis": price.basis}) + "\n")
            n += 1
    return out
