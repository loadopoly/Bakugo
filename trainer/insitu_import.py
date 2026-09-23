"""In-situ data reaching the trainer.

``import_inbox``
    The app writes confirmed photos from the owner's devices into
    ``<private>/inbox/<date>/`` (``cardcenter.insitu.write_inbox``). The
    manifest scan picks the image up as ``inbox/...``; this step applies the
    sidecar: the owner-confirmed label (``confirmed_by =
    owner-device:<id>``), ``source_type = app_capture``, and the price
    attribution if one was entered. The detected outline in the sidecar is the
    system's own output, so it is not imported as a quad label. Applied
    sidecars are renamed ``*.json.imported``.

``public_feedback_queue``
    Aggregate confirmations and corrections from every other device, read
    from the app's SQLite file (``BAKUGO_APP_DB``, opened read-only). No images
    and no device ids; each row is a review item for the owner, never a
    label.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from . import manifest
from .paths import Layout


def import_inbox(con, layout: Layout) -> dict:
    from cardcenter.pricing import PriceAttribution, PriceError

    applied, skipped = 0, []
    if not layout.inbox.exists():
        return {"applied": 0, "skipped": []}
    for side in sorted(layout.inbox.rglob("*.json")):
        try:
            body = json.loads(side.read_text(encoding="utf-8"))
        except ValueError:
            skipped.append(f"{side.name}: unreadable")
            continue
        if body.get("schema") != "bakugo-insitu/1":
            skipped.append(f"{side.name}: unknown schema")
            continue
        img = side.parent / str(body.get("image", ""))
        rel = "inbox/" + img.relative_to(layout.inbox).as_posix()
        row = con.execute("SELECT sha256 FROM file_paths WHERE rel_path = ?", [rel]).fetchone()
        if row is None:
            skipped.append(f"{side.name}: image not in manifest yet")
            continue
        sha = row[0]
        who = str(body.get("confirmed_by") or "")
        if not who.startswith("owner-device:"):
            skipped.append(f"{side.name}: not from an owner device")
            continue
        label = body.get("label") or {}
        if label.get("action") not in ("confirm", "correct"):
            skipped.append(f"{side.name}: not a confirmation")
            continue
        franchise = label.get("franchise")
        franchise = franchise if franchise in manifest.FRANCHISES else None
        ident = body.get("identification") or {}
        ppm = ident.get("px_per_mm")
        manifest.confirm(con, sha, confirmed_by=who, label=label.get("name"),
                         number=label.get("number"), franchise=franchise,
                         source_type="app_capture",
                         px_per_mm=float(ppm) if isinstance(ppm, (int, float)) else None,
                         note=f"in-situ identification {ident.get('id')}")
        price = body.get("price")
        if price:
            try:
                manifest.attribute_price(con, PriceAttribution.from_dict(price),
                                         recorded_by=who, sha256=sha)
            except (PriceError, TypeError) as exc:
                skipped.append(f"{side.name}: price not recorded ({exc})")
        side.rename(side.with_name(side.name + ".imported"))
        applied += 1
    return {"applied": applied, "skipped": skipped}


def public_feedback_queue(app_db: Optional[str], since: float = 0.0) -> list[dict]:
    if not app_db or not Path(app_db).exists():
        return []
    from cardcenter.insitu import public_feedback_summary

    uri = Path(app_db).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    try:
        try:
            rows = public_feedback_summary(conn, since)
        except sqlite3.OperationalError:
            return []          # app has not created the in-situ tables yet
    finally:
        conn.close()
    out = []
    for r in rows:
        reason = (f"{r['n']} {r['action']} report(s) from {r['devices']} device(s): "
                  f"OCR {r['ocr_name']!r} (token {r['ocr_token']!r}) -> {r['user_name']!r}")
        out.append({"task": "public_feedback", **r, "reasons": [reason]})
    return out
