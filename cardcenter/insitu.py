"""In-situ learning: what a user's confirmation is allowed to change.

Policy (chosen 2026-09-16): per-device learning plus owner review.

* Every ``/identify`` result is recorded server-side (``identifications``),
  scoped to the caller's device. Feedback refers to that record by id, so the
  "what OCR read" side of a correction comes from the server, not the client.
* A user's confirmation or correction is SELF-REPORTED. It updates only that
  device's own encounter counts (``device_priors``), which ``/identify`` uses
  for one thing: breaking an OCR tie between species names when the device's
  history clearly favours one (``break_tie``). The service worker on the same
  device adds the confirmed crop's embedding to its own on-device index.
  Nothing a device reports changes another device's results.
* Shared models change only through the private trainer, after the owner
  confirms. Public feedback reaches the trainer as aggregate counts with no
  images (``public_feedback_summary``); the trainer queues them for review.
* The owner's own devices (``CARDCENTER_OWNER_TOKEN`` sent as
  ``X-Bakugo-Owner``) can additionally save the confirmed photo with its
  labels and price into ``CARDCENTER_INSITU_INBOX``, a folder in the private
  tier. The trainer imports those as confirmed items
  (``confirmed_by = owner-device:<id>``).

Photos from non-owner devices are never written anywhere.
"""

from __future__ import annotations

import datetime as _dt
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS identifications (
    id TEXT PRIMARY KEY,
    device_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    name TEXT,
    dex INTEGER,
    matched_token TEXT,
    edits INTEGER,
    corroborated INTEGER NOT NULL DEFAULT 0,
    alternatives TEXT NOT NULL DEFAULT '[]',
    engine TEXT,
    px_per_mm REAL,
    shot_ratio REAL,
    resolvable INTEGER,
    quad TEXT,
    resolved_by TEXT,
    decision TEXT,
    decision_reason TEXT,
    vote TEXT
);
CREATE INDEX IF NOT EXISTS idx_ident_device ON identifications(device_id, created_at);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identification_id TEXT,
    scan_id INTEGER,
    device_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('confirm', 'correct', 'reject')),
    name TEXT,
    number INTEGER,
    franchise TEXT,
    note TEXT,
    owner INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_device ON feedback(device_id, created_at);
CREATE TABLE IF NOT EXISTS price_attributions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    identification_id TEXT,
    scan_id INTEGER,
    amount TEXT NOT NULL,
    currency TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    source TEXT NOT NULL,
    scope TEXT NOT NULL,
    scope_key TEXT,
    kind TEXT NOT NULL,
    method TEXT NOT NULL,
    venue TEXT,
    observed_at REAL NOT NULL,
    raw_text TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_price_device ON price_attributions(device_id, created_at);
CREATE TABLE IF NOT EXISTS device_priors (
    device_id TEXT NOT NULL,
    name TEXT NOT NULL,
    count INTEGER NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (device_id, name)
);
"""

MAX_TEXT = 120
TIE_MIN_COUNT = 3
TIE_MIN_RATIO = 3.0
PRIOR_ALPHA = 1.0


class FeedbackError(ValueError):
    pass


def _clean(value, what: str, max_len: int = MAX_TEXT) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) > max_len or any(ord(c) < 32 for c in text):
        raise FeedbackError(f"{what} is not a short single-line string")
    return text


class InSituStore:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- identifications ---------------------------------------------------
    def record_identification(self, device_id: str, payload: dict) -> str:
        ident = secrets.token_urlsafe(12)
        self.conn.execute(
            """INSERT INTO identifications (id, device_id, created_at, name, dex, matched_token,
                   edits, corroborated, alternatives, engine, px_per_mm, shot_ratio, resolvable,
                   quad, resolved_by, decision, decision_reason)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ident, device_id, time.time(), payload.get("name"), payload.get("dex"),
             payload.get("matched_token"), payload.get("edits"),
             int(bool(payload.get("corroborated"))),
             json.dumps(list(payload.get("alternatives") or [])), payload.get("engine"),
             payload.get("px_per_mm"), payload.get("shot_ratio"),
             None if payload.get("resolvable") is None else int(bool(payload["resolvable"])),
             json.dumps(payload.get("quad")) if payload.get("quad") is not None else None,
             payload.get("resolved_by"), payload.get("decision"), payload.get("decision_reason")),
        )
        self.conn.commit()
        return ident

    def get_identification(self, ident: str, device_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM identifications WHERE id = ? AND device_id = ?", (ident, device_id)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["alternatives"] = json.loads(d["alternatives"] or "[]")
        d["vote"] = json.loads(d["vote"]) if d.get("vote") else None
        return d

    def record_decision(self, ident: str, device_id: str, decision: str, reason: str,
                        vote: Optional[dict]) -> None:
        self.conn.execute(
            "UPDATE identifications SET decision = ?, decision_reason = ?, vote = ? "
            "WHERE id = ? AND device_id = ?",
            (decision, reason, json.dumps(vote) if vote is not None else None, ident, device_id),
        )
        self.conn.commit()

    # -- feedback and per-device priors ------------------------------------
    def add_feedback(self, device_id: str, action: str, *, identification_id: Optional[str] = None,
                     scan_id: Optional[int] = None, name=None, number=None, franchise=None,
                     note=None, owner: bool = False) -> dict:
        if action not in ("confirm", "correct", "reject"):
            raise FeedbackError("action must be confirm, correct or reject")
        ident = None
        if identification_id:
            ident = self.get_identification(identification_id, device_id)
            if ident is None:
                raise FeedbackError("unknown identification for this device")
        name = _clean(name, "name")
        franchise = _clean(franchise, "franchise", 40)
        note = _clean(note, "note", 500)
        if number is not None:
            try:
                number = int(number)
            except (TypeError, ValueError):
                raise FeedbackError("number must be an integer")
            if not 0 <= number < 100000:
                raise FeedbackError("number out of range")
        if action == "confirm":
            if ident is None or not ident.get("name"):
                raise FeedbackError("confirm needs an identification that named a card")
            name = name or ident["name"]
            if name.lower() != ident["name"].lower():
                raise FeedbackError("confirm names a different card; use correct")
            if number is None:
                number = ident.get("dex")
        if action == "correct" and not name:
            raise FeedbackError("correct needs the right name")
        now = time.time()
        cur = self.conn.execute(
            """INSERT INTO feedback (identification_id, scan_id, device_id, action, name, number,
                   franchise, note, owner, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (identification_id, scan_id, device_id, action, name, number, franchise, note,
             int(bool(owner)), now),
        )
        learned = {}
        if action in ("confirm", "correct") and name:
            self.conn.execute(
                """INSERT INTO device_priors (device_id, name, count, updated_at) VALUES (?,?,1,?)
                   ON CONFLICT(device_id, name) DO UPDATE SET count = count + 1, updated_at = ?""",
                (device_id, name, now, now),
            )
            learned["device_prior"] = name
        self.conn.commit()
        return {"feedback_id": int(cur.lastrowid), "name": name, "number": number,
                "franchise": franchise, "learned": learned, "identification": ident}

    def device_prior(self, device_id: str) -> dict[str, int]:
        return {r["name"]: int(r["count"]) for r in self.conn.execute(
            "SELECT name, count FROM device_priors WHERE device_id = ?", (device_id,))}

    def break_tie(self, device_id: str, alternatives) -> Optional[tuple[str, float]]:
        """Pick one of ``alternatives`` when this device's confirmed history
        favours it by ``TIE_MIN_RATIO`` in posterior odds (Dirichlet counts,
        alpha = 1) with at least ``TIE_MIN_COUNT`` confirmations. Returns
        (name, posterior probability among the alternatives) or None."""
        alts = [a for a in dict.fromkeys(alternatives or []) if a]
        if len(alts) < 2:
            return None
        prior = self.device_prior(device_id)
        lower = {k.lower(): v for k, v in prior.items()}
        counts = [(a, lower.get(a.lower(), 0)) for a in alts]
        counts.sort(key=lambda kv: -kv[1])
        (best, c1), (_, c2) = counts[0], counts[1]
        if c1 < TIE_MIN_COUNT or (c1 + PRIOR_ALPHA) < TIE_MIN_RATIO * (c2 + PRIOR_ALPHA):
            return None
        total = sum(c for _, c in counts) + PRIOR_ALPHA * len(counts)
        return best, (c1 + PRIOR_ALPHA) / total

    # -- prices --------------------------------------------------------------
    def add_price(self, device_id: str, attribution, *, identification_id: Optional[str] = None,
                  scan_id: Optional[int] = None) -> int:
        a = attribution
        cur = self.conn.execute(
            """INSERT INTO price_attributions (device_id, identification_id, scan_id, amount,
                   currency, quantity, source, scope, scope_key, kind, method, venue, observed_at,
                   raw_text, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (device_id, identification_id, scan_id, str(a.amount), a.currency, a.quantity,
             a.source, a.scope, a.scope_key or None, a.kind, a.method, a.venue or None,
             a.observed_at, a.raw_text or None, time.time()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def prices_for(self, device_id: str, identification_id: str):
        from .pricing import PriceAttribution

        rows = self.conn.execute(
            "SELECT * FROM price_attributions WHERE device_id = ? AND identification_id = ?",
            (device_id, identification_id)).fetchall()
        return [PriceAttribution(amount=r["amount"], currency=r["currency"], quantity=r["quantity"],
                                 source=r["source"], scope=r["scope"], scope_key=r["scope_key"] or "",
                                 kind=r["kind"], method=r["method"], venue=r["venue"] or "",
                                 observed_at=r["observed_at"], raw_text=r["raw_text"] or "")
                for r in rows]


def public_feedback_summary(conn: sqlite3.Connection, since: float = 0.0) -> list[dict]:
    """Aggregate feedback for owner review: no images, no device ids, only
    counts of how often a reading was confirmed or corrected to a name."""
    rows = conn.execute(
        """SELECT i.name AS ocr_name, i.matched_token AS ocr_token, f.action AS action,
                  f.name AS user_name, f.number AS user_number,
                  COUNT(*) AS n, COUNT(DISTINCT f.device_id) AS devices, MAX(f.created_at) AS last_at
           FROM feedback f LEFT JOIN identifications i ON i.id = f.identification_id
           WHERE f.created_at >= ? AND f.owner = 0
           GROUP BY i.name, i.matched_token, f.action, f.name, f.number
           ORDER BY n DESC""",
        (since,),
    ).fetchall()
    return [dict(zip(("ocr_name", "ocr_token", "action", "user_name", "user_number", "n",
                      "devices", "last_at"), tuple(r))) for r in rows]


# ---------------------------------------------------------------------------
# Owner devices: photo retention and the private inbox
# ---------------------------------------------------------------------------

def owner_token_matches(header_value: Optional[str]) -> bool:
    expected = os.environ.get("CARDCENTER_OWNER_TOKEN", "")
    if len(expected) < 24 or not header_value:
        return False
    return hmac.compare_digest(expected.encode(), header_value.strip().encode())


class _PendingPhotos:
    """Owner photos held in memory between /identify and /feedback."""

    def __init__(self, max_items: int = 32, ttl_s: float = 900.0):
        self._items: "OrderedDict[str, tuple[float, bytes]]" = OrderedDict()
        self._lock = threading.Lock()
        self.max_items, self.ttl_s = max_items, ttl_s

    def put(self, key: str, data: bytes) -> None:
        with self._lock:
            self._items[key] = (time.time(), data)
            self._items.move_to_end(key)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    def pop(self, key: str) -> Optional[bytes]:
        with self._lock:
            item = self._items.pop(key, None)
        if item is None or time.time() - item[0] > self.ttl_s:
            return None
        return item[1]


PENDING = _PendingPhotos()


def _image_ext(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def write_inbox(identification: dict, feedback: dict, price: Optional[dict], photo: bytes,
                device_id: str, inbox: Optional[str] = None) -> Optional[Path]:
    """Write the confirmed photo and a sidecar into the private inbox.
    Returns the sidecar path, or None when no inbox is configured."""
    root = inbox or os.environ.get("CARDCENTER_INSITU_INBOX")
    if not root:
        return None
    day = _dt.date.today().isoformat()
    folder = Path(root) / day
    folder.mkdir(parents=True, exist_ok=True)
    ident = identification["id"]
    img_path = folder / f"{ident}{_image_ext(photo)}"
    side = folder / f"{ident}.json"
    tmp = img_path.with_suffix(img_path.suffix + ".tmp")
    tmp.write_bytes(photo)
    tmp.replace(img_path)
    body = {
        "schema": "bakugo-insitu/1",
        "image": img_path.name,
        "device_id": device_id,
        "confirmed_by": f"owner-device:{device_id}",
        "identification": {k: identification.get(k) for k in
                           ("id", "name", "dex", "matched_token", "corroborated", "px_per_mm",
                            "quad", "decision", "vote", "created_at")},
        "label": {"name": feedback.get("name"), "number": feedback.get("number"),
                  "franchise": feedback.get("franchise"), "action": feedback.get("action")},
        "price": price,
        "written_at": time.time(),
    }
    tmp = side.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, indent=1, default=str), encoding="utf-8")
    tmp.replace(side)
    return side
