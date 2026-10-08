# Vendored from QUIPU src/quipu/edge_client.py (v0.48.0). Keep identical: QUIPU verifies exactly this protocol.
"""edge_client — the one way anything outside QUIPU writes into it.

Single file, stdlib only, safe to vendor: copy it into any client (a fleet
container, a host script, a server on another machine) unchanged.

    from quipu_edge_client import QuipuEdgeClient
    q = QuipuEdgeClient("perceptopoly")          # env: QUIPU_URL, QUIPU_EDGE_KEY
    q.observe("standoff 2.4 m bearing 118", meta={"frame": {...}})

What it guarantees
------------------
* **Nothing is lost.**  Every write goes first into a local SQLite outbox
  (``QUIPU_OUTBOX``, default ``~/.quipu/outbox-<source>.sqlite``) and is
  removed only when QUIPU has answered 2xx.  QUIPU down, the network down, the
  process killed mid-send: the write stays and is sent later.
* **Nothing is counted twice.**  Each write carries an idempotency key
  (``X-Quipu-Idempotency``, sha256 of source, route and body unless the caller
  gives one).  QUIPU records the keys it has applied and answers a repeat with
  ``duplicate: true`` without learning from it again, so retries are safe.
* **It never breaks the caller.**  ``observe``/``feedback``/``submit`` only
  enqueue; sending happens on a daemon thread; every error is caught, counted
  and visible in ``stats()``.
* **It is who it says it is.**  With ``QUIPU_EDGE_KEY`` each request is signed
  (HMAC-SHA256 over method, path, timestamp, body hash and the idempotency key)
  as the client's source.  The idempotency key is inside the signature, so it
  cannot be swapped on a captured request.
* **It obeys the edge.**  429 waits for ``Retry-After``; 401/403 hold the write
  and retry slowly (the operator has not issued a key or granted the source
  yet — the write is kept, not dropped); 400 is permanent (the payload itself
  is refused) and the row is kept, marked ``rejected``, for inspection; 5xx and
  network errors back off exponentially up to one hour.
* **It is bounded.**  The queued payloads are capped (``QUIPU_OUTBOX_MAX_MB``,
  default 256).  When full, new writes are refused and counted (``dropped_full``);
  nothing already queued is discarded.

Remote servers
--------------
Point ``QUIPU_URL`` at the TLS front door the operator publishes (``https://``);
plain ``http://`` is accepted only for loopback, private bridge addresses and
container names, so a key never crosses the internet in clear.

翈 — write once, keep until heard, never say it twice.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import os
import random
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

__all__ = ["QuipuEdgeClient", "sign_headers", "idempotency_key"]
__version__ = "1.0"

_LOG = logging.getLogger("quipu.edge_client")
_BACKOFF_MAX_S = 3600.0
_AUTH_HOLD_S = 900.0

H_SOURCE, H_TS, H_SIG, H_IDEM = "X-Quipu-Source", "X-Quipu-Timestamp", "X-Quipu-Signature", "X-Quipu-Idempotency"


def idempotency_key(source: str, path: str, body: bytes) -> str:
    return hashlib.sha256(f"{source}\n{path}\n".encode() + body).hexdigest()[:40]


def sign_headers(key_hex: str, source: str, method: str, path: str, body: bytes,
                 idem: str | None = None, ts: float | None = None) -> dict[str, str]:
    """Headers for one request (the canonical form QUIPU's edge verifies)."""
    t = str(int(time.time() if ts is None else ts))
    msg = f"{method.upper()}\n{path}\n{t}\n{hashlib.sha256(body).hexdigest()}"
    if idem:
        msg += f"\n{idem}"
    mac = hmac.new(bytes.fromhex(key_hex), msg.encode(), hashlib.sha256).hexdigest()
    out = {H_SOURCE: source, H_TS: t, H_SIG: mac}
    if idem:
        out[H_IDEM] = idem
    return out


def _clear_text_ok(url: str) -> bool:
    p = urlparse(url)
    if p.scheme == "https":
        return True
    if p.scheme != "http":
        return False
    host = (p.hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "::1") or "." not in host:
        return True                      # loopback or a container / compose service name
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(host))
    except (OSError, ValueError):
        return False
    return ip.is_private or ip.is_loopback


_SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    idem        TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    body        BLOB NOT NULL,
    created     REAL NOT NULL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    next_at     REAL NOT NULL DEFAULT 0,
    state       TEXT NOT NULL DEFAULT 'queued',   -- queued | rejected
    last_code   INTEGER,
    last_error  TEXT
);
CREATE INDEX IF NOT EXISTS outbox_due ON outbox(state, next_at);
"""


class QuipuEdgeClient:
    def __init__(self, source: str, base_url: str | None = None, key: str | None = None,
                 outbox: str | os.PathLike | None = None, *, timeout: float = 5.0,
                 background: bool = True, max_mb: float | None = None,
                 opener=None, clock=time.time):
        self.source = source
        self.base_url = (base_url or os.environ.get("QUIPU_URL") or "http://127.0.0.1:7100").rstrip("/")
        self.key = (key if key is not None else os.environ.get("QUIPU_EDGE_KEY", "")).strip() or None
        path = outbox or os.environ.get("QUIPU_OUTBOX") or (Path.home() / ".quipu" / f"outbox-{source}.sqlite")
        self.outbox_path = Path(path)
        self.outbox_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = float(timeout)
        self.max_bytes = int(float(max_mb if max_mb is not None else os.environ.get("QUIPU_OUTBOX_MAX_MB", "256"))
                             * 1024 * 1024)
        self._open = opener or urllib.request.urlopen
        self._clock = clock
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._counts = {"enqueued": 0, "sent": 0, "duplicate": 0, "rejected": 0, "retry": 0,
                        "held_auth": 0, "dropped_full": 0, "errors": 0}
        self._db = sqlite3.connect(str(self.outbox_path), check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()
        if self.key:
            try:
                bytes.fromhex(self.key)
            except ValueError:
                _LOG.warning("QUIPU_EDGE_KEY is not hex; requests will be sent unsigned")
                self.key = None
        self._secure = _clear_text_ok(self.base_url)
        if not self._secure:
            _LOG.error("QUIPU_URL %s is plain http to a public host; writes are queued, not sent. "
                       "Use the operator's https front door.", self.base_url)
        self._thread = None
        if background:
            self._thread = threading.Thread(target=self._run, name=f"quipu-edge-{source}", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ enqueue
    def submit(self, path: str, payload: Mapping[str, Any], idem: str | None = None) -> Optional[str]:
        """Queue one write.  Returns its idempotency key, or None if it could not be queued."""
        try:
            body = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str).encode()
            k = idem or idempotency_key(self.source, path, body)
            with self._lock:
                size = int(self._db.execute("SELECT COALESCE(SUM(LENGTH(body)), 0) FROM outbox").fetchone()[0])
                if size + len(body) > self.max_bytes:
                    self._counts["dropped_full"] += 1
                    return None
                cur = self._db.execute(
                    "INSERT OR IGNORE INTO outbox(idem, path, body, created, next_at) VALUES(?,?,?,?,?)",
                    (k, path, body, self._clock(), 0.0))
                self._db.commit()
                if cur.rowcount:
                    self._counts["enqueued"] += 1
            self._wake.set()
            return k
        except Exception as exc:                      # never into the caller
            self._counts["errors"] += 1
            _LOG.warning("quipu outbox enqueue failed: %s", exc)
            return None

    def observe(self, text: str, **fields: Any) -> Optional[str]:
        return self.submit("/observe", {"source": self.source, "text": text, **fields})

    def feedback(self, expected: str, **fields: Any) -> Optional[str]:
        return self.submit("/feedback", {"source": self.source, "expected": expected, **fields})

    # ------------------------------------------------------------------ send
    def _send_one(self, row) -> None:
        idem, path, body, attempts = row
        now = self._clock()
        headers = {"Content-Type": "application/json", H_IDEM: idem}
        if self.key:
            headers.update(sign_headers(self.key, self.source, "POST", path, body, idem=idem, ts=now))
        req = urllib.request.Request(self.base_url + path, data=body, headers=headers, method="POST")
        code, err, retry_after = 0, None, None
        try:
            with self._open(req, timeout=self.timeout) as resp:
                code = int(getattr(resp, "status", 200))
                payload = json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            code = int(exc.code)
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                payload = json.loads(exc.read().decode("utf-8") or "{}")
            except Exception:
                payload = {}
            err = str(payload.get("error") or exc.reason)
        except Exception as exc:
            code, payload, err = 0, {}, f"{type(exc).__name__}: {exc}"
        with self._lock:
            if 200 <= code < 300:
                self._db.execute("DELETE FROM outbox WHERE idem = ?", (idem,))
                self._counts["duplicate" if payload.get("duplicate") else "sent"] += 1
            elif code == 400 or code == 413:
                self._db.execute("UPDATE outbox SET state='rejected', attempts=?, last_code=?, last_error=? "
                                 "WHERE idem=?", (attempts + 1, code, err, idem))
                self._counts["rejected"] += 1
            else:
                if code == 429:
                    try:
                        wait = max(1.0, float(retry_after))
                    except (TypeError, ValueError):
                        wait = 60.0
                elif code in (401, 403):
                    wait = _AUTH_HOLD_S
                    self._counts["held_auth"] += 1
                else:
                    wait = min(_BACKOFF_MAX_S, 2.0 ** min(attempts, 12)) * (0.5 + random.random() / 2)
                self._db.execute("UPDATE outbox SET attempts=?, next_at=?, last_code=?, last_error=? WHERE idem=?",
                                 (attempts + 1, now + wait, code or None, err, idem))
                self._counts["retry"] += 1
            self._db.commit()

    def flush(self, max_items: int = 100) -> dict:
        """Send what is due now (the background thread calls this; tests call it directly)."""
        if not self._secure:
            return self.stats()
        with self._lock:
            rows = self._db.execute(
                "SELECT idem, path, body, attempts FROM outbox WHERE state='queued' AND next_at <= ? "
                "ORDER BY created LIMIT ?", (self._clock(), int(max_items))).fetchall()
        for row in rows:
            try:
                self._send_one(row)
            except Exception as exc:
                self._counts["errors"] += 1
                _LOG.warning("quipu outbox send failed: %s", exc)
        return self.stats()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.flush()
                with self._lock:
                    nxt = self._db.execute("SELECT MIN(next_at) FROM outbox WHERE state='queued'").fetchone()[0]
            except Exception:
                nxt = None
            wait = 30.0 if nxt is None else max(0.2, min(30.0, nxt - self._clock()))
            self._wake.wait(wait)
            self._wake.clear()

    def close(self, drain_s: float = 2.0) -> None:
        deadline = time.time() + drain_s
        while time.time() < deadline and self.pending():
            self.flush()
            time.sleep(0.1)
        self._stop.set()
        self._wake.set()

    # ------------------------------------------------------------------ look
    def pending(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM outbox WHERE state='queued'").fetchone()[0])

    def stats(self) -> dict:
        with self._lock:
            rej = int(self._db.execute("SELECT COUNT(*) FROM outbox WHERE state='rejected'").fetchone()[0])
            q = int(self._db.execute("SELECT COUNT(*) FROM outbox WHERE state='queued'").fetchone()[0])
        return {"source": self.source, "url": self.base_url, "signed": bool(self.key), "secure": self._secure,
                "queued": q, "rejected_kept": rej, **self._counts}
