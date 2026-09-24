"""Keep what the reader was given at the counter, so a miss can be replayed.

Every field report so far arrived as phone screenshots: the preview with the
HUD over it, at screen resolution, the bottom tenth under a banner. The
frames had to be rebuilt from them (inpainting the overlay) before the reader
could be run on them again, and the stills behind "Measure Card" were never
seen at all -- the 2.19.1 report's Freeze failures ("could not locate a
card-shaped quadrilateral" on a card in plain view) could only be reasoned
about. This keeps the exact inputs on the server that received them:

* every still sent to /measure and /identify, with the outcome;
* live frames the AR session measured (at most one every two seconds per
  session), with the outline and the numbers it produced.

Files go to ``CARDCENTER_FIELD_DIR`` (default: ``field/`` beside the
database), as ``<UTC time>_<kind>.jpg`` + ``.json``. The newest
``CARDCENTER_FIELD_LOG`` pairs are kept (default 200; 0 turns it off). This
is the owner's own server and the owner's own photos; nothing leaves it.
Recording never fails a request.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_lock = threading.Lock()
_seq = [0]


def _limit() -> int:
    try:
        return max(0, int(os.environ.get("CARDCENTER_FIELD_LOG", "200")))
    except ValueError:
        return 200


def field_dir() -> Optional[Path]:
    d = os.environ.get("CARDCENTER_FIELD_DIR")
    if d:
        return Path(d)
    db = os.environ.get("CARDCENTER_DB")
    if not db:
        return None
    return Path(db).resolve().parent / "field"


def record(kind: str, image_bytes: bytes, meta: dict) -> Optional[Path]:
    """Write one input and what came of it. Returns the image path, or None
    when recording is off or failed."""
    limit = _limit()
    root = field_dir()
    if limit <= 0 or root is None or not image_bytes:
        return None
    try:
        with _lock:
            root.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc)
            _seq[0] = (_seq[0] + 1) % 1000
            stem = (now.strftime("%Y%m%dT%H%M%S_") + f"{now.microsecond // 1000:03d}"
                    + f"{_seq[0]:03d}_{kind}")
            img = root / (stem + ".jpg")
            img.write_bytes(image_bytes)
            doc = {"kind": kind, "utc": now.isoformat(), "unix": time.time(), **meta}
            (root / (stem + ".json")).write_text(json.dumps(doc, default=str, indent=1))
            jsons = sorted(root.glob("*.json"))
            for old in jsons[: max(0, len(jsons) - limit)]:
                old.unlink(missing_ok=True)
                old.with_suffix(".jpg").unlink(missing_ok=True)
            return img
    except Exception:  # pragma: no cover - never fail a request over this
        return None
