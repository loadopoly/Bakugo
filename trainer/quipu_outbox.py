"""Proposals to QUIPU. Written, never realised.

The trainer writes JSON proposals into ``<private>/outbox/quipu/``. Nothing in
this package sends them anywhere or touches QUIPU's database. Realising a
proposal is QUIPU's business and needs ``QUIPU_REALISE_GRANT_REF`` plus
accepted attestations (Invariance #7); ``isolation`` makes sure the trainer
process cannot see either.

Each proposal file is created exclusively (never overwritten) and records
``status: proposed``, ``realised: false`` and what realisation would require.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from collections import Counter
from pathlib import Path

from .paths import Layout

REQUIRES = ("QUIPU_REALISE_GRANT_REF", "approved attestations (two distinct signers at gate 6)")


def propose(layout: Layout, kind: str, payload: dict, evidence: dict) -> Path:
    body = {
        "schema": "bakugo-quipu-proposal/1",
        "source": "bakugo-trainer",
        "kind": kind,
        "status": "proposed",
        "realised": False,
        "authorised": False,
        "requires": list(REQUIRES),
        "exposure": "collection",
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "payload": payload,
        "evidence": evidence,
    }
    text = json.dumps(body, indent=1, sort_keys=True)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    layout.quipu_outbox.mkdir(parents=True, exist_ok=True)
    path = layout.quipu_outbox / f"{_dt.date.today().isoformat()}-{kind}-{digest}.json"
    if path.exists():
        return path          # identical proposal already filed today
    with open(path, "x", encoding="utf-8") as fh:
        fh.write(text)
    return path


def number_priors_payload(con) -> tuple[dict, dict]:
    """Collector-number frequencies from confirmed TRAIN labels, in the shape
    of QUIPU's ``numeric_lexicon`` (what ``quipu_client.number_priors`` reads).
    Test labels are excluded so the priors cannot carry test answers into the
    recognition evaluation."""
    rows = con.execute(
        "SELECT sha256, label_number FROM files WHERE split = 'train' "
        "AND label_strength = 'confirmed' AND label_number IS NOT NULL AND present"
    ).fetchall()
    counts = Counter(str(n) for _, n in rows)
    payload = {"numeric_lexicon": [{"token": t, "freq": c} for t, c in sorted(counts.items())]}
    evidence = {
        "n_items": len(rows),
        "train_set_digest": hashlib.sha256(
            "\n".join(sorted(s for s, _ in rows)).encode()).hexdigest(),
        "confirmed_only": True,
    }
    return payload, evidence


def propose_number_priors(con, layout: Layout):
    payload, evidence = number_priors_payload(con)
    if not payload["numeric_lexicon"]:
        return None
    return propose(layout, "number_priors", payload, evidence)
