"""Split by physical card, never by image.

The duplicates folder holds the same card on different dates and sleeved and
unsleeved. An image-level split puts near-copies on both sides and every score
it reports is inflated. So the unit of assignment is a connected component
over two kinds of link:

* the same confirmed ``card_uid`` (one physical card, any capture), and
* near-duplicate pixels: 256-bit difference-hash Hamming distance <=
  ``nd_radius`` (the same photo re-exported, re-encoded, or re-exposed;
  see ``manifest.near_dup_hash`` for what it does not catch).

``link_sessions=True`` also links every item from one capture session. That
is the stricter reading of "split by capture session", and on a binder
session it collapses a whole page set into one group, so it is off by
default; turn it on when sessions are small (e.g. one duplicate shot per
session).

Freezing. A test item, once assigned, is written to ``frozen_test`` and never
leaves the test split. If a later confirmation links a component that holds a
frozen test item to items already assigned ``train``, those train items move
to ``quarantine`` (excluded from training and from evaluation) -- the test
side is never the one that moves, because the champion's scores were measured
on it. Unassigned new items in such a component join ``test``.

Assignment of a fresh component is deterministic: a keyed hash of its
smallest SHA-256 against ``test_fraction``, so re-running is stable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

SPLIT_SALT = "bakugo-split-v1"


@dataclass(frozen=True)
class SplitConfig:
    test_fraction: float = 0.2
    nd_radius: int = 12          # of 256 bits
    max_component_frac: float = 0.25   # warn when one group swallows this much
    link_sessions: bool = False
    salt: str = SPLIT_SALT


class _DSU:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _bucket_hash(key: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def components(rows, cfg: SplitConfig) -> dict[str, list[str]]:
    """rows: iterable of (sha256, card_uid, nd_hash_hex, session_id). Returns
    {component_root: [sha256, ...]}."""
    rows = list(rows)
    dsu = _DSU([r[0] for r in rows])
    by_uid: dict[str, str] = {}
    by_session: dict[str, str] = {}
    hashed = []
    for sha, uid, dh, session in rows:
        if uid:
            if uid in by_uid:
                dsu.union(sha, by_uid[uid])
            else:
                by_uid[uid] = sha
        if cfg.link_sessions and session:
            if session in by_session:
                dsu.union(sha, by_session[session])
            else:
                by_session[session] = sha
        if dh is not None:
            hashed.append((sha, int(dh, 16)))
    # O(n^2) over hashed items. Collections here are thousands, not millions;
    # replace with a BK-tree if that changes.
    for i in range(len(hashed)):
        si, hi = hashed[i]
        for j in range(i + 1, len(hashed)):
            sj, hj = hashed[j]
            if _hamming(hi, hj) <= cfg.nd_radius:
                dsu.union(si, sj)
    comps: dict[str, list[str]] = {}
    for sha, *_ in rows:
        comps.setdefault(dsu.find(sha), []).append(sha)
    return comps


def assign(con, cfg: SplitConfig = SplitConfig(), now=None) -> dict:
    """Assign/repair splits for every trainable row. Returns counts."""
    import datetime as _dt

    now = now or _dt.datetime.now().replace(microsecond=0)
    rows = con.execute(
        "SELECT sha256, card_uid, nd_hash, session_id FROM files WHERE kind IN ('image', 'pdf_page')"
    ).fetchall()
    current = dict(con.execute(
        "SELECT sha256, split FROM files WHERE kind IN ('image', 'pdf_page')").fetchall())
    frozen = {r[0] for r in con.execute("SELECT sha256 FROM frozen_test").fetchall()}

    changes: dict[str, tuple[str, Optional[str]]] = {}
    stats = {"test": 0, "train": 0, "quarantined": 0, "components": 0}
    comps = components(rows, cfg)
    largest = max((len(m) for m in comps.values()), default=0)
    stats["largest_component"] = largest
    if rows and len(rows) >= 20 and largest > cfg.max_component_frac * len(rows):
        stats["warning"] = (f"one group holds {largest} of {len(rows)} items; the split has "
                            "little to work with. Check card_uid confirmations and nd_radius.")
    for root, members in comps.items():
        stats["components"] += 1
        has_test = any(m in frozen or current.get(m) == "test" for m in members)
        has_train = any(current.get(m) == "train" for m in members)
        if has_test:
            for m in members:
                if m in frozen or current.get(m) == "test":
                    if current.get(m) != "test":
                        changes[m] = ("test", "frozen test item restored")
                elif current.get(m) == "train":
                    changes[m] = ("quarantine",
                                  f"linked to frozen test component {root[:12]} after training use")
                    stats["quarantined"] += 1
                elif current.get(m) is None:
                    changes[m] = ("test", f"joined frozen test component {root[:12]}")
        else:
            if has_train:
                target = "train"
            else:
                target = "test" if _bucket_hash(min(members), cfg.salt) < cfg.test_fraction else "train"
            for m in members:
                if current.get(m) is None:
                    changes[m] = (target, None)

    con.execute("BEGIN")
    try:
        for sha, (split, note) in changes.items():
            con.execute("UPDATE files SET split = ?, split_note = COALESCE(?, split_note) WHERE sha256 = ?",
                        [split, note, sha])
            if split == "test" and sha not in frozen:
                con.execute("INSERT OR IGNORE INTO frozen_test VALUES (?, ?)", [sha, now])
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    for split, n in con.execute(
            "SELECT split, count(*) FROM files WHERE kind IN ('image','pdf_page') GROUP BY split").fetchall():
        if split in ("test", "train"):
            stats[split] = n
    stats["changed"] = len(changes)
    return stats


def leakage_report(con, cfg: SplitConfig = SplitConfig()) -> list[dict]:
    """Components that contain both a usable train item and a test item. Must
    be empty after assign(); a test pins that."""
    rows = con.execute(
        "SELECT sha256, card_uid, nd_hash, session_id FROM files WHERE kind IN ('image', 'pdf_page')"
    ).fetchall()
    split = dict(con.execute("SELECT sha256, split FROM files").fetchall())
    bad = []
    for root, members in components(rows, cfg).items():
        kinds = {split.get(m) for m in members}
        if "train" in kinds and "test" in kinds:
            bad.append({"component": root, "members": members})
    return bad
