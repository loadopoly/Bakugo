"""Build-time guard: the public image ships only generic, lineage-tagged
artifacts.

The private trainer learns from a personal collection. Some of what it learns
is generic (a detector threshold, an OCR preprocessing constant, a calibrated
card-back hue) and may ship. Some of it *encodes the collection* (a card index,
priced priors, a sticker dataset, collector-number frequencies) and must never
reach the public app, the tunnel, or a public git remote.

This module is stdlib-only and runs twice:

* in the public ``Dockerfile``, before the package is installed, so a build
  that contains a disallowed artifact fails;
* from ``tests/test_release_guard.py`` over the working tree.

An artifact is a directory ``cardcenter/data/released/<kind>/`` holding
``artifact.json`` and the payload it names. It passes only if:

1. ``artifact.json`` parses and has ``schema == "bakugo-artifact/1"``;
2. ``exposure == "generic"`` and ``kind`` is on ``GENERIC_KINDS`` and not on
   ``COLLECTION_KINDS`` (the kind list is the control; the exposure field is a
   declaration that must agree with it);
3. the payload's SHA-256 matches ``payload_sha256``;
4. the payload is a flat JSON object of at most ``MAX_PAYLOAD_KEYS`` scalar
   values -- a card index or a price table cannot hide inside a "generic"
   artifact as a nested list;
5. ``lineage`` records the training-set digest and count, and
   ``confirmed_only`` is true.

Anywhere under the scanned root, the guard also refuses any ``artifact.json``
whose exposure is not generic, and files that belong only to the private tier
(``*.duckdb``, ``drive_listing.json``, ``rclone.conf``, review queues).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

SCHEMA = "bakugo-artifact/1"
GENERIC_KINDS = frozenset({"quad_detect", "back_hue", "ocr_preprocess"})
COLLECTION_KINDS = frozenset({
    "card_index", "priced_priors", "sticker_dataset", "number_priors",
    "collection_lookup", "embedding_index", "review_queue",
})
MAX_PAYLOAD_KEYS = 64
PRIVATE_TIER_PATTERNS = (
    "*.duckdb", "*.duckdb.wal", "drive_listing.json", "rclone.conf", "*.rclone.conf",
    "review-*.jsonl", "attest.key",
)
RELEASED_DIR = Path("data") / "released"


class ReleaseRefused(Exception):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_artifact(art_dir: Path) -> dict:
    """Validate one released artifact directory. Returns its metadata."""
    meta_path = art_dir / "artifact.json"
    if not meta_path.is_file():
        raise ReleaseRefused(f"{art_dir}: no artifact.json (untagged content)")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ReleaseRefused(f"{meta_path}: unreadable ({exc})")
    if meta.get("schema") != SCHEMA:
        raise ReleaseRefused(f"{meta_path}: schema {meta.get('schema')!r} != {SCHEMA!r}")
    kind = meta.get("kind")
    if kind in COLLECTION_KINDS:
        raise ReleaseRefused(f"{meta_path}: kind {kind!r} encodes collection content")
    if kind not in GENERIC_KINDS:
        raise ReleaseRefused(f"{meta_path}: kind {kind!r} is not on the generic allowlist")
    if meta.get("exposure") != "generic":
        raise ReleaseRefused(f"{meta_path}: exposure {meta.get('exposure')!r} is not 'generic'")
    if art_dir.name != kind:
        raise ReleaseRefused(f"{meta_path}: directory {art_dir.name!r} != kind {kind!r}")
    payload_name = meta.get("payload")
    if not payload_name or Path(payload_name).name != payload_name:
        raise ReleaseRefused(f"{meta_path}: payload must be a bare filename")
    payload_path = art_dir / payload_name
    if not payload_path.is_file():
        raise ReleaseRefused(f"{meta_path}: payload {payload_name} missing")
    if _sha256(payload_path) != meta.get("payload_sha256"):
        raise ReleaseRefused(f"{payload_path}: SHA-256 does not match artifact.json")
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ReleaseRefused(f"{payload_path}: not JSON ({exc})")
    if not isinstance(payload, dict) or len(payload) > MAX_PAYLOAD_KEYS:
        raise ReleaseRefused(f"{payload_path}: payload must be a flat object of <= {MAX_PAYLOAD_KEYS} keys")
    for k, v in payload.items():
        if not isinstance(v, (int, float, str, bool)) or (isinstance(v, str) and len(v) > 256):
            raise ReleaseRefused(f"{payload_path}: key {k!r} is not a short scalar")
    lineage = meta.get("lineage") or {}
    if not lineage.get("train_set_digest") or not isinstance(lineage.get("n_train"), int):
        raise ReleaseRefused(f"{meta_path}: lineage lacks train_set_digest / n_train")
    if lineage.get("confirmed_only") is not True:
        raise ReleaseRefused(f"{meta_path}: lineage does not assert confirmed_only")
    extra = {p.name for p in art_dir.iterdir()} - {"artifact.json", payload_name}
    if extra:
        raise ReleaseRefused(f"{art_dir}: unexpected files {sorted(extra)}")
    return meta


def scan(package_root: Path) -> list[str]:
    """Return a list of refusal messages (empty means the tree may ship)."""
    package_root = Path(package_root)
    problems: list[str] = []
    for path in package_root.rglob("*"):
        if not path.is_file():
            continue
        if any(fnmatch.fnmatch(path.name, pat) for pat in PRIVATE_TIER_PATTERNS):
            problems.append(f"{path}: private-tier file in public build tree")
        if path.name == "artifact.json":
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                problems.append(f"{path}: unreadable artifact.json")
                continue
            if meta.get("exposure") != "generic" or meta.get("kind") in COLLECTION_KINDS:
                problems.append(f"{path}: collection-derived artifact in public build tree")
    released = package_root / RELEASED_DIR
    if released.exists():
        for entry in sorted(released.iterdir()):
            if not entry.is_dir():
                problems.append(f"{entry}: stray file in released/ (artifacts are directories)")
                continue
            try:
                check_artifact(entry)
            except ReleaseRefused as exc:
                problems.append(str(exc))
    return problems


def released_params(kind: str, package_root: Optional[Path] = None) -> Optional[dict]:
    """Payload of a released generic artifact, or None. Re-validates on read,
    so a hand-edited payload is ignored rather than trusted."""
    root = Path(package_root) if package_root else Path(__file__).resolve().parent
    art = root / RELEASED_DIR / kind
    if not art.is_dir():
        return None
    try:
        meta = check_artifact(art)
    except ReleaseRefused:
        return None
    return json.loads((art / meta["payload"]).read_text(encoding="utf-8"))


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0]) if args else Path(__file__).resolve().parent
    problems = scan(root)
    if problems:
        print("release guard: REFUSED", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"release guard: ok ({root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
