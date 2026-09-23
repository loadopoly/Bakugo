"""Step 1b: the manifest -- one DuckDB table keyed by SHA-256, in the
``trainer`` schema of the local vault (``supabase_vault.duckdb``).

Loose files are not the unit of record. Every image and every rasterised PDF
page gets a row keyed by the SHA-256 of its bytes; the diff between the mirror
and this table is how the loop knows something is new.

Label discipline is structural, not a convention:

* ``files.label`` / ``files.label_strength`` hold only ``weak`` (parsed from a
  filename or folder) or ``confirmed`` (a human confirmed it via
  ``confirm()``). The CHECK constraint rejects anything else.
* Pseudo-labels from the current system go to ``pseudo_labels``, a separate
  table. Nothing copies them into ``files.label``; ``confirm()`` requires an
  explicit ``confirmed_by``.
* Every confirmation is appended to ``confirmations`` so a label's provenance
  can be audited.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional

from .paths import Layout
from .sync import load_listing

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
PDF_EXTS = {".pdf"}
RASTER_DPI = 200

SOURCE_TYPES = ("binder_front", "binder_back", "labelled_duplicate", "sleeved", "bulk_pile",
                "app_capture", "unknown")
FRANCHISES = ("pokemon", "animal_crossing", "magic_the_gathering", "yugioh")

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    sha256         VARCHAR PRIMARY KEY,
    kind           VARCHAR NOT NULL CHECK (kind IN ('image', 'pdf', 'pdf_page')),
    parent_sha256  VARCHAR,
    page_index     INTEGER,
    rel_path       VARCHAR NOT NULL,
    drive_file_id  VARCHAR,
    drive_path     VARCHAR,
    franchise      VARCHAR,
    capture_date   DATE,
    session_id     VARCHAR,
    source_type    VARCHAR NOT NULL DEFAULT 'unknown',
    label          VARCHAR,
    label_number   INTEGER,
    label_strength VARCHAR CHECK (label_strength IS NULL OR label_strength IN ('weak', 'confirmed')),
    card_uid       VARCHAR,
    page_key       VARCHAR,   -- binder page / box / lot / venue the item was priced under
    box_key        VARCHAR,
    lot_key        VARCHAR,
    venue          VARCHAR,
    px_per_mm      DOUBLE,
    nd_hash        VARCHAR,   -- 256-bit difference hash, hex (near-duplicate linkage)
    split          VARCHAR CHECK (split IS NULL OR split IN ('train', 'test', 'quarantine')),
    split_note     VARCHAR,
    first_seen     TIMESTAMP NOT NULL,
    last_seen      TIMESTAMP NOT NULL,
    present        BOOLEAN NOT NULL DEFAULT TRUE
);
CREATE TABLE IF NOT EXISTS file_paths (
    sha256        VARCHAR NOT NULL,
    rel_path      VARCHAR NOT NULL,
    drive_file_id VARCHAR,
    PRIMARY KEY (sha256, rel_path)
);
CREATE TABLE IF NOT EXISTS scan_cache (
    rel_path VARCHAR PRIMARY KEY,
    size     BIGINT NOT NULL,
    mtime_ns BIGINT NOT NULL,
    sha256   VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS rasters (
    pdf_sha256 VARCHAR PRIMARY KEY,
    dpi        INTEGER NOT NULL,
    renderer   VARCHAR NOT NULL,
    pages      INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL,
    error      VARCHAR
);
CREATE TABLE IF NOT EXISTS pseudo_labels (
    sha256     VARCHAR NOT NULL,
    model_id   VARCHAR NOT NULL,
    label      VARCHAR,
    number     INTEGER,
    confidence DOUBLE,
    resolved   BOOLEAN NOT NULL,
    ambiguous  BOOLEAN NOT NULL,
    detail     VARCHAR,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (sha256, model_id)
);
CREATE SEQUENCE IF NOT EXISTS confirmations_seq;
CREATE TABLE IF NOT EXISTS confirmations (
    seq          BIGINT PRIMARY KEY DEFAULT nextval('confirmations_seq'),
    sha256       VARCHAR NOT NULL,
    label        VARCHAR,
    label_number INTEGER,
    card_uid     VARCHAR,
    franchise    VARCHAR,
    source_type  VARCHAR,
    px_per_mm    DOUBLE,
    confirmed_by VARCHAR NOT NULL,
    confirmed_at TIMESTAMP NOT NULL,
    note         VARCHAR
);
CREATE SEQUENCE IF NOT EXISTS price_attributions_seq;
CREATE TABLE IF NOT EXISTS price_attributions (
    seq          BIGINT PRIMARY KEY DEFAULT nextval('price_attributions_seq'),
    sha256       VARCHAR,          -- item scope: the photo of the item
    card_uid     VARCHAR,          -- item scope: the physical card
    scope        VARCHAR NOT NULL CHECK (scope IN ('item','page','box','lot','venue')),
    scope_key    VARCHAR,
    amount       DECIMAL(14, 4) NOT NULL,
    currency     VARCHAR NOT NULL,
    quantity     INTEGER NOT NULL DEFAULT 1,
    source       VARCHAR NOT NULL,
    kind         VARCHAR NOT NULL,
    method       VARCHAR NOT NULL,
    venue        VARCHAR,
    observed_at  TIMESTAMP NOT NULL,
    raw_text     VARCHAR,
    evidence     VARCHAR,
    recorded_by  VARCHAR NOT NULL,
    recorded_at  TIMESTAMP NOT NULL,
    CHECK (scope <> 'item' OR sha256 IS NOT NULL OR card_uid IS NOT NULL),
    CHECK (scope = 'item' OR scope_key IS NOT NULL)
);
CREATE TABLE IF NOT EXISTS quad_labels (
    sha256       VARCHAR NOT NULL,
    card_index   INTEGER NOT NULL,
    corners      VARCHAR NOT NULL,   -- JSON [[x,y]*4], image pixels, TL,TR,BR,BL
    confirmed_by VARCHAR NOT NULL,
    confirmed_at TIMESTAMP NOT NULL,
    PRIMARY KEY (sha256, card_index)
);
CREATE TABLE IF NOT EXISTS frozen_test (
    sha256    VARCHAR PRIMARY KEY,
    frozen_at TIMESTAMP NOT NULL
);
CREATE SEQUENCE IF NOT EXISTS test_uses_seq;
CREATE TABLE IF NOT EXISTS test_uses (
    seq      BIGINT PRIMARY KEY DEFAULT nextval('test_uses_seq'),
    model_id VARCHAR NOT NULL,
    task     VARCHAR NOT NULL,
    n_items  INTEGER NOT NULL,
    used_at  TIMESTAMP NOT NULL
);
CREATE SEQUENCE IF NOT EXISTS promotions_seq;
CREATE TABLE IF NOT EXISTS promotions (
    seq          BIGINT PRIMARY KEY DEFAULT nextval('promotions_seq'),
    task         VARCHAR NOT NULL,
    candidate_id VARCHAR NOT NULL,
    champion_id  VARCHAR,
    promoted     BOOLEAN NOT NULL,
    reason       VARCHAR NOT NULL,
    report       VARCHAR,
    decided_at   TIMESTAMP NOT NULL
);
"""


def _now() -> _dt.datetime:
    return _dt.datetime.now().replace(microsecond=0)


def connect(layout: Layout, read_only: bool = False):
    """Open the vault and switch to the ``trainer`` schema.

    DuckDB allows one writing process per file, so the trainer and the
    Supabase transfer (``cardcenter --transfer-supabase``) must not run at the
    same time.
    """
    import duckdb
    from cardcenter.vault import TRAINER_SCHEMA

    layout.vault_db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(layout.vault_db), read_only=read_only)
    if not read_only:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {TRAINER_SCHEMA}")
    con.execute(f"USE {TRAINER_SCHEMA}")
    if not read_only:
        con.execute(SCHEMA)
    return con


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Weak metadata from paths. Everything here is a guess and is stored as such.
# --------------------------------------------------------------------------

_CAMERA_RE = re.compile(r"^(?:PXL|IMG|VID|DSC|MVIMG|Screenshot)[_-]?(\d{8})", re.I)
_DATE_RE = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})")
_TRAILING_NUM_RE = re.compile(r"^(?P<name>.*?)[\s_#-]+(?P<num>\d{1,4})$")

_SOURCE_KEYWORDS = (
    ("labelled_duplicate", ("duplicate", "duplicates", "dupes", "dups", "dup")),
    ("binder_back", ("back", "backs")),
    ("sleeved", ("sleeve", "sleeved", "sleeves")),
    ("bulk_pile", ("bulk", "pile", "piles")),
    ("binder_front", ("binder", "front", "fronts")),
)


def _norm(s: str) -> str:
    import unicodedata

    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return re.sub(r"[\s\-]+", "_", s.strip().lower())


def infer_franchise(parts: Iterable[str]) -> Optional[str]:
    for part in parts:
        n = _norm(part)
        for f in FRANCHISES:
            if n == f or n.startswith(f + "_") or n.endswith("_" + f):
                return f
        if n in ("mtg", "magic"):
            return "magic_the_gathering"
        if n in ("yu_gi_oh", "yugioh!"):
            return "yugioh"
    return None


def infer_source_type(parts: Iterable[str]) -> str:
    tokens = set()
    for part in parts:
        tokens.update(t for t in re.split(r"[^a-z]+", _norm(part)) if t)
    for source, words in _SOURCE_KEYWORDS:
        if tokens.intersection(words):
            return source
    return "unknown"


def infer_capture_date(name: str, fallback: Optional[str] = None) -> Optional[_dt.date]:
    m = _CAMERA_RE.match(name) or _DATE_RE.search(name)
    if m:
        digits = "".join(m.groups()) if len(m.groups()) > 1 else m.group(1)
        try:
            return _dt.datetime.strptime(digits, "%Y%m%d").date()
        except ValueError:
            pass
    if fallback:
        try:
            return _dt.datetime.fromisoformat(fallback.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def weak_label_from_name(stem: str) -> tuple[Optional[str], Optional[int]]:
    """A camera filename carries no label. Anything else is taken as a weak
    ``name[_number]`` label -- to be confirmed, never trusted."""
    if _CAMERA_RE.match(stem) or not re.search(r"[A-Za-z]{3,}", stem):
        return None, None
    m = _TRAILING_NUM_RE.match(stem)
    name, num = (m.group("name"), int(m.group("num"))) if m else (stem, None)
    name = re.sub(r"[_\-]+", " ", name).strip()
    return (name or None), num


@dataclass
class Scanned:
    rel_path: str
    abs_path: Path
    sha256: str
    kind: str
    drive: dict = field(default_factory=dict)


def _walk(mirror: Path):
    for p in sorted(mirror.rglob("*")):
        if p.is_file():
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS:
                yield p, "image"
            elif ext in PDF_EXTS:
                yield p, "pdf"


def _hash_cached(con, rel: str, path: Path) -> str:
    st = path.stat()
    row = con.execute(
        "SELECT sha256 FROM scan_cache WHERE rel_path = ? AND size = ? AND mtime_ns = ?",
        [rel, st.st_size, st.st_mtime_ns],
    ).fetchone()
    if row:
        return row[0]
    digest = sha256_file(path)
    con.execute(
        "INSERT OR REPLACE INTO scan_cache VALUES (?, ?, ?, ?)",
        [rel, st.st_size, st.st_mtime_ns, digest],
    )
    return digest


ND_HASH_SIZE = 16


def near_dup_hash(path: Path, size: int = ND_HASH_SIZE) -> Optional[str]:
    """256-bit difference hash of the whole photo, as hex.

    ``cardcenter.multicard.dhash`` (64 bits) is built for card crops. On whole
    photos with a large uniform background it produces sparse hashes that sit
    a few bits apart for unrelated scenes, and the split's transitive linkage
    then chains a whole folder into one group. 256 bits separates unrelated
    scenes by tens of bits while a re-encode or exposure change moves it by
    one or two. It does not survive re-framing or cropping; the same card shot
    twice is linked by its confirmed ``card_uid`` instead.
    """
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:0{size * size // 4}x}"


def rasterise_pdf(pdf: Path, out_dir: Path, dpi: int = RASTER_DPI) -> tuple[list[Path], str]:
    """Render every page once. Returns (page paths, renderer id)."""
    import pypdfium2 as pdfium

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf))
    pages = []
    try:
        for i in range(len(doc)):
            target = out_dir / f"p{i + 1:04d}.png"
            if not target.exists():
                bitmap = doc[i].render(scale=dpi / 72.0)
                bitmap.to_pil().save(target)
            pages.append(target)
    finally:
        doc.close()
    version = getattr(pdfium, "PYPDFIUM_INFO", None) or getattr(pdfium, "__version__", "?")
    return pages, f"pypdfium2 {version}"


def _upsert(con, *, sha, kind, rel, drive, parts, stem, now, parent=None, page=None, dh=None):
    exists = con.execute("SELECT 1 FROM files WHERE sha256 = ?", [sha]).fetchone()
    con.execute(
        "INSERT OR IGNORE INTO file_paths VALUES (?, ?, ?)", [sha, rel, drive.get("ID")]
    )
    if exists:
        con.execute(
            "UPDATE files SET last_seen = ?, present = TRUE WHERE sha256 = ?", [now, sha]
        )
        return False
    franchise = infer_franchise(parts)
    capture = infer_capture_date(stem, drive.get("ModTime"))
    folder = "/".join(parts[:-1]) if len(parts) > 1 else ""
    label, number = (weak_label_from_name(stem)
                     if kind == "image" and not rel.startswith("inbox/") else (None, None))
    con.execute(
        """INSERT INTO files (sha256, kind, parent_sha256, page_index, rel_path,
               drive_file_id, drive_path, franchise, capture_date, session_id,
               source_type, label, label_number, label_strength, nd_hash,
               first_seen, last_seen, present)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE)""",
        [sha, kind, parent, page, rel, drive.get("ID"), drive.get("Path"),
         franchise, capture, f"{capture or 'undated'}:{folder}",
         infer_source_type(parts[:-1]) if len(parts) > 1 else "unknown",
         label, number, "weak" if label else None, dh, now, now],
    )
    return True


def refresh(layout: Layout, con=None, rasterise=rasterise_pdf, hasher=near_dup_hash) -> dict:
    """Scan the mirror, add new rows, mark vanished ones. Returns the diff."""
    own = con is None
    con = con or connect(layout)
    listing = load_listing(layout)
    now = _now()
    new, errors = [], []
    seen_rel = set()
    roots = [(layout.mirror, "")]
    if layout.inbox.exists():
        roots.append((layout.inbox, "inbox/"))
    walked = [(p, k, prefix + p.relative_to(base).as_posix())
              for base, prefix in roots for p, k in _walk(base)]
    try:
        con.execute("BEGIN")
        for path, kind, rel in walked:
            seen_rel.add(rel)
            sha = _hash_cached(con, rel, path)
            parts = PurePosixPath(rel).parts
            drive = listing.get(rel, {})
            dh = hasher(path) if kind == "image" else None
            if _upsert(con, sha=sha, kind=kind, rel=rel, drive=drive, parts=parts,
                       stem=path.stem, now=now, dh=dh):
                new.append(sha)
            if kind == "pdf":
                done = con.execute("SELECT 1 FROM rasters WHERE pdf_sha256 = ? AND error IS NULL",
                                   [sha]).fetchone()
                if done:
                    continue
                try:
                    pages, renderer = rasterise(path, layout.raster / sha, RASTER_DPI)
                except Exception as exc:  # recorded, retried next run
                    con.execute("INSERT OR REPLACE INTO rasters VALUES (?, ?, ?, 0, ?, ?)",
                                [sha, RASTER_DPI, "pypdfium2", now, str(exc)])
                    errors.append(f"{rel}: {exc}")
                    continue
                for i, page in enumerate(pages):
                    psha = sha256_file(page)
                    prel = f"raster/{page.relative_to(layout.raster).as_posix()}"
                    if _upsert(con, sha=psha, kind="pdf_page", rel=prel, drive=drive,
                               parts=parts, stem=path.stem, now=now, parent=sha,
                               page=i, dh=hasher(page)):
                        new.append(psha)
                con.execute("INSERT OR REPLACE INTO rasters VALUES (?, ?, ?, ?, ?, NULL)",
                            [sha, RASTER_DPI, renderer, len(pages), now])
        # A file is present if any of its paths was seen this run. Pages follow
        # their PDF.
        rows = con.execute("SELECT sha256, rel_path FROM file_paths").fetchall()
        present = {s for s, r in rows if r in seen_rel}
        missing = []
        for (sha, kind, parent, was) in con.execute(
                "SELECT sha256, kind, parent_sha256, present FROM files").fetchall():
            is_present = (parent in present) if kind == "pdf_page" else (sha in present)
            if was and not is_present:
                missing.append(sha)
            if was != is_present:
                con.execute("UPDATE files SET present = ? WHERE sha256 = ?", [is_present, sha])
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        if own:
            con.close()
    return {"new": new, "missing": missing, "raster_errors": errors, "scanned": len(seen_rel)}


def resolve_path(layout: Layout, rel_path: str) -> Path:
    if rel_path.startswith("raster/"):
        return layout.raster / rel_path[len("raster/"):]
    if rel_path.startswith("inbox/"):
        return layout.inbox / rel_path[len("inbox/"):]
    return layout.mirror / rel_path


# --------------------------------------------------------------------------
# Human confirmation -- the only way a label becomes 'confirmed'.
# --------------------------------------------------------------------------

_HUMAN_RESERVED = {"trainer", "pseudo", "model", "auto", "system"}


def _require_human(who: str, what: str) -> str:
    if not who or not who.strip():
        raise ValueError(f"{what} needs a non-empty confirmed_by")
    if who.strip().lower() in _HUMAN_RESERVED:
        raise ValueError(f"confirmed_by={who!r} is reserved for automated steps")
    return who.strip()


def confirm(con, sha256: str, *, confirmed_by: str, label: Optional[str] = None,
            number: Optional[int] = None, card_uid: Optional[str] = None,
            franchise: Optional[str] = None, source_type: Optional[str] = None,
            px_per_mm: Optional[float] = None, note: str = "",
            page_key: Optional[str] = None, box_key: Optional[str] = None,
            lot_key: Optional[str] = None, venue: Optional[str] = None) -> None:
    confirmed_by = _require_human(confirmed_by, "confirm()")
    if source_type is not None and source_type not in SOURCE_TYPES:
        raise ValueError(f"source_type must be one of {SOURCE_TYPES}")
    if franchise is not None and franchise not in FRANCHISES:
        raise ValueError(f"franchise must be one of {FRANCHISES}")
    if not con.execute("SELECT 1 FROM files WHERE sha256 = ?", [sha256]).fetchone():
        raise KeyError(f"no manifest row for {sha256}")
    now = _now()
    con.execute(
        """INSERT INTO confirmations (sha256, label, label_number, card_uid, franchise,
               source_type, px_per_mm, confirmed_by, confirmed_at, note)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [sha256, label, number, card_uid, franchise, source_type, px_per_mm,
         confirmed_by.strip(), now, note],
    )
    con.execute(
        """UPDATE files SET
               label = COALESCE(?, label),
               label_number = COALESCE(?, label_number),
               label_strength = 'confirmed',
               card_uid = COALESCE(?, card_uid),
               franchise = COALESCE(?, franchise),
               source_type = COALESCE(?, source_type),
               px_per_mm = COALESCE(?, px_per_mm),
               page_key = COALESCE(?, page_key),
               box_key = COALESCE(?, box_key),
               lot_key = COALESCE(?, lot_key),
               venue = COALESCE(?, venue)
           WHERE sha256 = ?""",
        [label, number, card_uid, franchise, source_type, px_per_mm,
         page_key, box_key, lot_key, venue, sha256],
    )


def confirm_quad(con, sha256: str, card_index: int, corners, *, confirmed_by: str) -> None:
    confirmed_by = _require_human(confirmed_by, "confirm_quad()")
    pts = [[float(x), float(y)] for x, y in corners]
    if len(pts) != 4:
        raise ValueError("a card quad has four corners")
    con.execute(
        "INSERT OR REPLACE INTO quad_labels VALUES (?, ?, ?, ?, ?)",
        [sha256, int(card_index), json.dumps(pts), confirmed_by, _now()],
    )


# --------------------------------------------------------------------------
# Price attribution (any source: sticker, tag, sign, page, lot, receipt, ...)
# --------------------------------------------------------------------------

def attribute_price(con, attribution, *, recorded_by: str, sha256: Optional[str] = None,
                    card_uid: Optional[str] = None) -> None:
    """Record one ``cardcenter.pricing.PriceAttribution``. Item-scope prices
    name the item (``sha256`` or ``card_uid``); wider scopes name their
    ``scope_key`` and reach items through ``files.page_key``/``box_key``/
    ``lot_key``/``venue``."""
    import datetime as dt

    recorded_by = _require_human(recorded_by, "attribute_price()")
    a = attribution
    con.execute(
        """INSERT INTO price_attributions (sha256, card_uid, scope, scope_key, amount, currency,
               quantity, source, kind, method, venue, observed_at, raw_text, evidence,
               recorded_by, recorded_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [sha256, card_uid, a.scope, a.scope_key or None, str(a.amount), a.currency, a.quantity,
         a.source, a.kind, a.method, a.venue or None,
         dt.datetime.fromtimestamp(a.observed_at).replace(microsecond=0),
         a.raw_text or None, a.evidence or None, recorded_by, _now()],
    )


_SCOPE_COLUMN = {"page": "page_key", "box": "box_key", "lot": "lot_key", "venue": "venue"}


def item_price(con, sha256: str, kind: str = "asking"):
    """Resolve the general price of one item from every attribution that
    covers it. Returns ``cardcenter.pricing.ResolvedPrice`` or None."""
    from cardcenter.pricing import PriceAttribution, resolve_item_price

    row = con.execute(
        "SELECT card_uid, page_key, box_key, lot_key, venue FROM files WHERE sha256 = ?", [sha256]
    ).fetchone()
    if row is None:
        raise KeyError(sha256)
    card_uid, keys = row[0], dict(zip(("page_key", "box_key", "lot_key", "venue"), row[1:]))
    clauses, args = ["(scope = 'item' AND (sha256 = ? OR (card_uid IS NOT NULL AND card_uid = ?)))"], [sha256, card_uid]
    for scope, col in _SCOPE_COLUMN.items():
        if keys[col]:
            clauses.append("(scope = ? AND scope_key = ?)")
            args += [scope, keys[col]]
    rows = con.execute(
        "SELECT amount, currency, quantity, source, scope, scope_key, kind, method, venue, "
        "epoch(observed_at), raw_text, evidence FROM price_attributions WHERE "
        + " OR ".join(clauses), args).fetchall()
    atts = [PriceAttribution(amount=str(r[0]), currency=r[1], quantity=r[2], source=r[3],
                             scope=r[4], scope_key=r[5] or "", kind=r[6], method=r[7],
                             venue=r[8] or "", observed_at=float(r[9]), raw_text=r[10] or "",
                             evidence=r[11] or "") for r in rows]
    return resolve_item_price(atts, kind=kind)
