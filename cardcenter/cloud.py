"""Supabase mirror for Bakugo scans — same project as Loadopoly-OCR.

Local ``ScanStore`` is the source of truth. This module best-effort upserts
measurement metadata to ``bakugo_scans`` / ``bakugo_labels`` via PostgREST
and the **anon** key. It never reads or sends a service-role key.

Photos are not uploaded here. Certified labels without a cert number are
rejected before they touch the wire (same contamination firewall as
``ScanStore.add_label`` / ``ConnectionManager``).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from . import __version__
from .store import LabelKind, ScanStore

# Never treat these as the Bakugo client key.
_SERVICE_KEY_NAMES = (
    "SUPABASE_SERVICE_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SERVICE_ROLE_KEY",
)


@dataclass(frozen=True)
class CloudConfig:
    url: str
    anon_key: str
    timeout_sec: float = 8.0

    def rest_url(self, table: str) -> str:
        return f"{self.url.rstrip('/')}/rest/v1/{table}"


@dataclass
class CloudResult:
    ok: bool
    skipped: bool = False
    status: int = 0
    error: Optional[str] = None
    table: str = ""
    count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "skipped": self.skipped,
            "status": self.status,
            "error": self.error,
            "table": self.table,
            "count": self.count,
        }


def resolve_config(
    url: Optional[str] = None,
    anon_key: Optional[str] = None,
) -> Optional[CloudConfig]:
    """Resolve PostgREST credentials. Service-role keys are ignored."""
    resolved_url = (
        url
        or os.environ.get("CARDCENTER_SUPABASE_URL")
        or os.environ.get("VITE_SUPABASE_URL")
        or os.environ.get("SUPABASE_URL")
        or ""
    ).strip()
    resolved_key = (
        anon_key
        or os.environ.get("CARDCENTER_SUPABASE_ANON_KEY")
        or os.environ.get("VITE_SUPABASE_ANON_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or ""
    ).strip()
    if not resolved_url or not resolved_key:
        return None
    if not resolved_url.startswith(("http://", "https://")):
        resolved_url = "https://" + resolved_url
    return CloudConfig(url=resolved_url.rstrip("/"), anon_key=resolved_key)


def resolve_device_id(store: Optional[ScanStore] = None) -> str:
    env = (os.environ.get("CARDCENTER_DEVICE_ID") or "").strip()
    if env:
        return env
    if store is None:
        return "anonymous"
    row = store.conn.execute(
        "SELECT value FROM _schema_metadata WHERE key = 'device_id'"
    ).fetchone()
    if row and row[0]:
        return str(row[0])
    import uuid

    device_id = str(uuid.uuid4())
    store.conn.execute(
        "INSERT OR REPLACE INTO _schema_metadata (key, value, updated_at) "
        "VALUES ('device_id', ?, ?)",
        (device_id, time.time()),
    )
    store.conn.commit()
    return device_id


def firewall_label(kind: str, cert_number: Optional[str]) -> None:
    """Refuse certified-without-cert before any network write."""
    if str(kind).lower() == LabelKind.CERTIFIED.value and not (
        cert_number and str(cert_number).strip()
    ):
        raise ValueError(
            "refusing to sync a certified label without a cert number. "
            "That claim is self-reported; mislabelling it as certified "
            "poisons every training set built from the cloud mirror."
        )


def scan_to_row(
    scan: Mapping[str, Any],
    *,
    device_id: str,
    engine_version: str = __version__,
) -> Dict[str, Any]:
    return {
        "local_id": int(scan["id"]) if scan.get("id") is not None else None,
        "device_id": device_id,
        "card_key": scan.get("card_key") or "unidentified",
        "holder": scan.get("holder") or "raw",
        "worst_ratio_pct": scan.get("worst_ratio_pct"),
        "worst_ratio_sigma": scan.get("worst_ratio_sigma"),
        "worst_axis": scan.get("worst_axis"),
        "h_ratio_pct": scan.get("h_ratio_pct"),
        "v_ratio_pct": scan.get("v_ratio_pct"),
        "left_mm": scan.get("left_mm"),
        "right_mm": scan.get("right_mm"),
        "top_mm": scan.get("top_mm"),
        "bottom_mm": scan.get("bottom_mm"),
        "px_per_mm": scan.get("px_per_mm"),
        "inner_confidence": scan.get("inner_confidence"),
        "refraction_applied": bool(scan.get("refraction_applied")),
        "warnings": scan.get("warnings") or "",
        "phash": scan.get("phash") or 0,
        "source": scan.get("source") or "",
        "engine_version": engine_version,
    }


def label_to_row(
    label: Mapping[str, Any],
    *,
    device_id: str,
    scan_cloud_id: Optional[str] = None,
) -> Dict[str, Any]:
    kind = str(label.get("kind") or "")
    cert = label.get("cert_number")
    firewall_label(kind, cert)
    row: Dict[str, Any] = {
        "local_id": int(label["id"]) if label.get("id") is not None else None,
        "device_id": device_id,
        "grader": label.get("grader"),
        "grade": label.get("grade"),
        "centering_subgrade": label.get("centering_subgrade"),
        "kind": kind,
        "cert_number": cert,
        "attributed_to": label.get("attributed_to"),
    }
    if scan_cloud_id:
        row["scan_id"] = scan_cloud_id
    return row


def _postgrest(
    cfg: CloudConfig,
    table: str,
    rows: List[Dict[str, Any]],
    *,
    on_conflict: str = "device_id,local_id",
) -> CloudResult:
    if not rows:
        return CloudResult(ok=True, skipped=True, table=table, count=0)
    body = json.dumps(rows if len(rows) > 1 else rows[0]).encode("utf-8")
    req = urllib.request.Request(
        f"{cfg.rest_url(table)}?on_conflict={on_conflict}",
        data=body,
        method="POST",
        headers={
            "apikey": cfg.anon_key,
            "Authorization": f"Bearer {cfg.anon_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
            "User-Agent": f"cardcenter-cloud/{__version__}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_sec) as resp:
            return CloudResult(ok=True, status=int(resp.status), table=table, count=len(rows))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        return CloudResult(
            ok=False,
            status=int(exc.code),
            table=table,
            count=len(rows),
            error=f"HTTP {exc.code}: {detail}",
        )
    except urllib.error.URLError as exc:
        return CloudResult(ok=False, table=table, count=len(rows), error=str(exc.reason))
    except TimeoutError as exc:
        return CloudResult(ok=False, table=table, count=len(rows), error=str(exc))


def upsert_scan(
    scan: Mapping[str, Any],
    *,
    device_id: str,
    config: Optional[CloudConfig] = None,
) -> CloudResult:
    cfg = config or resolve_config()
    if cfg is None:
        return CloudResult(ok=True, skipped=True, table="bakugo_scans")
    return _postgrest(cfg, "bakugo_scans", [scan_to_row(scan, device_id=device_id)])


def upsert_label(
    label: Mapping[str, Any],
    *,
    device_id: str,
    scan_cloud_id: Optional[str] = None,
    config: Optional[CloudConfig] = None,
) -> CloudResult:
    cfg = config or resolve_config()
    if cfg is None:
        return CloudResult(ok=True, skipped=True, table="bakugo_labels")
    row = label_to_row(label, device_id=device_id, scan_cloud_id=scan_cloud_id)
    return _postgrest(cfg, "bakugo_labels", [row])


def sync_scan_id(store: ScanStore, scan_id: int) -> CloudResult:
    """Best-effort upsert of one local scan. Never raises."""
    cfg = resolve_config()
    if cfg is None:
        return CloudResult(ok=True, skipped=True, table="bakugo_scans")
    row = store.get_scan(scan_id)
    if not row:
        return CloudResult(ok=False, table="bakugo_scans", error=f"scan {scan_id} not found")
    device_id = resolve_device_id(store)
    result = upsert_scan(row, device_id=device_id, config=cfg)
    if result.ok and not result.skipped:
        store.mark_synced(scan_id)
    return result


def sync_store(store: ScanStore, *, include_labels: bool = False) -> CloudResult:
    """Push unsynced local scans (and their labels) to Supabase."""
    cfg = resolve_config()
    if cfg is None:
        return CloudResult(ok=True, skipped=True, table="bakugo_scans")
    device_id = resolve_device_id(store)
    scans = store.unsynced_scans()
    if not scans:
        return CloudResult(ok=True, skipped=True, table="bakugo_scans", count=0)
    scan_result = _postgrest(
        cfg,
        "bakugo_scans",
        [scan_to_row(s, device_id=device_id) for s in scans],
    )
    if scan_result.ok:
        for s in scans:
            store.mark_synced(int(s["id"]))
    if not include_labels or not scan_result.ok:
        return scan_result
    labels: List[Dict[str, Any]] = []
    for s in scans:
        for lab in store.labels_for_scan(int(s["id"])):
            try:
                labels.append(label_to_row(lab, device_id=device_id))
            except ValueError:
                continue
    if not labels:
        return scan_result
    label_result = _postgrest(cfg, "bakugo_labels", labels)
    if not label_result.ok:
        return label_result
    label_result.count = scan_result.count
    return label_result
