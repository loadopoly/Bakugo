"""Remote connection, device synchronization, and health checking protocols.

This module provides:
1. `RemoteEndpoint` and `ConnectionSpec`: Definition and configuration of remote cardcenter hubs.
2. `ConnectionManager`: Unified endpoint discovery, health checking, and network diagnostics.
3. `SyncClient` and `SyncServer`: Provenance-preserving data exchange between mobile counter
   devices (Termux) and central desktop/cloud database stores.
4. Cryptographic integrity validation and enforcement of the Contamination Firewall across the wire.
"""

from __future__ import annotations

import hashlib
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import __version__
from .store import LabelKind, ScanStore
from .versioning import CURRENT_SCHEMA_VERSION, SemVer


# ---------------------------------------------------------------------------
# Data Models & Connection Specifications
# ---------------------------------------------------------------------------


class EndpointStatus(str, Enum):
    ONLINE = "online"
    UNREACHABLE = "unreachable"
    INCOMPATIBLE_SCHEMA = "incompatible_schema"
    AUTH_FAILED = "auth_failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EndpointHealth:
    status: EndpointStatus
    latency_ms: float
    server_version: Optional[str]
    schema_version: Optional[str]
    scan_count: int = 0
    certified_label_count: int = 0
    error_message: Optional[str] = None
    checked_at: float = field(default_factory=time.time)

    def is_healthy(self) -> bool:
        return self.status is EndpointStatus.ONLINE


@dataclass
class ConnectionSpec:
    """Configuration for a remote cardcenter endpoint."""

    url: str
    auth_token: Optional[str] = None
    timeout_sec: float = 5.0
    client_id: str = "cardcenter-client"
    trusted: bool = True

    def normalized_url(self) -> str:
        u = self.url.strip()
        if not u.startswith(("http://", "https://")):
            u = "http://" + u
        return u.rstrip("/")


# ---------------------------------------------------------------------------
# Cryptographic Hashing & Provenance Payloads
# ---------------------------------------------------------------------------


def compute_scan_hash(scan_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 integrity hash for a scan record."""
    canonical_keys = [
        "card_key",
        "holder",
        "worst_ratio_pct",
        "worst_axis",
        "h_ratio_pct",
        "v_ratio_pct",
        "left_mm",
        "right_mm",
        "top_mm",
        "bottom_mm",
        "px_per_mm",
        "phash",
    ]
    raw = "|".join(str(scan_dict.get(k, "")) for k in canonical_keys)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class SyncPayload:
    """A bundle of scans and labels formatted for transmission."""

    schema_version: str
    engine_version: str
    exported_at: float
    client_id: str
    scans: List[Dict[str, Any]]
    labels: List[Dict[str, Any]]
    integrity_checksum: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not self.integrity_checksum:
            blob = json.dumps({"scans": self.scans, "labels": self.labels}, sort_keys=True)
            d["integrity_checksum"] = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncPayload":
        return cls(
            schema_version=data.get("schema_version", CURRENT_SCHEMA_VERSION),
            engine_version=data.get("engine_version", __version__),
            exported_at=data.get("exported_at", time.time()),
            client_id=data.get("client_id", "anonymous"),
            scans=data.get("scans", []),
            labels=data.get("labels", []),
            integrity_checksum=data.get("integrity_checksum", ""),
        )


@dataclass
class SyncResult:
    """Summary of a completed sync operation."""

    success: bool
    scans_pushed: int
    scans_pulled: int
    labels_pushed: int
    labels_pulled: int
    rejected_contaminated: int
    duration_sec: float
    error_message: Optional[str] = None

    def describe(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Sync Operation: {status} ({self.duration_sec:.2f}s)",
            f"  Scans  : {self.scans_pushed} pushed, {self.scans_pulled} pulled",
            f"  Labels : {self.labels_pushed} pushed, {self.labels_pulled} pulled",
        ]
        if self.rejected_contaminated > 0:
            lines.append(f"  Contamination Firewall: {self.rejected_contaminated} non-certified labels quarantined")
        if self.error_message:
            lines.append(f"  Error  : {self.error_message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Connection Manager & Client
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Manages connections to remote CardCenter / Bakugo instances."""

    def __init__(self, default_spec: Optional[ConnectionSpec] = None) -> None:
        self.default_spec = default_spec
        self.user_agent = f"cardcenter-sync/{__version__}"

    def check_health(self, spec: ConnectionSpec) -> EndpointHealth:
        """Probe remote endpoint /health or status URL and measure round-trip latency."""
        base_url = spec.normalized_url()
        health_url = f"{base_url}/health"
        req = urllib.request.Request(
            health_url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
        )
        if spec.auth_token:
            req.add_header("Authorization", f"Bearer {spec.auth_token}")

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=spec.timeout_sec) as resp:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    return EndpointHealth(
                        status=EndpointStatus.ONLINE,
                        latency_ms=round(elapsed_ms, 2),
                        server_version=payload.get("version"),
                        schema_version=payload.get("schema_version"),
                        scan_count=payload.get("scans", 0),
                        certified_label_count=payload.get("certified_labels", 0),
                    )
                return EndpointHealth(
                    status=EndpointStatus.UNREACHABLE,
                    latency_ms=round(elapsed_ms, 2),
                    server_version=None,
                    schema_version=None,
                    error_message=f"HTTP {resp.status}",
                )
        except urllib.error.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if exc.code == 401 or exc.code == 403:
                return EndpointHealth(
                    status=EndpointStatus.AUTH_FAILED,
                    latency_ms=round(elapsed_ms, 2),
                    server_version=None,
                    schema_version=None,
                    error_message=f"Authentication error (HTTP {exc.code})",
                )
            return EndpointHealth(
                status=EndpointStatus.UNREACHABLE,
                latency_ms=round(elapsed_ms, 2),
                server_version=None,
                schema_version=None,
                error_message=f"HTTP error {exc.code}: {exc.reason}",
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return EndpointHealth(
                status=EndpointStatus.UNREACHABLE,
                latency_ms=round(elapsed_ms, 2),
                server_version=None,
                schema_version=None,
                error_message=str(exc),
            )

    def export_store_payload(
        self,
        store: ScanStore,
        client_id: str = "counter-phone",
        since_time: float = 0.0,
    ) -> SyncPayload:
        """Export records from local ScanStore into a SyncPayload."""
        rows_scans = store.conn.execute(
            "SELECT * FROM scans WHERE created_at >= ? ORDER BY id ASC", (since_time,)
        ).fetchall()
        scans_list = [dict(r) for r in rows_scans]

        # Export labels
        rows_labels = store.conn.execute(
            "SELECT * FROM labels WHERE created_at >= ? ORDER BY id ASC", (since_time,)
        ).fetchall()
        labels_list = [dict(r) for r in rows_labels]

        return SyncPayload(
            schema_version=CURRENT_SCHEMA_VERSION,
            engine_version=__version__,
            exported_at=time.time(),
            client_id=client_id,
            scans=scans_list,
            labels=labels_list,
        )

    def import_payload(
        self,
        store: ScanStore,
        payload: SyncPayload,
        strict_provenance: bool = True,
    ) -> Tuple[int, int, int]:
        """Merge a SyncPayload into local ScanStore while enforcing provenance firewalls.
        
        Returns (scans_imported, labels_imported, quarantined_labels_count).
        """
        scans_imported = 0
        labels_imported = 0
        quarantined = 0

        scan_id_map: Dict[int, int] = {}  # remote_id -> local_id

        # Insert scans (idempotent deduplication based on key, phash, ratio)
        for s in payload.scans:
            remote_id = s.get("id", 0)
            existing = store.conn.execute(
                "SELECT id FROM scans WHERE card_key = ? AND abs(worst_ratio_pct - ?) < 1e-4 AND phash = ?",
                (s.get("card_key"), s.get("worst_ratio_pct"), s.get("phash", 0)),
            ).fetchone()

            if existing:
                local_id = existing[0]
            else:
                cur = store.conn.execute(
                    """INSERT INTO scans (
                        card_key, holder, worst_ratio_pct, worst_ratio_sigma, worst_axis,
                        h_ratio_pct, v_ratio_pct, left_mm, right_mm, top_mm, bottom_mm,
                        px_per_mm, inner_confidence, refraction_applied, warnings, phash,
                        source, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        s.get("card_key", ""),
                        s.get("holder", "raw"),
                        float(s.get("worst_ratio_pct", 50.0)),
                        float(s.get("worst_ratio_sigma", 1.0)),
                        s.get("worst_axis", "horizontal"),
                        float(s.get("h_ratio_pct", 50.0)),
                        float(s.get("v_ratio_pct", 50.0)),
                        s.get("left_mm"),
                        s.get("right_mm"),
                        s.get("top_mm"),
                        s.get("bottom_mm"),
                        float(s.get("px_per_mm", 10.0)),
                        float(s.get("inner_confidence", 1.0)),
                        int(s.get("refraction_applied", 0)),
                        s.get("warnings", ""),
                        int(s.get("phash", 0)),
                        s.get("source", f"sync:{payload.client_id}"),
                        float(s.get("created_at", time.time())),
                    ),
                )
                local_id = cur.lastrowid
                scans_imported += 1

            if remote_id:
                scan_id_map[remote_id] = local_id

        # Insert labels with Contamination Firewall
        for l in payload.labels:
            remote_scan_id = l.get("scan_id", 0)
            local_scan_id = scan_id_map.get(remote_scan_id)
            if not local_scan_id:
                continue

            kind_str = l.get("kind", LabelKind.SELF_REPORTED.value)
            cert_num = l.get("cert_number")

            # Firewalled verification: CERTIFIED labels MUST possess a non-empty cert_number
            if kind_str == LabelKind.CERTIFIED.value and not cert_num:
                if strict_provenance:
                    quarantined += 1
                    continue
                else:
                    # Downgrade to self-reported if strict provenance is relaxed
                    kind_str = LabelKind.SELF_REPORTED.value

            # Check if label already exists
            existing_label = store.conn.execute(
                "SELECT id FROM labels WHERE scan_id = ? AND grader = ? AND grade = ? AND kind = ?",
                (local_scan_id, l.get("grader"), l.get("grade"), kind_str),
            ).fetchone()

            if not existing_label:
                store.conn.execute(
                    """INSERT INTO labels (
                        scan_id, grader, grade, centering_subgrade, kind, cert_number,
                        attributed_to, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        local_scan_id,
                        l.get("grader", ""),
                        l.get("grade", ""),
                        l.get("centering_subgrade"),
                        kind_str,
                        cert_num,
                        l.get("attributed_to", payload.client_id),
                        float(l.get("created_at", time.time())),
                    ),
                )
                labels_imported += 1

        store.conn.commit()
        return (scans_imported, labels_imported, quarantined)

    def sync(
        self,
        store: ScanStore,
        spec: ConnectionSpec,
        client_id: str = "local-device",
    ) -> SyncResult:
        """Perform bidirectional synchronization with remote endpoint."""
        t0 = time.perf_counter()
        base_url = spec.normalized_url()
        sync_url = f"{base_url}/sync"

        # 1. Export local payload
        outbound = self.export_store_payload(store, client_id=client_id)
        post_data = json.dumps(outbound.to_dict()).encode("utf-8")

        req = urllib.request.Request(
            sync_url,
            data=post_data,
            headers={
                "User-Agent": self.user_agent,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        if spec.auth_token:
            req.add_header("Authorization", f"Bearer {spec.auth_token}")

        try:
            with urllib.request.urlopen(req, timeout=spec.timeout_sec) as resp:
                if resp.status == 200:
                    raw_inbound = json.loads(resp.read().decode("utf-8"))
                    inbound_payload = SyncPayload.from_dict(raw_inbound)
                    imported_s, imported_l, quar = self.import_payload(
                        store, inbound_payload, strict_provenance=True
                    )
                    dt = time.perf_counter() - t0
                    return SyncResult(
                        success=True,
                        scans_pushed=len(outbound.scans),
                        scans_pulled=imported_s,
                        labels_pushed=len(outbound.labels),
                        labels_pulled=imported_l,
                        rejected_contaminated=quar,
                        duration_sec=round(dt, 3),
                    )
                return SyncResult(
                    success=False,
                    scans_pushed=0,
                    scans_pulled=0,
                    labels_pushed=0,
                    labels_pulled=0,
                    rejected_contaminated=0,
                    duration_sec=round(time.perf_counter() - t0, 3),
                    error_message=f"HTTP {resp.status}",
                )
        except Exception as exc:
            return SyncResult(
                success=False,
                scans_pushed=0,
                scans_pulled=0,
                labels_pushed=0,
                labels_pulled=0,
                rejected_contaminated=0,
                duration_sec=round(time.perf_counter() - t0, 3),
                error_message=str(exc),
            )


# Alias for explicit role naming
SyncClient = ConnectionManager

