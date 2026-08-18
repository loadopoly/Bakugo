# Versioning & Remote Connection Specification

Bakugo (`cardcenter`) incorporates a resilient versioning and distributed connection architecture designed for offline-first mobile usage (Termux), multi-device synchronization, and upstream repository alignment.

---

## 1. Semantic Versioning & Upstream Tracking

The core engine adheres to **SemVer 2.0.0** (`MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]`):
* **`SemVer`**: Immutable dataclass with full comparison (`<`, `<=`, `==`, `>=`, `>`) and compatibility checking (`is_compatible_with`).
* **Canonical Upstream Repository**: [`https://github.com/PoodlesOfWar/Bakugo`](https://github.com/PoodlesOfWar/Bakugo).

### Upstream Update Checker
`cardcenter.versioning.check_for_updates()` connects to the GitHub API to check for new releases:

```bash
cardcenter --check-updates
```

```text
Checking upstream GitHub repository for updates...

Update Status: UP_TO_DATE
  Installed version : v1.9.0
  Latest upstream   : v1.9.0
  Release title     : Current Head
  Release URL       : https://github.com/PoodlesOfWar/Bakugo
```

---

## 2. Database Schema Versioning & Migrations

Local scans and provenance records are stored in SQLite. Database schemas are explicitly version-tagged:

| Schema Version | Introduced | Key Features |
|---|---|---|
| `cardcenter/1` | v1.0.0 | Base `scans` table and `labels` table with foreign key indices. |
| `cardcenter/2` | v1.9.0 | Added `_schema_metadata` table, `client_uuid`, `integrity_hash`, `synced_at`, and `provenance_hash` columns. |

### Migration Command
```bash
cardcenter --migrate-db my_cards.db
```

The migration engine automatically inspects the database PRAGMA info and applies forward alterations without data loss.

---

## 3. Remote Sync Protocol & Connection Manager

### Connection Specification (`ConnectionSpec`)
Represents a remote endpoint with optional bearer token authentication and timeout policies:

```python
from cardcenter.connection import ConnectionSpec, ConnectionManager

spec = ConnectionSpec(
    url="http://192.168.1.50:8765",
    auth_token="secret-token-if-configured",
    timeout_sec=5.0
)
mgr = ConnectionManager()
health = mgr.check_health(spec)
print(f"Status: {health.status}, Latency: {health.latency_ms}ms")
```

### Sync Payload Architecture
Sync packets (`SyncPayload`) include:
1. `schema_version`: Engine schema version string.
2. `client_id`: Node identifier (e.g. `counter-tablet-1`).
3. `scans`: Array of canonical scan records.
4. `labels`: Associated grade labels.
5. `integrity_checksum`: SHA-256 hash across the serialized payload.

---

## 4. Contamination Firewall

To safeguard active learning datasets from circular degradation:

$$\text{Certified Label} \implies \text{Non-empty } \texttt{cert\_number}$$

During synchronization:
1. Any incoming label marked as `certified` that lacks a valid `cert_number` is **quarantined** and rejected from the certified label pool.
2. Self-reported, crowd votes, and model predictions are safely imported with their explicit `kind` tag but excluded from training manifests.
3. Training exports (`store.export_training_set()`) enforce strict ground-truth purity.
