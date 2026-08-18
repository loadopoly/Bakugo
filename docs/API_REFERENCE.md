# Python API Reference

This document covers the core modules and programmatic APIs available in `cardcenter`.

---

## 1. Top-Level Imports (`cardcenter`)

```python
from cardcenter import (
    measure_centering,
    grade_band,
    all_grade_bands,
    available_graders,
    GradeBand,
    CenteringResult,
    BorderPair,
    Measured,
    SlabSpec,
    CaptureSpec,
    SemVer,
    VersionInfo,
    EngineCapabilities,
    UpdateCheckResult,
    get_version_info,
    get_engine_capabilities,
    check_for_updates,
    migrate_database,
    ConnectionManager,
    ConnectionSpec,
    EndpointHealth,
    SyncPayload,
    SyncResult,
)
```

---

## 2. Metrology & Centering Pipeline

### `cardcenter.centering.measure_centering`
```python
def measure_centering(
    image: np.ndarray,
    slab: SlabSpec | str = "raw",
    capture: CaptureSpec = CaptureSpec(),
    target_ppm: float = 12.0,
    face: str = "front",
) -> CenteringResult:
    """Measure card centering ratio with honest 95% confidence intervals."""
```

### `cardcenter.types.CenteringResult`
Key Attributes:
* `horizontal: BorderPair`: Left/right border widths and ratio.
* `vertical: BorderPair`: Top/bottom border widths and ratio.
* `worst_axis: BorderPair`: The binding axis determining grade ceiling.
* `worst_ratio: Measured`: Value, standard error $\sigma$, and 95% CI.
* `quality: DetectionQuality`: Outer line-fit residual, inner edge confidence, and refraction flags.
* `rectified: np.ndarray`: Rectified card image in canonical metric space.

---

## 3. Versioning & Capabilities (`cardcenter.versioning`)

### `cardcenter.versioning.get_version_info() -> VersionInfo`
Returns build metadata, Python version, platform architecture, OpenCV/NumPy versions, and Git commit hash.

### `cardcenter.versioning.get_engine_capabilities() -> EngineCapabilities`
Inspects environment and returns availability of metrology solvers, OCR, AR, active learning, and sync protocols.

### `cardcenter.versioning.check_for_updates(repo='PoodlesOfWar/Bakugo') -> UpdateCheckResult`
Queries GitHub releases and tags for updates.

### `cardcenter.versioning.migrate_database(db_path: str) -> str`
Applies forward SQLite schema migrations to the latest `cardcenter/2` format.

---

## 4. Remote Connection & Synchronization (`cardcenter.connection`)

### `cardcenter.connection.ConnectionManager`
```python
class ConnectionManager:
    def check_health(self, spec: ConnectionSpec) -> EndpointHealth: ...
    def export_store_payload(self, store: ScanStore, client_id: str, since_time: float = 0.0) -> SyncPayload: ...
    def import_payload(self, store: ScanStore, payload: SyncPayload, strict_provenance: bool = True) -> Tuple[int, int, int]: ...
    def sync(self, store: ScanStore, spec: ConnectionSpec, client_id: str = "local-device") -> SyncResult: ...
```

---

## 5. Storage & Provenance Store (`cardcenter.store`)

### `cardcenter.store.ScanStore`
```python
class ScanStore:
    def __init__(self, path: str = "cardcenter.db"): ...
    def add_scan(self, card_key: str, result: CenteringResult, phash: int = 0, source: str = "") -> int: ...
    def add_label(self, scan_id: int, grader: str, grade: str, kind: LabelKind, cert_number: Optional[str] = None) -> int: ...
    def export_training_set(self, include_kinds=(LabelKind.CERTIFIED,), acknowledge_contamination=False) -> dict: ...
    def circularity_report(self) -> str: ...
```

---

## 6. Financial & Liquidation Engine (`cardcenter.execution`)

### `cardcenter.execution.almgren_chriss_schedule`
```python
def almgren_chriss_schedule(
    total_units: int,
    horizon_days: float,
    intervals: int,
    sigma_daily: float,
    gamma: float,
    eta: float,
    risk_aversion: float,
) -> ExecutionSchedule:
    """Compute optimal liquidation trajectory minimising cost + lambda * variance."""
```
