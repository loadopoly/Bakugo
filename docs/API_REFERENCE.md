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
    GradeOutcomeModel,
    LearningStore,
    ingest_certified_labels,
    maybe_load_grade_model,
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
* `channel: Optional[ChannelConditions]`: Pixel-space lossy-channel observables on the worst-axis low side (contrast, noise, PSF, rows). Attached by `measure_centering`; failure here never sinks a measurement that already cleared the confidence gate.

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
    def add_scan_from_measure(self, payload: dict, source: str = "measure") -> int: ...
    def get_scan(self, scan_id: int) -> Optional[dict]: ...
    def unsynced_scans(self) -> list[dict]: ...
    def labels_for_scan(self, scan_id: int) -> list[dict]: ...
    def mark_synced(self, scan_id: int, when: Optional[float] = None) -> None: ...
    def add_label(self, scan_id: int, grader: str, grade: str, kind: LabelKind, cert_number: Optional[str] = None) -> int: ...
    def export_training_set(self, include_kinds=(LabelKind.CERTIFIED,), acknowledge_contamination=False) -> dict: ...
    def circularity_report(self) -> str: ...
```

### `cardcenter.cloud`
Best-effort PostgREST mirror to the Loadopoly-OCR Supabase project (`bakugo_scans` / `bakugo_labels`). Uses the **anon** key only (`CARDCENTER_SUPABASE_*`, then `VITE_SUPABASE_*`, then `SUPABASE_URL` + `SUPABASE_ANON_KEY`). Service-role keys are ignored. Certified labels without a cert number are refused before the wire.

```python
def resolve_config(url: Optional[str] = None, anon_key: Optional[str] = None) -> Optional[CloudConfig]: ...
def upsert_scan(scan: Mapping[str, Any], *, device_id: str, config: Optional[CloudConfig] = None) -> CloudResult: ...
def sync_scan_id(store: ScanStore, scan_id: int) -> CloudResult: ...
def sync_store(store: ScanStore, *, include_labels: bool = False) -> CloudResult: ...
```

---

## 5b. Grade Outcome Learning (`cardcenter.learning`)

### `cardcenter.learning.GradeOutcomeModel`
Dirichlet-multinomial `P(issued grade | ratio band, quality, grader)`. Only `CERTIFIED` rows with a `cert_number` update counts. Zero observations leave `predict_overall_grade` identical to the published-table heuristic.

```python
class GradeOutcomeModel:
    def observe(self, example: Mapping[str, Any]) -> bool: ...
    def observe_many(self, examples: Iterable[Mapping[str, Any]]) -> int: ...
    def observations(self, grader: Optional[str] = None) -> int: ...
    def bin_counts(self, grader: str, ratio: float, inner_confidence: float = 0.9) -> dict[str, int]: ...
    def posterior(self, grader: str, ratio: float, inner_confidence: float = 0.9, heuristic: Optional[Mapping[str, float]] = None) -> dict[str, float]: ...
    def posterior_variance(self, grader: str, ratio: float, inner_confidence: float = 0.9) -> Optional[float]: ...

def ingest_certified_labels(scan_store: ScanStore, learning_store: LearningStore) -> GradeOutcomeModel: ...
def maybe_load_grade_model(path: Optional[str] = None) -> GradeOutcomeModel: ...
```

`ingest_certified_labels` **rebuilds** from `export_training_set()` so a second import cannot double-count. `maybe_load_grade_model` reads `path` or `CARDCENTER_DB` and returns an empty identity model when the file is missing.

### `cardcenter.grading.predict_overall_grade`
```python
def predict_overall_grade(
    ratio: Measured,
    quality: Optional[Any] = None,
    geometry: Optional[Any] = None,
    grader: str = "PSA",
    face: Face = "front",
    model: Optional[GradeOutcomeModel] = None,
) -> CardGradePrediction: ...
```
When `model` has certified counts in the matching ratio band, the overall score and probabilities blend toward issued grades. The published centering ceiling is never raised. Corners / edges / surface stay heuristic.

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
