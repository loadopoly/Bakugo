# Changelog

All notable changes to **Bakugo** (`cardcenter`) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [2.3.0] - 2026-08-20

### Added
- Live fusion now consumes the QUIPU overlay: `measure_centering` attaches a `CenteringResult.channel`, `LiveSession.observe` stores it, and `worst_ratio` / `settled` / `status` inflate the PDG bar when `rhythm_boost < 1`. Boost above 1 remains a status signal and never tightens the live interval below the inverse-variance combination.
- **QUIPU Observer client** (`cardcenter.quipu_client`): optional `CARDCENTER_QUIPU_URL` feed of structured observations and cached numeric priors. Collector-number OCR may break catalog ties only when a prior dominates by `QUIPU_TIE_DOMINANCE`; otherwise the reading stays ambiguous. `GET /quipu` reports whether the Observer is linked.
- Docker image (`Dockerfile`, `docker-compose.yml`) serving `cardcenter --serve` on `:8765` with optional Observer URL.

### Changed
- `modulate` no longer pins boost at 1.0 when recursive-strengthening potential is zero, so a washed channel can refuse naive `1/sqrt(N)`.

## [2.2.0] - 2026-08-20

### Changed
- Public web app is **Bakugo** at `loadopoly.com/bakugo` (was `/cardcenter`). The Python package and wheel stay `cardcenter`. Old `/cardcenter/` URL redirects.
- Photos stay on-device. Measurement metadata may sync to the Loadopoly-OCR Supabase project when configured.

### Added
- Live AR camera view in the Pages app: rear camera + `ARSession` guidance for glare, standoff, crop, and square-up, so a messy counter is usable without a clean still.
- **Local + cloud persist**: every successful measure writes `ScanStore` (`CARDCENTER_DB` / browser sqlite). Optional metadata upsert to `bakugo_scans` / `bakugo_labels` in the same Supabase project as Loadopoly-OCR (`cardcenter.cloud`, CLI `--sync-cloud`). Anon key only; never a service-role key. Photos are not uploaded.
- **QUIPU temporal-spatial overlay on the pixel-space lossy channel** (`cardcenter.information`): coherence, relational wash, 7-D Weyl centroid, and boost in `[0.5, 1.5]`. Boost scales effective independent-row count for multi-frame fusion only; the single-shot Cramér–Rao bound is unchanged. `audit_measurement` reports the rhythm and a wash-adjusted fusion floor. `LiveSession.rhythm_boost` consumes multi-frame χ²/dof so disagreeing frames refuse naive `1/√N` credit.

### Security
- Cloud writes reuse the contamination firewall: certified labels without a cert number never leave the device. Pages config is `config.json` (gitignored) or `window.__BAKUGO_SUPABASE__`; the committed file is `config.example.json`.

---

## [2.1.0] - 2026-08-19

### Added
- **Expandable grade-outcome model (`cardcenter.learning.GradeOutcomeModel`)**:
  Dirichlet-multinomial posterior over issued grades, stratified by grader, centering-ratio band, and detection quality. Identity reduction: with zero certified observations, `predict_overall_grade` is identical to the published-table heuristic.
- **Certified ingest path**: `ingest_certified_labels` rebuilds the model from `ScanStore.export_training_set()` (certified + cert number only). `ConnectionManager.import_payload` / `sync` persist the rebuilt model so predictions expand as labels arrive. Re-import does not double-count.
- **CLI `--ingest-grades --db`**: rebuild the model from an existing store. Measure and JSON output now load `CARDCENTER_DB` / `--db` and report `used_learned` / `n_observations`.
- Public exports: `GradeOutcomeModel`, `LearningStore`, `ingest_certified_labels`, `maybe_load_grade_model`.

### Changed
- `predict_overall_grade` / `predict_all_grades` accept an optional `model`. Certified mass blends with the heuristic prior (`GRADE_PRIOR_STRENGTH = 8`); the published centering ceiling is never raised. Subgrades stay heuristic.
- Serve / Perceptopoly payloads include `used_learned` and `n_observations` when a persisted model is present.

### Security
- Circularity firewall unchanged: `observe` refuses `MODEL_PREDICTED`, `MARKETPLACE_VOTE`, `SELF_REPORTED`, and certified-without-cert. Training export remains certified-only.

---

## [2.0.0] - 2026-08-18

### Added
- **Comprehensive Card Grading & Subgrade Predictions (`cardcenter.grading`)**:
  - `CardGradePrediction` dataclass providing predicted overall grades (e.g. `PSA 10 Gem Mint`, `BGS 9.5 Gem Mint`), condition tier classification, and 4-subgrade breakdown (Centering, Corners, Edges, Surface).
  - `predict_overall_grade` & `predict_all_grades` computing grade probability distributions and confidence scores across PSA, BGS, CGC, SGC, and ACE.
  - CLI and Web/AR payloads displaying prominent predicted overall grades alongside centering ceilings.

- **Semantic Versioning Engine (`cardcenter.versioning`)**:
  - `SemVer` class implementing full SemVer 2.0.0 parsing, comparison, and compatibility checks.
  - `VersionInfo` reporting detailed environment, platform architecture, and dependency versions.
  - `EngineCapabilities` dynamic feature matrix tracking metrology, AR, OCR, and sync capabilities.
  - `check_for_updates` querying GitHub releases/tags directly from [`https://github.com/PoodlesOfWar/Bakugo`](https://github.com/PoodlesOfWar/Bakugo).
  - Database schema versioning and forward migration engine (`migrate_database`) supporting `cardcenter/1` to `cardcenter/2`.

- **Remote Connection & Synchronization (`cardcenter.connection`)**:
  - `ConnectionManager` and `ConnectionSpec` for managing remote hubs.
  - `SyncPayload` and `SyncResult` supporting distributed bidirectional sync between mobile devices (Termux) and central database vaults.
  - Deterministic SHA-256 integrity checksums for scan records and wire payloads.
  - **Contamination Firewall**: Enforces strict provenance by quarantining incoming certified labels that lack verified physical slab certification numbers.
  - Endpoint health diagnostics measuring round-trip latency and schema compatibility.

- **CLI Enhancements (`cardcenter.cli`)**:
  - `--info`: Outputs detailed build, runtime, and schema versioning info.
  - `--capabilities`: Displays active engine capabilities.
  - `--check-updates`: Probes upstream GitHub repository for new releases.
  - `--check-health <URL>`: Checks remote endpoint health and latency.
  - `--sync-url <URL> --db <PATH>`: Synchronizes local SQLite store with a remote hub.
  - `--migrate-db <PATH>`: Migrates SQLite database schemas to `cardcenter/2`.

- **Comprehensive Documentation Suite (`docs/`)**:
  - `docs/ARCHITECTURE.md`: Pipeline design, optical ray-tracing, and contamination firewall.
  - `docs/GETTING_STARTED.md`: Setup guide for Android (Termux), Linux, macOS, and Windows.
  - `docs/CLI_GUIDE.md`: Full CLI reference and examples.
  - `docs/VERSIONING_AND_CONNECTION.md`: Detailed specification for versioning, migrations, and sync protocols.
  - `docs/API_REFERENCE.md`: Complete Python API reference.
  - `docs/MATHEMATICAL_FOUNDATIONS.md`: First-principles derivations for optical stack refraction, sensor fusion, and Almgren-Chriss liquidation.

- **Automated Test Suites**:
  - `tests/test_versioning.py`: Testing SemVer parsing, comparisons, capability matrix, schema migrations, and update checks.
  - `tests/test_connection.py`: Testing sync protocols, payload serialization, and contamination firewall quarantining.

### Changed
- Refactored `cardcenter/channel.py` `select_by_consensus` to integrate 3D path continuity into consensus selection.
- Enhanced `ScanStore` with `_schema_metadata` table and automated schema version tagging.
- Modernized `README.md` with status badges, upstream repository links, and a complete documentation index.

---

## [1.9.0] - Prior Release
- Initial release with optical stack refraction, multi-card detection, AR capture, and Almgren-Chriss liquidation.
