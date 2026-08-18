# Changelog

All notable changes to **Bakugo** (`cardcenter`) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-08-18

### Added
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
