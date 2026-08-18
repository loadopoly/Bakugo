"""Tests for Remote Connection, Sync Protocol, and Provenance Firewall."""

import json
import time
import pytest

from cardcenter import BorderPair, CenteringResult, DetectionQuality, Measured, SlabSpec, __version__
from cardcenter.connection import (
    ConnectionManager,
    ConnectionSpec,
    EndpointHealth,
    EndpointStatus,
    SyncPayload,
    SyncResult,
    compute_scan_hash,
)
from cardcenter.store import LabelKind, ScanStore


def _dummy_result() -> CenteringResult:
    import numpy as np
    return CenteringResult(
        horizontal=BorderPair("horizontal", "left", "right", Measured(3.0, 0.05), Measured(3.0, 0.05)),
        vertical=BorderPair("vertical", "top", "bottom", Measured(3.0, 0.05), Measured(3.0, 0.05)),
        quality=DetectionQuality(0.5, 0.9, {"left": 0.9, "right": 0.9, "top": 0.9, "bottom": 0.9}, [], False, 0.0),
        px_per_mm=10.0,
        corners_px=np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 140.0], [0.0, 140.0]]),
        inner_rect_mm=(3.0, 3.0, 60.5, 85.0),
        slab=SlabSpec("raw", 0.0, 0.0, 1.0),
    )



def test_connection_spec_normalization() -> None:
    spec1 = ConnectionSpec("192.168.1.100:8765")
    assert spec1.normalized_url() == "http://192.168.1.100:8765"

    spec2 = ConnectionSpec("https://hub.example.com/api/")
    assert spec2.normalized_url() == "https://hub.example.com/api"


def test_compute_scan_hash_consistency() -> None:
    scan = {
        "card_key": "sv03-charizard",
        "holder": "bgs",
        "worst_ratio_pct": 52.5,
        "worst_axis": "horizontal",
        "h_ratio_pct": 52.5,
        "v_ratio_pct": 50.0,
        "left_mm": 2.9,
        "right_mm": 3.1,
        "top_mm": 3.0,
        "bottom_mm": 3.0,
        "px_per_mm": 12.4,
        "phash": 12345678,
    }
    h1 = compute_scan_hash(scan)
    h2 = compute_scan_hash(scan)
    assert h1 == h2
    assert len(h1) == 16


def test_sync_payload_serialization() -> None:
    payload = SyncPayload(
        schema_version="cardcenter/2",
        engine_version=__version__,
        exported_at=time.time(),
        client_id="counter-tablet-1",
        scans=[{"card_key": "test1", "worst_ratio_pct": 50.0}],
        labels=[{"grader": "PSA", "grade": "10", "kind": "certified", "cert_number": "12345678"}],
    )
    d = payload.to_dict()
    assert d["client_id"] == "counter-tablet-1"
    assert "integrity_checksum" in d
    assert len(d["integrity_checksum"]) > 0

    reconstructed = SyncPayload.from_dict(d)
    assert reconstructed.client_id == "counter-tablet-1"
    assert len(reconstructed.scans) == 1
    assert len(reconstructed.labels) == 1


def test_provenance_firewall_quarantine(tmp_path) -> None:
    db_path = str(tmp_path / "hub.db")
    mgr = ConnectionManager()

    with ScanStore(db_path) as store:
        # Create a payload with:
        # 1. Valid certified label (with cert number)
        # 2. Contaminated certified label (missing cert number)
        # 3. Self reported label
        # 4. Marketplace vote label
        payload = SyncPayload(
            schema_version="cardcenter/2",
            engine_version=__version__,
            exported_at=time.time(),
            client_id="peer-scanner",
            scans=[
                {
                    "id": 1,
                    "card_key": "sv01-mew",
                    "holder": "raw",
                    "worst_ratio_pct": 51.0,
                    "worst_ratio_sigma": 0.5,
                    "worst_axis": "horizontal",
                    "h_ratio_pct": 51.0,
                    "v_ratio_pct": 50.0,
                    "px_per_mm": 10.0,
                    "inner_confidence": 0.95,
                    "refraction_applied": 0,
                    "warnings": "",
                    "phash": 42,
                    "created_at": time.time(),
                }
            ],
            labels=[
                # Valid certified
                {
                    "scan_id": 1,
                    "grader": "PSA",
                    "grade": "10",
                    "kind": LabelKind.CERTIFIED.value,
                    "cert_number": "99887766",
                },
                # Contaminated: claims certified but NO cert number
                {
                    "scan_id": 1,
                    "grader": "BGS",
                    "grade": "9.5",
                    "kind": LabelKind.CERTIFIED.value,
                    "cert_number": None,
                },
                # Self-reported
                {
                    "scan_id": 1,
                    "grader": "CGC",
                    "grade": "9",
                    "kind": LabelKind.SELF_REPORTED.value,
                    "cert_number": None,
                },
            ],
        )

        scans_imp, labels_imp, quar = mgr.import_payload(store, payload, strict_provenance=True)
        assert scans_imp == 1
        assert labels_imp == 2  # 1 valid certified + 1 self-reported
        assert quar == 1  # 1 contaminated certified quarantined!

        # Check what is in store
        counts = store.label_counts()
        assert counts.get(LabelKind.CERTIFIED.value) == 1
        assert counts.get(LabelKind.SELF_REPORTED.value) == 1

        # Check export training set is pure
        training = store.export_training_set()
        assert training["manifest"]["ground_truth_only"] is True
        assert len(training["examples"]) == 1
        assert training["examples"][0]["cert_number"] == "99887766"


def test_export_import_roundtrip(tmp_path) -> None:
    src_db = str(tmp_path / "src.db")
    dst_db = str(tmp_path / "dst.db")
    mgr = ConnectionManager()

    with ScanStore(src_db) as src_store:
        scan_id = src_store.add_scan("pikachu-001", _dummy_result(), phash=999)
        src_store.add_label(scan_id, "PSA", "10", LabelKind.CERTIFIED, cert_number="11223344")

        payload = mgr.export_store_payload(src_store, client_id="unit-test-device")
        assert len(payload.scans) == 1
        assert len(payload.labels) == 1

    with ScanStore(dst_db) as dst_store:
        scans_imp, labels_imp, quar = mgr.import_payload(dst_store, payload)
        assert scans_imp == 1
        assert labels_imp == 1
        assert quar == 0
        assert dst_store.scan_count() == 1
        assert dst_store.label_counts()[LabelKind.CERTIFIED.value] == 1


def test_endpoint_health_object() -> None:
    h = EndpointHealth(
        status=EndpointStatus.ONLINE,
        latency_ms=14.5,
        server_version="1.9.0",
        schema_version="cardcenter/2",
        scan_count=120,
        certified_label_count=35,
    )
    assert h.is_healthy()
    assert h.latency_ms == 14.5
