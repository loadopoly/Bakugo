"""Offline tests for the Loadopoly-OCR Supabase mirror.

No live network. Credentials, firewall, and PostgREST headers only.
"""

from __future__ import annotations

import io
from typing import Any

import pytest

from cardcenter.cloud import (
    CloudConfig,
    firewall_label,
    label_to_row,
    resolve_config,
    scan_to_row,
    upsert_scan,
)
from cardcenter.store import LabelKind


def test_resolve_config_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "CARDCENTER_SUPABASE_URL",
        "CARDCENTER_SUPABASE_ANON_KEY",
        "VITE_SUPABASE_URL",
        "VITE_SUPABASE_ANON_KEY",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    assert resolve_config() is None


def test_resolve_config_prefers_cardcenter_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CARDCENTER_SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("CARDCENTER_SUPABASE_ANON_KEY", "anon-test-key")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-must-never-be-used")
    cfg = resolve_config()
    assert cfg is not None
    assert cfg.url == "https://example.supabase.co"
    assert cfg.anon_key == "anon-test-key"
    assert "service" not in cfg.anon_key


def test_firewall_rejects_certified_without_cert() -> None:
    with pytest.raises(ValueError, match="cert number"):
        firewall_label(LabelKind.CERTIFIED.value, None)
    with pytest.raises(ValueError, match="cert number"):
        firewall_label("certified", "   ")


def test_firewall_allows_certified_with_cert_and_opinions() -> None:
    firewall_label("certified", "99887766")
    firewall_label(LabelKind.MODEL_PREDICTED.value, None)
    firewall_label(LabelKind.SELF_REPORTED.value, "")


def test_scan_to_row_mapping() -> None:
    row = scan_to_row(
        {
            "id": 7,
            "card_key": "sv01-mew",
            "holder": "raw",
            "worst_ratio_pct": 55.0,
            "worst_ratio_sigma": 0.4,
            "worst_axis": "horizontal",
            "h_ratio_pct": 55.0,
            "v_ratio_pct": 50.0,
            "left_mm": 2.7,
            "right_mm": 3.3,
            "top_mm": 3.0,
            "bottom_mm": 3.0,
            "px_per_mm": 12.0,
            "inner_confidence": 0.91,
            "refraction_applied": 1,
            "warnings": "glare",
            "phash": 42,
            "source": "cli",
        },
        device_id="device-1",
        engine_version="2.2.0",
    )
    assert row["local_id"] == 7
    assert row["device_id"] == "device-1"
    assert row["card_key"] == "sv01-mew"
    assert row["refraction_applied"] is True
    assert row["engine_version"] == "2.2.0"


def test_label_to_row_refuses_poison() -> None:
    with pytest.raises(ValueError, match="cert number"):
        label_to_row(
            {"id": 1, "grader": "PSA", "grade": "10", "kind": "certified"},
            device_id="d1",
        )


def test_upsert_scan_skipped_without_config(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "CARDCENTER_SUPABASE_URL",
        "CARDCENTER_SUPABASE_ANON_KEY",
        "VITE_SUPABASE_URL",
        "VITE_SUPABASE_ANON_KEY",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    result = upsert_scan({"id": 1, "worst_ratio_pct": 50.0}, device_id="d1")
    assert result.ok and result.skipped


def test_upsert_scan_posts_merge_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Resp(io.BytesIO):
        status = 201

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def fake_urlopen(req: Any, timeout: float = 0) -> _Resp:
        captured["url"] = req.full_url
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["timeout"] = timeout
        captured["body"] = req.data
        return _Resp(b"")

    monkeypatch.setattr("cardcenter.cloud.urllib.request.urlopen", fake_urlopen)
    cfg = CloudConfig(url="https://example.supabase.co", anon_key="anon-test-key")
    result = upsert_scan(
        {"id": 3, "card_key": "x", "holder": "raw", "worst_ratio_pct": 52.0},
        device_id="dev-9",
        config=cfg,
    )
    assert result.ok and result.status == 201
    assert "bakugo_scans" in captured["url"]
    assert "on_conflict=device_id,local_id" in captured["url"]
    assert captured["headers"]["authorization"] == "Bearer anon-test-key"
    assert captured["headers"]["apikey"] == "anon-test-key"
    assert "merge-duplicates" in captured["headers"]["prefer"]
    assert b"service" not in captured["body"]
