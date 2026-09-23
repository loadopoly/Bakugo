"""The public build refuses collection-derived or untagged artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cardcenter import release_guard as rg

PKG = Path(rg.__file__).resolve().parent


def _artifact(root: Path, kind="quad_detect", exposure="generic", payload=None, **lineage):
    d = root / "data" / "released" / kind
    d.mkdir(parents=True)
    body = json.dumps(payload if payload is not None else {"min_area_frac": 0.01})
    (d / "params.json").write_text(body)
    meta = {
        "schema": rg.SCHEMA, "kind": kind, "task": kind, "exposure": exposure,
        "payload": "params.json", "payload_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "lineage": {"train_set_digest": "d" * 64, "n_train": 3, "confirmed_only": True, **lineage},
    }
    (d / "artifact.json").write_text(json.dumps(meta))
    return d


def test_working_tree_package_passes():
    assert rg.scan(PKG) == []


def test_generic_artifact_passes_and_loads(tmp_path):
    _artifact(tmp_path)
    assert rg.scan(tmp_path) == []
    assert rg.released_params("quad_detect", tmp_path) == {"min_area_frac": 0.01}


@pytest.mark.parametrize("kind", ["card_index", "priced_priors", "sticker_dataset", "number_priors"])
def test_collection_kinds_refused_even_if_declared_generic(tmp_path, kind):
    _artifact(tmp_path, kind=kind)
    assert rg.scan(tmp_path)


def test_collection_exposure_refused(tmp_path):
    _artifact(tmp_path, exposure="collection")
    assert rg.scan(tmp_path)


def test_nested_payload_refused(tmp_path):
    _artifact(tmp_path, payload={"cards": [{"name": "Pikachu", "price": 3}]})
    assert rg.scan(tmp_path)


def test_tampered_payload_refused_and_not_loaded(tmp_path):
    d = _artifact(tmp_path)
    (d / "params.json").write_text(json.dumps({"min_area_frac": 0.5}))
    assert rg.scan(tmp_path)
    assert rg.released_params("quad_detect", tmp_path) is None


def test_missing_lineage_refused(tmp_path):
    _artifact(tmp_path, confirmed_only=False)
    assert rg.scan(tmp_path)


def test_untagged_file_and_private_tier_files_refused(tmp_path):
    (tmp_path / "data" / "released").mkdir(parents=True)
    (tmp_path / "data" / "released" / "prices.json").write_text("{}")
    (tmp_path / "manifest.duckdb").write_text("")
    problems = rg.scan(tmp_path)
    assert len(problems) == 2


def test_stray_collection_artifact_anywhere_refused(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "artifact.json").write_text(json.dumps({"kind": "card_index", "exposure": "collection"}))
    assert rg.scan(tmp_path)


def test_main_exit_codes(tmp_path):
    assert rg.main([str(tmp_path)]) == 0
    _artifact(tmp_path, exposure="collection")
    assert rg.main([str(tmp_path)]) == 1


def test_trainer_release_roundtrip(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    monkeypatch.setenv("BAKUGO_VAULT_DB", str(tmp_path / "vault.duckdb"))
    from trainer import artifacts
    from trainer.paths import resolve_layout

    layout = resolve_layout(tmp_path / "private").ensure()
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    art = artifacts.write_candidate(layout, task="quad_detect", exposure="generic",
                                    params={"min_area_frac": 0.012}, train_shas=["a" * 64])
    artifacts.set_champion(layout, art, {"mean_diff": 0.1, "ci": [0.05, 0.15], "n_test": 40})
    target = artifacts.release(layout, "quad_detect", package_root=pkg)
    meta = json.loads((target / "artifact.json").read_text())
    assert "train_sha256" not in meta["lineage"]          # private hashes stay private
    assert rg.released_params("quad_detect", pkg) == {"min_area_frac": 0.012}

    coll = artifacts.write_candidate(layout, task="back_hue", exposure="collection",
                                     params={"yugioh_hue": 20.0}, train_shas=["a" * 64])
    artifacts.set_champion(layout, coll, {})
    with pytest.raises(rg.ReleaseRefused):
        artifacts.release(layout, "back_hue", package_root=pkg)
    assert not (pkg / "data" / "released" / "back_hue").exists()
