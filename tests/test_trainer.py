"""Invariants of the private trainer (trainer/). Each test pins one rule from
the architecture: private root placement, read-only Drive scope, label
discipline, grouped and frozen splits, promotion statistics, QUIPU
proposal-only, and runtime isolation."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")
cv2 = pytest.importorskip("cv2")

from trainer import (artifacts, isolation, loop, manifest, promote, pseudo,  # noqa: E402
                     quipu_outbox, split, sync)
from trainer.data import frozen_test_items, training_items  # noqa: E402
from trainer.paths import REPO_ROOT, UnsafeLocation, check_secret_file, resolve_layout  # noqa: E402
from trainer.tasks import ItemScore  # noqa: E402


# ----------------------------------------------------------------- fixtures

def _noise(seed, shape=(60, 80)):
    rng = np.random.default_rng(seed)
    img = (rng.random((*shape, 3)) * 255).astype("uint8")
    return cv2.resize(img, (shape[1] * 5, shape[0] * 5), interpolation=cv2.INTER_NEAREST)


@pytest.fixture(autouse=True)
def _private_vault(tmp_path, monkeypatch):
    """Never let a test touch the real supabase_vault.duckdb."""
    monkeypatch.setenv("BAKUGO_VAULT_DB", str(tmp_path / "vault" / "supabase_vault.duckdb"))


@pytest.fixture()
def layout(tmp_path):
    return resolve_layout(tmp_path / "private").ensure()


@pytest.fixture()
def con(layout):
    c = manifest.connect(layout)
    yield c
    c.close()


def _write(layout, rel, img):
    p = layout.mirror / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)
    return p


def _sha_of(con, rel):
    return con.execute("SELECT sha256 FROM file_paths WHERE rel_path = ?", [rel]).fetchone()[0]


# ------------------------------------------------------------ private root

def test_private_root_inside_repo_is_refused():
    with pytest.raises(UnsafeLocation):
        resolve_layout(REPO_ROOT / "private_data")


def test_private_root_default_is_outside_repo(monkeypatch):
    monkeypatch.delenv("BAKUGO_PRIVATE_ROOT", raising=False)
    layout = resolve_layout()
    assert REPO_ROOT.resolve() not in layout.root.resolve().parents


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits")
def test_token_file_must_not_be_group_readable(tmp_path, monkeypatch):
    monkeypatch.delenv("BAKUGO_SECRET_PERMS", raising=False)
    f = tmp_path / "rclone.conf"
    f.write_text("[x]\n")
    f.chmod(0o644)
    with pytest.raises(UnsafeLocation):
        check_secret_file(f, "RCLONE_CONFIG")
    f.chmod(0o600)
    assert check_secret_file(f, "RCLONE_CONFIG") == f.resolve()


# ------------------------------------------------------------------- sync

def _conf(tmp_path, scope):
    f = tmp_path / "rclone.conf"
    body = "[bakugo-drive]\ntype = drive\n"
    if scope:
        body += f"scope = {scope}\n"
    body += 'token = {"access_token":"x"}\n'
    f.write_text(body)
    f.chmod(0o600)
    return f


@pytest.mark.parametrize("scope", [None, "drive", "drive.file"])
def test_sync_refuses_non_readonly_scope(tmp_path, scope):
    with pytest.raises(sync.ScopeError):
        sync.check_remote_scope(_conf(tmp_path, scope), "bakugo-drive")


def test_sync_uses_copy_not_sync_and_records_listing(tmp_path, layout, monkeypatch):
    monkeypatch.delenv("BAKUGO_SKIP_SCOPE_CHECK", raising=False)
    cfg = sync.SyncConfig("bakugo-drive", "Cards", _conf(tmp_path, "drive.readonly"))
    calls = []

    class Out:
        stdout = json.dumps([{"Path": "Pokemon/a.jpg", "ID": "drive-id-1"}])

    def runner(cmd, **kw):
        calls.append(cmd)
        return Out()

    result = sync.run_sync(cfg, layout, runner=runner)
    assert calls[0][1] == "copy" and "sync" not in calls[0][1:2]
    assert all("--drive-scope=drive.readonly" in c for c in calls)
    assert result["listed"] == 1
    assert sync.load_listing(layout)["Pokemon/a.jpg"]["ID"] == "drive-id-1"


# --------------------------------------------------------------- manifest

def test_manifest_diff_weak_labels_and_pdf_pages(layout, con):
    _write(layout, "Pokemon/Duplicates/Lechonk_182.png", _noise(1))
    _write(layout, "Pokemon/Binder/PXL_20260914_000000001.jpg", _noise(2))
    from PIL import Image

    (layout.mirror / "Yugioh").mkdir(parents=True)
    Image.fromarray(_noise(3)).save(layout.mirror / "Yugioh" / "scan.pdf")
    first = manifest.refresh(layout, con)
    assert len(first["new"]) == 4          # 2 images + pdf + 1 page
    assert manifest.refresh(layout, con)["new"] == []

    row = con.execute("SELECT label, label_number, label_strength, source_type, franchise "
                      "FROM files WHERE rel_path LIKE '%Lechonk%'").fetchone()
    assert row == ("Lechonk", 182, "weak", "labelled_duplicate", "pokemon")
    cam = con.execute("SELECT label, capture_date::VARCHAR FROM files WHERE rel_path LIKE '%PXL%'").fetchone()
    assert cam == (None, "2026-09-14")
    page = con.execute("SELECT franchise, parent_sha256 IS NOT NULL FROM files WHERE kind = 'pdf_page'").fetchone()
    assert page == ("yugioh", True)

    (layout.mirror / "Pokemon/Binder/PXL_20260914_000000001.jpg").unlink()
    gone = manifest.refresh(layout, con)
    assert len(gone["missing"]) == 1


def test_label_strength_rejects_pseudo(con, layout):
    _write(layout, "a/x.png", _noise(1))
    manifest.refresh(layout, con)
    with pytest.raises(duckdb.ConstraintException):
        con.execute("UPDATE files SET label_strength = 'pseudo'")


@pytest.mark.parametrize("who", ["", "  ", "trainer", "Model"])
def test_confirm_requires_a_human(con, layout, who):
    _write(layout, "a/x.png", _noise(1))
    manifest.refresh(layout, con)
    sha = _sha_of(con, "a/x.png")
    with pytest.raises(ValueError):
        manifest.confirm(con, sha, confirmed_by=who, label="Pikachu")


class _FixedPredictor:
    model_id = "fixed-1"

    def __init__(self, label, conf=1.0):
        self.label, self.conf = label, conf

    def predict(self, path):
        return pseudo.Pseudo(self.label, 25, self.conf, True, False)


def test_pseudo_labels_never_become_training_or_test_labels(con, layout):
    for i in range(4):
        _write(layout, f"Pokemon/x{i}/PXL_20260101_{i}.png", _noise(10 + i))
    manifest.refresh(layout, con)
    split.assign(con)
    assert pseudo.pseudo_label(con, layout, [_FixedPredictor("Pikachu")]) == 4
    assert con.execute("SELECT count(*) FROM files WHERE label_strength = 'confirmed'").fetchone()[0] == 0
    assert training_items(con, layout) == []
    assert frozen_test_items(con, layout, model_id="m", task="t") == []


def test_queue_flags_disagreement_and_all_test_items(con, layout):
    _write(layout, "Pokemon/d/Lechonk_1.png", _noise(1))
    manifest.refresh(layout, con)
    con.execute("UPDATE files SET split = 'train'")
    pseudo.pseudo_label(con, layout, [_FixedPredictor("Pikachu")])
    q = pseudo.build_queue(con, layout)
    assert any("disagrees with filename label" in r for r in q[0]["reasons"])
    con.execute("UPDATE files SET split = 'test'")
    con.execute("DELETE FROM pseudo_labels")
    pseudo.pseudo_label(con, layout, [_FixedPredictor("Lechonk")])
    q = pseudo.build_queue(con, layout)
    assert q and any("test split" in r for r in q[0]["reasons"])


# ------------------------------------------------------------------ split

def test_near_duplicates_and_same_card_share_a_split(con, layout):
    base = _noise(5)
    _write(layout, "Pokemon/2026-09-01/a.png", base)
    brighter = np.clip(base.astype(int) + 12, 0, 255).astype("uint8")
    _write(layout, "Pokemon/2026-09-09 sleeved/b.png", brighter)       # near-copy
    _write(layout, "Pokemon/other/c.png", _noise(6))
    _write(layout, "Pokemon/other2/d.png", _noise(7))
    manifest.refresh(layout, con)
    shas = {r: _sha_of(con, r) for r in ("Pokemon/2026-09-01/a.png", "Pokemon/2026-09-09 sleeved/b.png",
                                          "Pokemon/other/c.png", "Pokemon/other2/d.png")}
    manifest.confirm(con, shas["Pokemon/other/c.png"], confirmed_by="adam", card_uid="card-7")
    manifest.confirm(con, shas["Pokemon/other2/d.png"], confirmed_by="adam", card_uid="card-7")
    split.assign(con, split.SplitConfig(test_fraction=0.5))
    s = dict(con.execute("SELECT sha256, split FROM files").fetchall())
    assert s[shas["Pokemon/2026-09-01/a.png"]] == s[shas["Pokemon/2026-09-09 sleeved/b.png"]]
    assert s[shas["Pokemon/other/c.png"]] == s[shas["Pokemon/other2/d.png"]]
    assert split.leakage_report(con) == []


def test_frozen_test_never_moves_and_linked_train_is_quarantined(con, layout):
    _write(layout, "p/t.png", _noise(21))
    _write(layout, "p/r.png", _noise(22))
    manifest.refresh(layout, con)
    t, r = _sha_of(con, "p/t.png"), _sha_of(con, "p/r.png")
    con.execute("UPDATE files SET split = 'test' WHERE sha256 = ?", [t])
    con.execute("INSERT INTO frozen_test VALUES (?, now())", [t])
    con.execute("UPDATE files SET split = 'train' WHERE sha256 = ?", [r])
    # a later confirmation says both photos are the same physical card
    manifest.confirm(con, t, confirmed_by="adam", card_uid="card-1")
    manifest.confirm(con, r, confirmed_by="adam", card_uid="card-1")
    stats = split.assign(con)
    s = dict(con.execute("SELECT sha256, split FROM files").fetchall())
    assert s[t] == "test"
    assert s[r] == "quarantine"
    assert stats["quarantined"] == 1
    assert training_items(con, layout) == []
    assert split.leakage_report(con) == []


def test_split_assignment_is_stable(con, layout):
    for i in range(20):
        _write(layout, f"p/{i}.png", _noise(100 + i))
    manifest.refresh(layout, con)
    split.assign(con)
    before = dict(con.execute("SELECT sha256, split FROM files").fetchall())
    split.assign(con)
    assert dict(con.execute("SELECT sha256, split FROM files").fetchall()) == before
    frozen = {r[0] for r in con.execute("SELECT sha256 FROM frozen_test").fetchall()}
    assert frozen == {k for k, v in before.items() if v == "test"}


# -------------------------------------------------------------- promotion

def _scores(values, slice_of=lambda i: "9-12"):
    return [ItemScore(f"k{i}", slice_of(i), float(v)) for i, v in enumerate(values)]


def test_promotion_needs_gain_outside_error_bars():
    cfg = promote.PromotionConfig(min_items=30, n_boot=2000)
    champ = _scores([0] * 20 + [1] * 20)
    tiny_gain = _scores([1] + [0] * 19 + [1] * 20)
    assert promote.compare(tiny_gain, champ, cfg)["promote"] is False
    big_gain = _scores([1] * 15 + [0] * 5 + [1] * 20)
    r = promote.compare(big_gain, champ, cfg)
    assert r["promote"] is True and r["ci"][0] > 0


def test_promotion_blocked_by_any_slice_regression():
    cfg = promote.PromotionConfig(min_items=30, n_boot=2000)
    slice_of = lambda i: "<6 px/mm" if i >= 38 else "9-12"
    champ = _scores([0] * 20 + [1] * 20, slice_of)
    cand = _scores([1] * 20 + [1] * 18 + [0, 0], slice_of)   # big overall gain, small-slice loss
    r = promote.compare(cand, champ, cfg)
    assert r["ci"][0] > 0
    assert r["promote"] is False and "<6 px/mm" in r["reason"]


def test_promotion_refuses_small_test_set():
    r = promote.compare(_scores([1] * 5), _scores([0] * 5), promote.PromotionConfig())
    assert r["promote"] is False and "paired test items" in r["reason"]


class _ToyTask:
    """A task whose score is 'param >= item threshold', to drive the loop."""
    name = "quad_detect"
    exposure = "generic"

    def __init__(self, fitted):
        self.fitted = fitted

    def default_params(self):
        return {"min_area_frac": 0.0, "information_snap": 1}

    def fit(self, con, layout):
        return {"min_area_frac": self.fitted, "information_snap": 1}, ["a" * 64]

    def score(self, con, layout, params, model_id):
        return [ItemScore(f"i{i}", "9-12", 1.0 if params["min_area_frac"] >= i / 40 else 0.0)
                for i in range(40)]


def test_night_then_dawn_promotes_once_and_logs(layout):
    res = loop.night(layout, tasks=[_ToyTask(1.0)])
    assert res[0]["promote"] is True
    assert artifacts.champion(layout, "quad_detect") is None      # nothing promoted at night
    decisions = loop.dawn(layout)
    assert [d["promoted"] for d in decisions] == [True]
    assert artifacts.champion(layout, "quad_detect").params["min_area_frac"] == 1.0
    assert loop.dawn(layout) == []                                # decided once
    # a worse candidate is evaluated and kept out, with a reason
    loop.night(layout, tasks=[_ToyTask(0.5)])
    d = loop.dawn(layout)
    assert d[0]["promoted"] is False and d[0]["reason"]
    assert artifacts.champion(layout, "quad_detect").params["min_area_frac"] == 1.0
    con = manifest.connect(layout)
    try:
        assert con.execute("SELECT count(*) FROM promotions").fetchone()[0] == 2
    finally:
        con.close()


def test_candidate_artifact_records_lineage(layout):
    art = artifacts.write_candidate(layout, task="quad_detect", exposure="generic",
                                    params={"min_area_frac": 0.01}, train_shas=["b" * 64, "a" * 64])
    lin = art.meta["lineage"]
    assert lin["confirmed_only"] is True and lin["n_train"] == 2
    assert lin["train_sha256"] == ["a" * 64, "b" * 64]


# ------------------------------------------------------------------ QUIPU

def test_quipu_proposals_are_never_realised_and_use_train_only(con, layout):
    _write(layout, "p/a.png", _noise(1))
    _write(layout, "p/b.png", _noise(2))
    manifest.refresh(layout, con)
    a, b = _sha_of(con, "p/a.png"), _sha_of(con, "p/b.png")
    con.execute("UPDATE files SET split = 'train' WHERE sha256 = ?", [a])
    con.execute("UPDATE files SET split = 'test' WHERE sha256 = ?", [b])
    manifest.confirm(con, a, confirmed_by="adam", number=182)
    manifest.confirm(con, b, confirmed_by="adam", number=25)
    path = quipu_outbox.propose_number_priors(con, layout)
    body = json.loads(Path(path).read_text())
    assert body["status"] == "proposed" and body["realised"] is False and body["authorised"] is False
    assert body["payload"]["numeric_lexicon"] == [{"token": "182", "freq": 1}]
    assert quipu_outbox.propose_number_priors(con, layout) == path     # idempotent, no overwrite


@pytest.mark.parametrize("var,val", [("QUIPU_ATTEST_KEY_FILE", "/k"),
                                     ("QUIPU_REALISE_GRANT_REF", "IT505-CR-1"),
                                     ("QUIPU_ACCEPT_SELF_ASSERTED", "1")])
def test_trainer_refuses_to_hold_realisation_credentials(var, val):
    with pytest.raises(isolation.IsolationError):
        isolation.assert_no_realisation_credentials({var: val}, key_paths=[])


def test_trainer_refuses_readable_attest_key(tmp_path):
    key = tmp_path / "attest.key"
    key.write_text("k")
    with pytest.raises(isolation.IsolationError):
        isolation.assert_no_realisation_credentials({}, key_paths=[key])


# -------------------------------------------------------------- isolation

def test_side_channels_disabled(monkeypatch):
    monkeypatch.setenv("CARDCENTER_QUIPU_URL", "http://quipu:7100")
    monkeypatch.setenv("CARDCENTER_SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("CARDCENTER_SUPABASE_ANON_KEY", "anon")
    monkeypatch.delenv("CARDCENTER_QUIPU_DISABLE", raising=False)
    from cardcenter import cloud, quipu_client

    assert quipu_client.enabled()
    isolation.disable_side_channels(os.environ)
    assert not quipu_client.enabled()
    assert cloud.resolve_config() is None


def test_offline_check_fails_when_a_connection_succeeds():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        with pytest.raises(isolation.IsolationError):
            isolation.assert_offline(probes=[srv.getsockname()], timeout=1.0)
    finally:
        srv.close()
    isolation.assert_offline(probes=[("127.0.0.1", 9)], timeout=0.5)  # closed port passes


# ------------------------------------------------ real tasks, real detector

def _card_scene(seed, scale):
    rng = np.random.default_rng(seed)
    h, w = 900, 1200
    # dark, blocky, per-scene background so unrelated scenes hash apart
    bg = cv2.resize(_noise(seed + 50, (3, 4)), (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur((bg * 0.3).astype(np.uint8), (0, 0), 40)
    cx, cy = w / 2 + rng.uniform(-80, 80), h / 2 + rng.uniform(-60, 60)
    half_w, half_h = 63 * scale / 2, 88 * scale / 2
    ang = np.deg2rad(rng.uniform(-8, 8))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    corners = np.array([[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]]) @ R.T
    corners += [cx, cy]
    cv2.fillConvexPoly(img, corners.round().astype(np.int32), (230, 225, 215), lineType=cv2.LINE_AA)
    return img, corners


def test_quad_detect_task_on_real_detector(con, layout):
    from trainer.tasks import QuadDetectTask

    for i in range(6):
        img, corners = _card_scene(i, scale=6.0)
        rel = f"Pokemon/bulk pile/scene{i}.png"
        _write(layout, rel, img)
    manifest.refresh(layout, con)
    stats = split.assign(con, split.SplitConfig(test_fraction=0.5))
    assert stats["components"] == 6          # distinct scenes are not chained together
    # pin a mixed split so both fit and score run on the real detector
    shas = [_sha_of(con, f"Pokemon/bulk pile/scene{i}.png") for i in range(6)]
    con.execute("DELETE FROM frozen_test")
    for i, sha in enumerate(shas):
        con.execute("UPDATE files SET split = ? WHERE sha256 = ?", ["test" if i < 3 else "train", sha])
        if i < 3:
            con.execute("INSERT INTO frozen_test VALUES (?, now())", [sha])
    for i in range(6):
        _, corners = _card_scene(i, scale=6.0)
        manifest.confirm_quad(con, _sha_of(con, f"Pokemon/bulk pile/scene{i}.png"), 0,
                              corners.tolist(), confirmed_by="adam")
    task = QuadDetectTask()
    params, used = task.fit(con, layout)
    assert len(used) == 3
    assert params["min_area_frac"] in task.grid
    scores = task.score(con, layout, task.default_params(), "code-default")
    assert len(scores) == 3
    assert all(s.value == 1.0 for s in scores)         # clean scenes: detector finds every card
    assert con.execute("SELECT count(*) FROM test_uses WHERE task = 'quad_detect'").fetchone()[0] == 1


def test_back_hue_calibration_moves_toward_measured_backs(con, layout):
    from trainer.tasks import BackHueTask

    def back(hue, seed, value=70):
        rng = np.random.default_rng(seed)
        hsv = np.zeros((120, 90, 3), np.uint8)
        hsv[..., 0] = np.clip(hue + rng.normal(0, 1.5, (120, 90)), 0, 179)
        hsv[..., 1] = 170
        hsv[..., 2] = value
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    for i in range(6):
        _write(layout, f"Yugioh/backs/y{i}.png", back(19, i))
    for i in range(4):   # bright brown backs: inside the price-bar mask band
        _write(layout, f"Yugioh/backs/bright{i}.png", back(19, 100 + i, value=150))
    manifest.refresh(layout, con)
    for (sha,) in con.execute("SELECT sha256 FROM files").fetchall():
        manifest.confirm(con, sha, confirmed_by="adam", franchise="yugioh", source_type="binder_back")
    con.execute("UPDATE files SET split = 'train'")
    task = BackHueTask()
    params, used = task.fit(con, layout)
    # bright brown backs are no longer thrown away as "orange price label"
    assert len(used) == 10
    assert "0 unmeasurable" in task.last_fit_note
    assert abs(params["yugioh_hue"] - 19) < 1.5
    assert params["yugioh_n"] == 10
    assert params["pokemon_hue"] == 108.0               # no confirmed Pokemon backs: default kept
    # score: the candidate registry classifies confirmed test backs
    con.execute("UPDATE files SET split = 'test'")
    for (sha,) in con.execute("SELECT sha256 FROM files").fetchall():
        con.execute("INSERT OR IGNORE INTO frozen_test VALUES (?, now())", [sha])
    scores = task.score(con, layout, params, "cand")
    assert len(scores) == 10 and all(x.value == 1.0 for x in scores)


def test_cli_day_night_dawn_status(layout, monkeypatch, capsys):
    from trainer.__main__ import main

    monkeypatch.setenv("BAKUGO_TRAINER_REQUIRE_OFFLINE", "0")
    for k in isolation.REALISATION_VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(isolation, "_default_key_paths", lambda: [])
    _write(layout, "Pokemon/Duplicates/Lechonk_182.png", _noise(3))
    root = ["--root", str(layout.root)]
    assert main(root + ["day", "--no-predict"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["result"]["new"] == 1 and out["result"]["queued"] == 1
    assert out["isolation"]["side_channels_removed"] is not None
    assert os.environ["CARDCENTER_QUIPU_DISABLE"] == "1"
    assert main(root + ["night"]) == 0
    night = json.loads(capsys.readouterr().out)["result"]
    assert all("skipped" in r for r in night)            # nothing confirmed yet
    assert main(root + ["dawn"]) == 0
    capsys.readouterr()
    assert main(root + ["status"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["champions"] == {"quad_detect": "code default", "back_hue": "code default"}



# ------------------------------------------------------------------- vault

def test_trainer_tables_live_in_trainer_schema_of_the_vault(layout, con):
    con.execute("CREATE TABLE IF NOT EXISTS main.bakugo_scans (id INTEGER)")
    tables = set(con.execute(
        "SELECT table_schema || '.' || table_name FROM information_schema.tables").fetchall())
    names = {t[0] for t in tables}
    assert "trainer.files" in names and "trainer.price_attributions" in names
    assert "main.files" not in names and "main.bakugo_scans" in names
    assert layout.vault_db.name == "supabase_vault.duckdb"


def test_vault_inside_repo_must_be_ignored(monkeypatch):
    monkeypatch.delenv("BAKUGO_VAULT_DB", raising=False)
    from trainer.paths import resolve_vault

    # the real vault location (repo root) is ignored by git and .dockerignore
    assert resolve_vault(None).name == "supabase_vault.duckdb"
    with pytest.raises(UnsafeLocation):
        resolve_vault(REPO_ROOT / "cardcenter" / "vault.sqlite")


# ------------------------------------------------------------------ prices

def test_item_price_resolution_across_scopes(con, layout):
    from cardcenter.pricing import PriceAttribution

    _write(layout, "Pokemon/bin/a.png", _noise(1))
    manifest.refresh(layout, con)
    sha = _sha_of(con, "Pokemon/bin/a.png")
    assert manifest.item_price(con, sha) is None
    manifest.confirm(con, sha, confirmed_by="adam", card_uid="c1", box_key="bin-7",
                     venue="Shop A")
    manifest.attribute_price(con, PriceAttribution("0.25", source="shelf_sign", scope="box",
                                                   scope_key="bin-7"), recorded_by="adam")
    p = manifest.item_price(con, sha)
    assert str(p.amount) == "0.2500" and p.attribution.source == "shelf_sign"
    manifest.attribute_price(con, PriceAttribution("3", source="tag", scope="lot",
                                                   scope_key="lot-1", quantity=3),
                             recorded_by="adam")
    assert manifest.item_price(con, sha).attribution.source == "shelf_sign"   # not in lot-1
    manifest.attribute_price(con, PriceAttribution("1.50", source="sticker", method="ocr"),
                             recorded_by="adam", card_uid="c1")
    p = manifest.item_price(con, sha)
    assert str(p.amount) == "1.5000" and "OCR" in p.basis
    with pytest.raises(ValueError):
        manifest.attribute_price(con, PriceAttribution("1"), recorded_by="model", sha256=sha)


def test_cli_confirm_file_and_import_prices(layout, tmp_path, capsys):
    from trainer.__main__ import main

    _write(layout, "Pokemon/d/PXL_20260101_1.png", _noise(4))
    con = manifest.connect(layout)
    manifest.refresh(layout, con)
    sha = _sha_of(con, "Pokemon/d/PXL_20260101_1.png")
    con.close()
    f = tmp_path / "review.jsonl"
    f.write_text("\n".join([
        json.dumps({"task": "identity", "sha256": sha,
                    "confirm": {"label": "Lechonk", "number": 182, "page_key": "p3",
                                "price": {"text": "3 for $1", "source": "page_label",
                                          "scope": "item"}}}),
        json.dumps({"task": "public_feedback", "confirm": {"label": "ignored"}}),
        json.dumps({"task": "identity", "sha256": "f" * 64}),
    ]))
    root = ["--root", str(layout.root)]
    assert main(root + ["confirm-file", str(f), "--by", "adam"]) == 0
    assert json.loads(capsys.readouterr().out)["confirmed"] == 1
    prices = tmp_path / "prices.jsonl"
    prices.write_text(json.dumps({"amount": "0.10", "source": "shelf_sign", "scope": "page",
                                  "scope_key": "p3", "kind": "asking"}))
    assert main(root + ["import-prices", str(prices), "--by", "adam"]) == 0
    capsys.readouterr()
    con = manifest.connect(layout)
    try:
        p = manifest.item_price(con, sha)
        assert p.attribution.scope == "item" and str(p.amount) == "0.3333"
        assert con.execute("SELECT count(*) FROM price_attributions").fetchone()[0] == 2
    finally:
        con.close()


# ------------------------------------------------------------------ in-situ

def _inbox_capture(layout, name="Lechonk", action="confirm", who="owner-device:abc", price=None):
    day = layout.inbox / "2026-09-16"
    day.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(day / "id1.png"), _noise(9))
    (day / "id1.json").write_text(json.dumps({
        "schema": "bakugo-insitu/1", "image": "id1.png", "confirmed_by": who,
        "identification": {"id": "id1", "name": name, "px_per_mm": 11.0, "quad": "[[0,0]]"},
        "label": {"name": name, "number": 182, "franchise": "pokemon", "action": action},
        "price": price}))


def test_inbox_import_confirms_owner_captures(layout, con):
    from trainer.insitu_import import import_inbox

    _inbox_capture(layout, price={"amount": "2", "currency": "USD", "source": "verbal",
                                  "scope": "item", "kind": "paid", "method": "entered",
                                  "quantity": 1, "observed_at": 1789560000.0})
    manifest.refresh(layout, con)
    res = import_inbox(con, layout)
    assert res == {"applied": 1, "skipped": []}
    row = con.execute("SELECT label, label_number, label_strength, source_type, franchise, "
                      "rel_path FROM files WHERE rel_path LIKE 'inbox/%'").fetchone()
    assert row == ("Lechonk", 182, "confirmed", "app_capture", "pokemon", "inbox/2026-09-16/id1.png")
    who = con.execute("SELECT confirmed_by FROM confirmations").fetchone()[0]
    assert who == "owner-device:abc"
    assert manifest.item_price(con, _sha_of(con, "inbox/2026-09-16/id1.png"), kind="paid") is not None
    assert (layout.inbox / "2026-09-16" / "id1.json.imported").exists()
    assert import_inbox(con, layout)["applied"] == 0          # applied once
    assert con.execute("SELECT count(*) FROM quad_labels").fetchone()[0] == 0


@pytest.mark.parametrize("who,action", [("adam", "confirm"), ("owner-device:x", "reject")])
def test_inbox_import_skips_non_owner_or_non_confirmation(layout, con, who, action):
    from trainer.insitu_import import import_inbox

    _inbox_capture(layout, who=who, action=action)
    manifest.refresh(layout, con)
    res = import_inbox(con, layout)
    assert res["applied"] == 0 and len(res["skipped"]) == 1
    assert con.execute("SELECT count(*) FROM confirmations").fetchone()[0] == 0


def test_public_feedback_is_review_only(tmp_path):
    from cardcenter.insitu import InSituStore
    from trainer.insitu_import import public_feedback_queue

    db = tmp_path / "app.db"
    with InSituStore(str(db)) as st:
        ident = st.record_identification("dev-1", {"name": "Lechonk", "matched_token": "Lechnk"})
        st.add_feedback("dev-1", "correct", identification_id=ident, name="Lechonk ex")
        st.add_feedback("dev-9", "reject")
        st.add_feedback("owner", "reject", owner=True)
    q = public_feedback_queue(str(db))
    assert {r["action"] for r in q} == {"correct", "reject"}
    assert all("device" not in k or k == "devices" for r in q for k in r)
    assert all(r["task"] == "public_feedback" and "sha256" not in r for r in q)
    assert public_feedback_queue(str(tmp_path / "missing.db")) == []
