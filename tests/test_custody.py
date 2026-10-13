"""Custody evidence: a photo of a card beside an exchange challenge code, kept with what could be checked."""

import json

import pytest

from cardcenter import custody

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


@pytest.fixture(autouse=True)
def _dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDCENTER_CUSTODY_DIR", str(tmp_path / "custody"))
    return tmp_path / "custody"


def test_evidence_is_kept_with_the_card_read_and_whether_the_code_was_seen(_dir):
    doc = custody.record_evidence("dev_abc", "k7q-4mp", "bakugo:identifications:ab12", JPEG,
                                  identify=lambda img: {"identified": True, "name": "Charizard", "dex": 6, "secret": "x"},
                                  ocr=lambda img: "CHARIZARD K7Q 4MP 6/102")
    assert doc["challenge"] == "K7Q-4MP" and doc["challenge_seen"] is True and doc["card"]["name"] == "Charizard"
    assert "secret" not in doc["card"] and len(doc["sha256"]) == 64
    on_disk = json.loads((_dir / f"{doc['evidence_id']}.json").read_text())
    assert on_disk["device_id"] == "dev_abc" and (_dir / doc["image"]).read_bytes() == JPEG
    assert [d["evidence_id"] for d in custody.evidence_for("dev_abc", "K7Q-4MP")] == [doc["evidence_id"]]
    assert custody.evidence_for("dev_other", "K7Q-4MP") == []          # another device does not see it


def test_a_reader_that_fails_still_keeps_the_photo_for_the_verifier():
    def boom(img):
        raise RuntimeError("could not locate a card")

    doc = custody.record_evidence("dev_abc", "ABC-DEF", "bakugo:field:aa", JPEG, identify=boom, ocr=lambda img: "")
    assert doc["card"]["identified"] is False and doc["challenge_seen"] is False


@pytest.mark.parametrize("challenge, asset, image, why", [
    ("ABC-12O", "bakugo:identifications:x", JPEG, "challenge code"),          # O and 1 are never issued
    ("ABC-DEF", "x y", JPEG, "asset_id"),
    ("ABC-DEF", "bakugo:identifications:x", b"", "attach a photo"),
    ("ABC-DEF", "bakugo:identifications:x", b"GIF89a....", "JPEG, PNG or WebP"),
])
def test_bad_evidence_is_refused(challenge, asset, image, why):
    with pytest.raises(custody.CustodyError, match=why):
        custody.record_evidence("dev_abc", challenge, asset, image, ocr=lambda img: "")


def test_the_code_is_found_despite_ocr_confusions():
    assert custody.challenge_in("k7q 4mp", "K7Q-4MP")
    assert custody.challenge_in("XX K7Q-4MP YY", "K7Q-4MP")
    assert not custody.challenge_in("K7Q 4MQ", "K7Q-4MP")
