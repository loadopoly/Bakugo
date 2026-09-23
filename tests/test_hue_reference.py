"""Back-hue references: learned, released, noise-aware; price labels by profile."""

import json

import cv2
import numpy as np
import pytest

from cardcenter import hue_reference as hr
from ingest import binder_sticker


def _back(h, s=170, v=150, seed=0, shape=(120, 90), jitter=1.5):
    rng = np.random.default_rng(seed)
    hsv = np.zeros((*shape, 3), np.uint8)
    hsv[..., 0] = np.clip(h + rng.normal(0, jitter, shape), 0, 179)
    hsv[..., 1] = s
    hsv[..., 2] = v
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_bright_brown_back_is_not_masked_as_a_price_label():
    # The old fixed mask discarded every saturated pixel with hue < 25 and
    # value > 80, so a Yu-Gi-Oh back at its own reference hue was never seen.
    g = binder_sticker.classify_back_franchise(_back(12))
    assert g.guess == "yugioh" and "uncalibrated" in g.note


def test_small_orange_label_is_still_excluded():
    img = _back(108)
    img[:25, :30] = cv2.cvtColor(np.full((25, 30, 3), (10, 220, 220), np.uint8), cv2.COLOR_HSV2BGR)
    m = hr.measure_back_hue(img)
    assert 0 < m.label_frac < 0.35 and abs(m.hue - 108) < 1.0
    none = hr.measure_back_hue(img, hr.LabelProfile.none())
    assert none.label_frac == 0.0
    assert none.spread > 3 * m.spread and none.se > m.se   # the label widens the hue spread


def test_abstains_between_references_and_far_from_all():
    reg = hr.HueRegistry([hr.HueReference("a", 50, 6, 10, "file"),
                          hr.HueReference("b", 58, 6, 10, "file")])
    assert hr.classify_back(hr.measure_back_hue(_back(54)), reg).guess is None
    assert hr.classify_back(hr.measure_back_hue(_back(140)), reg).guess is None
    assert hr.classify_back(hr.measure_back_hue(_back(50)), reg).guess == "a"


def test_standard_error_grows_with_noise_and_shrinks_with_area():
    clean = hr.measure_back_hue(_back(100, jitter=0.5))
    rng = np.random.default_rng(1)
    noisy_img = np.clip(_back(100, jitter=0.5).astype(float) + rng.normal(0, 20, (120, 90, 3)), 0, 255)
    noisy = hr.measure_back_hue(noisy_img.astype(np.uint8))
    big = hr.measure_back_hue(_back(100, jitter=0.5, shape=(480, 360)))
    assert noisy.noise_sigma > clean.noise_sigma and noisy.se > clean.se
    assert big.se < clean.se


def test_registry_params_roundtrip_and_file_override(tmp_path, monkeypatch):
    reg = hr.HueRegistry(hr.DEFAULT_REFERENCES).with_reference(
        hr.fit_reference("yugioh", [18, 20, 22], "released"))
    params = reg.to_params()
    back = hr.HueRegistry.from_params(params, "released")
    y = back.get("yugioh")
    assert abs(y.hue - 20) < 0.01 and y.n == 3 and y.calibrated
    f = tmp_path / "hues.json"
    f.write_text(json.dumps({"yugioh_hue": 21.0, "yugioh_spread": 3.0, "yugioh_n": 40}))
    monkeypatch.setenv("BAKUGO_HUE_REFERENCES", str(f))
    loaded = hr.load_registry(refresh=True)
    assert loaded.get("yugioh").hue == 21.0 and loaded.get("yugioh").source == "file"
    g = binder_sticker.classify_back_franchise(_back(21))
    assert g.guess == "yugioh" and "uncalibrated" not in g.note
    monkeypatch.delenv("BAKUGO_HUE_REFERENCES")
    assert hr.load_registry(refresh=True).get("yugioh").source == "uncalibrated"


def test_released_artifact_feeds_registry(tmp_path, monkeypatch):
    from cardcenter import release_guard as rg

    params = {"yugioh_hue": 16.0, "yugioh_spread": 4.0, "yugioh_n": 12}
    monkeypatch.setattr(rg, "released_params",
                        lambda kind, root=None: params if kind == "back_hue" else None)
    reg = hr.load_registry(refresh=True)
    assert reg.get("yugioh").hue == 16.0 and reg.get("yugioh").source == "released"
    monkeypatch.undo()
    hr.load_registry(refresh=True)


def test_circular_mean_wraps():
    mean, std = hr.circular_mean_std([178, 2])
    assert min(mean, 180 - mean) < 0.01 and std < 3
