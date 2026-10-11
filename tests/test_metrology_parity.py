"""Bakugo and the shared holon.metrology/1 contract measure grading alike.

Metrology owns the measurement (borders, floors, defect sizes); Bakugo owns
the mapping to grades. These checks keep Bakugo's own physics and the
vendored contract from drifting apart.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from cardcenter import photometrology as pm
from cardcenter.information import ChannelConditions, SensorModel, cramer_rao_edge_px, \
    cramer_rao_ratio_pp, quantum_floor_px

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "metrology_golden.json"


def test_vendored_module_is_the_canonical_one():
    gv = json.loads(FIXTURE.read_text(encoding="utf-8"))
    src = Path(pm.__file__).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(src).hexdigest() == gv["module_sha256"]


@pytest.mark.parametrize("contrast,noise,psf,rows", [(60.0, 4.0, 1.2, 40), (25.0, 6.0, 2.0, 300)])
def test_edge_floors_match(contrast, noise, psf, rows):
    c = ChannelConditions(contrast=contrast, noise_sigma=noise, psf_sigma_px=psf, rows=rows)
    assert pm.cramer_rao_edge_px(contrast, noise, psf, rows) == pytest.approx(cramer_rao_edge_px(c))
    assert pm.quantum_floor_edge_px(contrast, 120.0, psf, rows) == pytest.approx(
        quantum_floor_px(c, 120.0, SensorModel()))


def test_centering_uncertainty_matches_bakugo():
    c = ChannelConditions(contrast=60.0, noise_sigma=4.0, psf_sigma_px=1.2, rows=40)
    px_per_mm = 22.0
    left_mm, right_mm = 3.1, 2.4
    bakugo_pp = cramer_rao_ratio_pp(c, left_mm, right_mm, px_per_mm)
    edge = cramer_rao_edge_px(c)
    ours = pm.centering(left_mm * px_per_mm, right_mm * px_per_mm, 50.0, 50.0, edge)
    assert ours["h_ratio_sigma"] == pytest.approx(bakugo_pp)
    assert ours["h_ratio_pct"] == pytest.approx(100 * 3.1 / 5.5)


def test_aspects_are_bakugos():
    from cardcenter.grading import ASPECTS
    assert tuple(ASPECTS) == pm.ASPECTS
