"""What the private data trains, as tasks the loop can fit, score and promote.

Each task has:

* ``exposure`` -- ``generic`` (parameters that do not encode the collection;
  may ship after promotion) or ``collection`` (never ships);
* ``default_params()`` -- the behaviour of the code as it stands, which is the
  champion until something beats it;
* ``fit(con, layout)`` -- train on confirmed train items only;
* ``score(con, layout, params, model_id)`` -- per-item scores (1.0 correct,
  0.0 wrong) on the frozen test set, each with a resolution slice.

The scores are paired: candidate and champion are scored on the same items,
so ``promote`` can compare them item by item.

Tasks implemented here:

``quad_detect``
    Card-outline detection on clutter. Labels are human-confirmed corners
    (``quad_labels``). The trainable parameters are ``find_card_quad``'s
    ``min_area_frac`` and whether the outline is snapped to the measured
    edge information afterwards (``edge_information.locate_card``); an item is correct when the detected quad's mean corner
    error is under ``CORNER_TOL`` of the labelled quad's diagonal. The labelled
    set doubles as a regression set for the detector whatever happens to the
    parameter.

``back_hue``
    Card-back hue references (``cardcenter.hue_reference``) for every
    franchise with at least three confirmed binder-back images: circular mean
    and spread of their hues. Scored by classifying confirmed test backs.
    Replaces the hard-coded ``binder_sticker`` constants once promoted and
    released.

The embedding recognizer (second vote) runs on the device, in the app's
service worker, with an index built from that device's own confirmations; it
is not a trainer task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .data import quad_items, frozen_test_items, training_items
from .paths import Layout

CORNER_TOL = 0.02


@dataclass(frozen=True)
class ItemScore:
    key: str          # sha256[:card_index]
    slice: str
    value: float


def resolution_slice(px_per_mm: Optional[float]) -> str:
    if px_per_mm is None:
        return "unknown"
    if px_per_mm < 6:
        return "<6 px/mm"
    if px_per_mm < 9:
        return "6-9"
    if px_per_mm < 12:
        return "9-12"
    return ">=12"


def _corner_error_frac(found: np.ndarray, truth: np.ndarray) -> float:
    from cardcenter.geometry import order_quad

    f = order_quad(np.asarray(found, dtype=np.float64).reshape(4, 2))
    t = order_quad(np.asarray(truth, dtype=np.float64).reshape(4, 2))
    diag = float(np.linalg.norm(t[0] - t[2]))
    return float(np.mean(np.linalg.norm(f - t, axis=1))) / max(diag, 1e-9)


def _px_per_mm_from_quad(truth) -> float:
    t = np.asarray(truth, dtype=np.float64).reshape(4, 2)
    long_px = max(np.linalg.norm(t[0] - t[3]), np.linalg.norm(t[1] - t[2]))
    return float(long_px / 88.0)   # standard card long side, mm


class QuadDetectTask:
    name = "quad_detect"
    exposure = "generic"
    grid = (0.002, 0.003, 0.005, 0.008, 0.012, 0.02, 0.03)

    def __init__(self, detector: Optional[Callable] = None, reader: Optional[Callable] = None):
        self._detector = detector
        self._reader = reader

    def default_params(self) -> dict:
        return {"min_area_frac": 0.008, "information_snap": 1}

    def _detect(self, img, params, prefer):
        if self._detector is not None:
            return self._detector(img, params["min_area_frac"], prefer)
        if int(params.get("information_snap", 0)):
            from cardcenter.edge_information import locate_card

            return locate_card(img, prefer_point=prefer,
                               min_area_frac=params["min_area_frac"])[0]
        from cardcenter.geometry import find_card_quad

        quad, _, _ = find_card_quad(img, min_area_frac=params["min_area_frac"],
                                    prefer_point=prefer)
        return quad

    def _read(self, path):
        if self._reader is not None:
            return self._reader(path)
        import cv2

        return cv2.imread(str(path))

    def _score_entries(self, entries, params) -> list[ItemScore]:
        out = []
        for e in entries:
            img = self._read(e["path"])
            for i, truth in enumerate(e["quads"]):
                t = np.asarray(truth, dtype=np.float64)
                ppm = e["px_per_mm"] or _px_per_mm_from_quad(t)
                ok = 0.0
                if img is not None:
                    try:
                        found = self._detect(img, params, tuple(t.mean(axis=0)))
                        ok = 1.0 if _corner_error_frac(found, t) <= CORNER_TOL else 0.0
                    except Exception:
                        ok = 0.0
                out.append(ItemScore(f"{e['sha256']}:{i}", resolution_slice(ppm), ok))
        return out

    def fit(self, con, layout: Layout):
        entries = quad_items(con, layout, "train")
        if not entries:
            return None, []
        best = None
        default = self.default_params()
        for snap in (1, 0):
            for v in self.grid:
                params = {"min_area_frac": v, "information_snap": snap}
                s = self._score_entries(entries, params)
                acc = sum(x.value for x in s) / len(s)
                # ties go to the code default
                key = (acc, snap == default["information_snap"],
                       -abs(math.log(v / default["min_area_frac"])))
                if best is None or key > best[0]:
                    best = (key, params)
        return best[1], [e["sha256"] for e in entries]

    def score(self, con, layout: Layout, params: dict, model_id: str) -> list[ItemScore]:
        import datetime as _dt

        entries = [e for e in quad_items(con, layout, "test")
                   if con.execute("SELECT 1 FROM frozen_test WHERE sha256 = ?",
                                  [e["sha256"]]).fetchone()]
        con.execute("INSERT INTO test_uses (model_id, task, n_items, used_at) VALUES (?, ?, ?, ?)",
                    [model_id, self.name, len(entries), _dt.datetime.now().replace(microsecond=0)])
        return self._score_entries(entries, params)


class BackHueTask:
    """Card-back hue references for every franchise with confirmed backs.

    Replaces the hard-coded ``_POKEMON_HUE`` / ``_YUGIOH_HUE``: fit the
    circular mean and spread of per-back hues (``hue_reference``), release
    them as the flat ``back_hue`` artifact, and score by classifying
    confirmed test backs with the candidate registry.
    """

    name = "back_hue"
    exposure = "generic"
    min_backs = 3

    def __init__(self, reader: Optional[Callable] = None):
        self._reader = reader
        self.last_fit_note = ""

    def default_params(self) -> dict:
        from cardcenter.hue_reference import DEFAULT_REFERENCES, HueRegistry

        return HueRegistry(DEFAULT_REFERENCES).to_params()

    def _read(self, path):
        if self._reader is not None:
            return self._reader(path)
        import cv2

        return cv2.imread(str(path))

    def fit(self, con, layout: Layout):
        from cardcenter.hue_reference import (DEFAULT_REFERENCES, HueRegistry,
                                              fit_reference, measure_back_hue)

        items = training_items(con, layout, source_type="binder_back")
        by_franchise: dict[str, list] = {}
        used, rejected, notes = [], 0, []
        for it in items:
            if not it.franchise:
                continue
            img = self._read(it.path)
            m = measure_back_hue(img) if img is not None else None
            if m is None or m.hue is None:
                rejected += 1
                continue
            by_franchise.setdefault(it.franchise, []).append(m.hue)
            used.append(it.sha256)
        reg = HueRegistry(DEFAULT_REFERENCES)
        fitted = 0
        for fr, hues in sorted(by_franchise.items()):
            if len(hues) < self.min_backs:
                notes.append(f"{fr}: {len(hues)} backs (< {self.min_backs}), kept existing")
                continue
            reg = reg.with_reference(fit_reference(fr, hues, "released"))
            fitted += 1
            notes.append(f"{fr}: {len(hues)} backs")
        self.last_fit_note = (f"{len(items)} confirmed backs, {rejected} unmeasurable; "
                              + "; ".join(notes))
        if not fitted:
            return None, []
        return reg.to_params(), used

    def score(self, con, layout: Layout, params: dict, model_id: str) -> list[ItemScore]:
        from cardcenter.hue_reference import HueRegistry, classify_back, measure_back_hue

        reg = HueRegistry.from_params(params, "candidate")
        known = {r.franchise for r in reg}
        items = [i for i in frozen_test_items(con, layout, model_id=model_id, task=self.name,
                                              source_type="binder_back")
                 if i.franchise in known]
        out = []
        for it in items:
            img = self._read(it.path)
            guess = classify_back(measure_back_hue(img), reg).guess if img is not None else None
            out.append(ItemScore(it.sha256, resolution_slice(it.px_per_mm),
                                 1.0 if guess == it.franchise else 0.0))
        return out


TASKS = {t.name: t for t in (QuadDetectTask, BackHueTask)}


def get_task(name: str):
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]()
