"""A card at the edge of the photo is refused for that reason, not for the lighting (2026-10-13 field still)."""

import numpy as np
import pytest

from cardcenter.framing import Framing, sides_at_frame_edge
from cardcenter.types import CaptureSpec


def _framing(quad, shape=(3072, 4080), scale=1.0, origin=(0.0, 0.0)):
    q = (np.asarray(quad, dtype=np.float64) - np.asarray(origin)) * scale
    return Framing(np.zeros((10, 10, 3), np.uint8), CaptureSpec(), scale, origin, True, 20.0, shape, q, 0.0)


def test_the_field_still_with_the_top_cut_off_is_named():
    # the outline Bakugo found on 20261014T021118 (top corners 45-91 px from the frame at ~32 px/mm)
    f = _framing([[905, 91], [2933, 45], [3044, 2968], [858, 2969]], shape=(3072, 4080))
    assert sides_at_frame_edge(f) == ["top"]


def test_a_card_with_background_all_round_is_not_flagged():
    f = _framing([[293, 906], [1307, 895], [1342, 2351], [346, 2392]], shape=(3840, 2160))
    assert sides_at_frame_edge(f) == []


def test_crop_scale_and_origin_are_undone_before_judging():
    src = [[10, 500], [1010, 500], [1010, 1900], [10, 1900]]          # left edge 10 px from the frame
    f = _framing(src, shape=(2400, 1800), scale=0.5, origin=(5.0, 400.0))
    assert sides_at_frame_edge(f) == ["left"]


def test_no_outline_means_nothing_to_say():
    import dataclasses

    f = dataclasses.replace(_framing([[0, 0], [1, 0], [1, 1], [0, 1]]), quad=None)
    assert sides_at_frame_edge(f) == []
