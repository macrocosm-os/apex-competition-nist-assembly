"""The success predicate: a seated part must be upright, not merely low and centred.

    pytest -q tests/test_success.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
import mujoco  # noqa: E402
from nist_sim.conditions import sample_episodes  # noqa: E402
from nist_sim.env import BLOCK_H, PEG_LEN, Episode  # noqa: E402

OPEN_AND_HOLD = np.array([0, 0, 0, 0, 1.0])


def _one_per_part():
    _, conds = sample_episodes(7, 1)
    seen = {}
    for c in conds:
        seen.setdefault((c.shape, c.diameter), c)
    return list(seen.values())


def _place(e: Episode, xyz, quat):
    q = e.part_qadr
    e.data.qpos[q : q + 3] = xyz
    e.data.qpos[q + 3 : q + 7] = quat
    e.data.qvel[e.part_vadr : e.part_vadr + 6] = 0
    hx, hy = e.c.hole_xy
    e.data.qpos[0:3] = [hx, hy, 0.12]  # hand well above the part
    e.target[0:3] = [hx, hy, 0.12]
    mujoco.mj_forward(e.model, e.data)


def _hold(e: Episode, steps: int = 15) -> None:
    for _ in range(steps):
        e.step(OPEN_AND_HOLD)
        if e.done:
            break


@pytest.mark.parametrize("c", _one_per_part(), ids=lambda c: f"{c.shape}_{round(c.diameter * 1000)}mm")
def test_part_lying_across_hole_is_not_assembled(c):
    e = Episode(c, np.random.default_rng(0))
    hx, hy = c.hole_xy
    # On its side, centred over the hole, resting on the block's top face.
    _place(e, [hx, hy, BLOCK_H + c.diameter / 2 + 2e-4], [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0])
    _hold(e)
    assert not e.success, e.terminal_reason


@pytest.mark.parametrize("c", _one_per_part(), ids=lambda c: f"{c.shape}_{round(c.diameter * 1000)}mm")
@pytest.mark.parametrize("flipped", [False, True])
def test_upright_seated_part_is_assembled(c, flipped):
    e = Episode(c, np.random.default_rng(0))
    hx, hy = c.hole_xy
    quat = [0, 1, 0, 0] if flipped else [1, 0, 0, 0]  # the peg is end-symmetric
    _place(e, [hx, hy, PEG_LEN / 2 + 2e-4], quat)
    _hold(e)
    assert e.success and e.terminal_reason == "assembled", e.terminal_reason
