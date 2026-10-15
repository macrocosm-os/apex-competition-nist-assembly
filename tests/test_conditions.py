"""Fast checks on round condition sampling (no player, no HTTP)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
from nist_sim import conditions  # noqa: E402
from nist_sim.reserve import reserved_from_secret  # noqa: E402


@pytest.fixture
def reserved(tmp_path, monkeypatch):
    spec = reserved_from_secret("test-only-secret")
    p = tmp_path / "reserved.json"
    p.write_text(json.dumps(spec))
    monkeypatch.setattr(conditions, "RESERVED_PATH", p)
    return spec["ranges"]


def test_same_seed_same_round():
    _, a = conditions.sample_episodes(5, 2)
    _, b = conditions.sample_episodes(5, 2)
    assert a == b


def test_rounds_differ():
    _, a = conditions.sample_episodes(5, 2)
    _, b = conditions.sample_episodes(6, 2)
    assert a != b


def test_strata_fixed():
    for seed in (1, 2, 3):
        _, eps = conditions.sample_episodes(seed, 4)
        assert len(eps) == 2 * 4 * 5 * 4


def test_live_rounds_avoid_reserved(reserved):
    for seed in range(5):
        _, eps = conditions.sample_episodes(seed, 16)
        for name, (a, b) in reserved.items():
            assert not any(a <= getattr(e, name) <= b for e in eps)


def test_reserved_split_inside(reserved):
    _, eps = conditions.sample_episodes(0, 16, "reserved")
    for name, (a, b) in reserved.items():
        assert all(a <= getattr(e, name) <= b for e in eps)


def test_reserved_split_needs_build_input(tmp_path, monkeypatch):
    monkeypatch.setattr(conditions, "RESERVED_PATH", tmp_path / "missing.json")
    with pytest.raises(RuntimeError):
        conditions.sample_episodes(0, 1, "reserved")
