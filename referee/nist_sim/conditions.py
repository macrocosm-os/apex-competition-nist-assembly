"""Per-round condition sampling. Everything derives from the platform SEED.

Stratum proportions (shape x size x reset type) are fixed every round; what rotates is
the draw inside each stratum: clearance, friction, part mass, controller gain, observation
noise, hole placement, and start-state offsets.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .env import EpisodeConditions

RESERVED_PATH = Path(os.environ.get("NIST_RESERVED_PATH", Path(__file__).with_name("reserved.json")))

SHAPES = ("round", "square")
DIAMETERS = (0.004, 0.008, 0.012, 0.016)  # NIST ATB-1 peg sizes
RESET_TYPES = ("grasped_above", "grasped_far", "near_hole", "loose", "lying")

# Live ranges. Every round covers each range evenly inside every stratum (Latin-hypercube draw),
# so difficulty stays stationary while the specific values rotate with the seed.
RANGES = {
    "clearance": (0.00015, 0.0006),
    "friction": (0.5, 1.2),
    "peg_mass": (0.005, 0.05),
    "gain_scale": (0.7, 1.3),
    "obs_noise": (0.0, 0.0005),
}


def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def load_reserved() -> dict[str, list[float]]:
    """Sub-ranges excluded from live rounds (absent in development builds)."""
    try:
        return {k: [float(a), float(b)] for k, (a, b) in json.loads(RESERVED_PATH.read_text())["ranges"].items()}
    except FileNotFoundError:
        return {}


def _map(u: np.ndarray, lo: float, hi: float, gap: list[float] | None, inside: bool) -> np.ndarray:
    if gap is None:
        return lo + u * (hi - lo)
    a, b = gap
    if inside:
        return a + u * (b - a)
    x = lo + u * ((hi - lo) - (b - a))
    return np.where(x >= a, x + (b - a), x)


def _lhs(rng, k: int, reserved: dict, inside: bool) -> dict[str, np.ndarray]:
    out = {}
    for name, (lo, hi) in RANGES.items():
        u = (rng.permutation(k) + rng.uniform(0, 1, k)) / k
        out[name] = _map(u, lo, hi, reserved.get(name), inside)
    return out


def sample_episodes(seed: int, episodes_per_stratum: int, split: str = "live") -> tuple[dict, list[EpisodeConditions]]:
    reserved = load_reserved()
    inside = split == "reserved"
    if inside and not reserved:
        raise RuntimeError("no reserved ranges in this build")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4E495354]))
    rc = {k: list(v) for k, v in RANGES.items()}
    eps: list[EpisodeConditions] = []
    for shape in SHAPES:
        for dia in DIAMETERS:
            for rt in RESET_TYPES:
                lhs = _lhs(rng, episodes_per_stratum, reserved, inside)
                for j in range(episodes_per_stratum):
                    hole = (_u(rng, -0.08, 0.08), _u(rng, -0.08, 0.08))
                    if rt == "grasped_above":
                        r = _u(rng, 0.0, 0.005); th = _u(rng, 0, 2 * np.pi)
                        off = (r * np.cos(th), r * np.sin(th)); h = _u(rng, 0.005, 0.025)
                        yaw = _u(rng, -0.15, 0.15) if shape == "square" else 0.0
                    elif rt == "grasped_far":
                        r = _u(rng, 0.01, 0.03); th = _u(rng, 0, 2 * np.pi)
                        off = (r * np.cos(th), r * np.sin(th)); h = _u(rng, 0.02, 0.05)
                        yaw = _u(rng, -0.5, 0.5) if shape == "square" else 0.0
                    elif rt == "near_hole":
                        # "near_hole": part resting on the hole's rim, just off-centre
                        r = _u(rng, 0.0008, 0.0025); th = _u(rng, 0, 2 * np.pi)
                        off = (r * np.cos(th), r * np.sin(th))
                        h = 0.0003; yaw = _u(rng, -0.1, 0.1) if shape == "square" else 0.0
                    elif rt == "lying":
                        r = _u(rng, 0.03, 0.06); th = _u(rng, 0, 2 * np.pi)
                        off = (r * np.cos(th), r * np.sin(th)); h = _u(rng, 0.01, 0.03)
                        yaw = _u(rng, -np.pi, np.pi)
                    else:  # loose
                        r = _u(rng, 0.03, 0.06); th = _u(rng, 0, 2 * np.pi)
                        off = (r * np.cos(th), r * np.sin(th)); h = _u(rng, 0.01, 0.03)
                        yaw = _u(rng, -np.pi / 4, np.pi / 4)
                    eps.append(EpisodeConditions(
                        shape=shape, diameter=dia,
                        clearance=float(lhs["clearance"][j]), friction=float(lhs["friction"][j]),
                        peg_mass=float(lhs["peg_mass"][j]), gain_scale=float(lhs["gain_scale"][j]),
                        obs_noise=float(lhs["obs_noise"][j]), reset_type=rt,
                        hole_xy=hole, offset_xy=(float(off[0]), float(off[1])), height=h, yaw_err=yaw,
                    ))
    return rc, eps
