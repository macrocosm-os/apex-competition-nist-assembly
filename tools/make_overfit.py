"""Build an over-fitted reference policy for one round, and check that round rotation defeats it.

The policy memorises one round: for every episode of seed S it tries a handful of controller
variants, keeps the observation/action stream of the first one that assembles, and serves the
union as a nearest-neighbour lookup (pure ONNX). On round S it replays those successes exactly;
on any other round it is a lookup table of someone else's episodes.

    python tools/make_overfit.py --seed 11 --out test_policies/overfit_s11.onnx
    python tools/local_run.py --policy test_policies/overfit_s11.onnx --seed 11   # its own round
    python tools/local_run.py --policy test_policies/overfit_s11.onnx --seed 12   # a new round
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "referee"))
from nist_sim.conditions import sample_episodes  # noqa: E402
from nist_sim.env import OBS_DIM, PEG_LEN, Episode  # noqa: E402

PER_STRATUM = 16


def controller(xy_gain=0.8, descend=-0.5, lift_steps=0, yaw_gain=12.5):
    def act(o, t):
        prel = o[9:12] / 10
        a = np.zeros(5)
        if t < lift_steps:
            a[2] = 1.0
            a[4] = -1
            return a
        if prel[2] < PEG_LEN / 2 + 0.0015:
            a[2] = 1
            a[4] = 1
            return a
        a[0:2] = np.clip(-prel[0:2] / 0.004 * xy_gain, -1, 1)
        a[2] = descend
        a[3] = np.clip(-yaw_gain * o[15], -1, 1)
        a[4] = -1
        return a
    return act


VARIANTS = [controller(), controller(lift_steps=4), controller(xy_gain=0.4), controller(xy_gain=1.2),
            controller(descend=-0.25), controller(descend=-0.9), controller(lift_steps=8, xy_gain=0.5),
            controller(yaw_gain=25.0), controller(descend=-0.25, xy_gain=0.4, lift_steps=4)]


def _episode(args):
    seed, i = args
    _, conds = sample_episodes(seed, PER_STRATUM)
    for pol in VARIANTS:
        ss = np.random.SeedSequence([seed, 0x5245, i])  # identical to the referee's per-episode streams
        r1, r2 = (np.random.default_rng(s) for s in ss.spawn(2))
        e = Episode(conds[i], r1)
        obs, acts, t = [], [], 0
        while not e.done:
            o = e.observe(r2)
            a = np.clip(pol(o, t), -1, 1)
            obs.append(np.round(o, 6))
            acts.append(a)
            e.step(a)
            t += 1
        if e.success:
            return np.array(obs, np.float32), np.array(acts, np.float32)
    return None


def build_nn(keys: np.ndarray, vals: np.ndarray, path: Path) -> None:
    i = helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["B", OBS_DIM])
    o = helper.make_tensor_value_info("action", TensorProto.FLOAT, ["B", 5])
    inits = [
        numpy_helper.from_array(keys.T.copy(), "KT"),
        numpy_helper.from_array((keys ** 2).sum(1).astype(np.float32), "K2"),
        numpy_helper.from_array(vals, "V"),
        numpy_helper.from_array(np.array(-2.0, np.float32), "m2"),
    ]
    nodes = [
        helper.make_node("MatMul", ["obs", "KT"], ["xk"]),
        helper.make_node("Mul", ["xk", "m2"], ["xk2"]),
        helper.make_node("Add", ["xk2", "K2"], ["d"]),          # ||k||^2 - 2 x.k (||x||^2 is constant per row)
        helper.make_node("ArgMin", ["d"], ["idx"], axis=1, keepdims=0),
        helper.make_node("Gather", ["V", "idx"], ["action"], axis=0),
    ]
    m = helper.make_model(helper.make_graph(nodes, "overfit_nn", [i], [o], inits),
                          opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(ROOT / "test_policies" / "overfit.onnx"))
    ap.add_argument("--procs", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()
    _, conds = sample_episodes(a.seed, PER_STRATUM)
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(_episode, [(a.seed, i) for i in range(len(conds))])
    got = [r for r in res if r is not None]
    keys = np.concatenate([r[0] for r in got])
    vals = np.concatenate([r[1] for r in got])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    build_nn(keys, vals, Path(a.out))
    print(f"memorised {len(got)}/{len(conds)} episodes of seed {a.seed} ({len(keys)} steps) -> {a.out}")


if __name__ == "__main__":
    main()
