"""Read nist_lite evaluation records (downloaded artifacts or a local run's out/history/).

    python tools/read_records.py out/history            # per-stratum table
    python tools/read_records.py out/history --episode 32   # one trajectory
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SUPPORTED_MAJOR = 1


def _check(rec: dict) -> None:
    name, _, major = rec.get("format", "/").partition("/")
    if name != "nist_lite_record" or int(major or 0) > SUPPORTED_MAJOR:
        raise SystemExit(f"unsupported record format: {rec.get('format')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("history_dir")
    ap.add_argument("--episode", type=int)
    a = ap.parse_args()
    d = Path(a.history_dir)
    if a.episode is not None:
        t = json.loads((d / f"traj_{a.episode:05d}.json").read_text())
        _check(t)
        for s in t["steps"]:
            print(s["t"], "part", s["part_pos"], "hand", s["hand"], "a", s["action"])
        return
    lines = [json.loads(x) for x in (d / "episodes.jsonl").read_text().splitlines() if x.strip()]
    for r in lines:
        _check(r)
    eps = [r for r in lines if "episode" in r]
    agg = defaultdict(lambda: [0, 0])
    reasons = defaultdict(int)
    for r in eps:
        c = r["conditions"]
        k = (c["shape"], round(c["diameter"] * 1000), c["reset_type"])
        agg[k][0] += r["success"]
        agg[k][1] += 1
        reasons[r["terminal_reason"]] += 1
    print(f"episodes: {len(eps)}  success: {sum(r['success'] for r in eps) / max(1, len(eps)):.4f}")
    print("terminal reasons:", dict(reasons))
    for k in sorted(agg):
        s, n = agg[k]
        print(f"  {k[0]:6s} {k[1]:2d}mm {k[2]:14s} {s:3d}/{n:<3d} {s / n:.2f}")


if __name__ == "__main__":
    main()
