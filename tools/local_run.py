"""Run one full evaluation locally WITHOUT Docker: player server + referee, same contract.

    python tools/local_run.py --policy baselines/baseline.onnx --seed 7 --input fixtures/input.json --out out/

Needs: numpy, mujoco, onnx, onnxruntime (versions pinned in referee/requirements.txt and
player/requirements.txt). For the honest sandboxed test use the Docker commands in README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input", default=str(ROOT / "fixtures" / "input.json"))
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--override", default="{}", help="JSON merged into the round input")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    port = free_port()
    env = {**os.environ, "NIST_POLICY_PATH": str(Path(a.policy).resolve()), "PYTHONPATH": str(ROOT / "player")}
    player = subprocess.Popen([sys.executable, str(ROOT / "player" / "launch.py"), "--port", str(port)], env=env)
    try:
        cfg = {**json.loads(Path(a.input).read_text()), **json.loads(a.override)}
        os.environ.update({"MATCH_ID": "local", "SEED": str(a.seed), "NUM_PLAYERS": "1",
                           "PLAYER_URLS": f"http://127.0.0.1:{port}", "CONFIG_JSON": json.dumps(cfg),
                           "NIST_HISTORY_DIR": str(out / "history")})
        sys.path.insert(0, str(ROOT / "referee"))
        import gym_v1.referee as gr  # vendored copy under referee/
        gr.RESULT_PATH = out / "result.json"
        gr.TRACE_PATH = out / "trace.jsonl"
        from referee import NistLiteReferee  # noqa: E402

        t = time.monotonic()
        try:
            NistLiteReferee().run()
        except Exception as e:  # mirrors the platform: no result.json -> referee failure
            print(f"referee failed: {e!r}", file=sys.stderr)
            return 6
        res = json.loads((out / "result.json").read_text())
        print(json.dumps({"raw_score": res["raw_scores"][0], "terminal_reason": res["terminal_reason"],
                          "player_calls": res["steps"], "wall_s": round(time.monotonic() - t, 1),
                          "by_reset_type": {s["key"]: round(s["value"], 3) for s in res["metadata"].get("summary", [])},
                          "fault": res["metadata"].get("details", {}).get("player_fault")}))
        return 0
    finally:
        player.kill()


if __name__ == "__main__":
    raise SystemExit(main())
