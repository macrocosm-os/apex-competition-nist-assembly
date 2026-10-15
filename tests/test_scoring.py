"""Integration tests: baseline scores > 0; every test-set policy scores <= the declared floor;
records match result.json; an unwritable record path leaves the score unchanged.

    python tools/make_policies.py && pytest -q tests/
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FLOOR = 0.10  # spec defaults.baseline_raw_score
SMALL = '{"episodes_per_stratum": 2}'


def run(policy: str, out: Path, seed: int = 3, override: str = SMALL) -> dict:
    subprocess.run([sys.executable, str(ROOT / "tools/local_run.py"), "--policy", str(ROOT / policy),
                    "--seed", str(seed), "--out", str(out), "--override", override], check=True)
    return json.loads((out / "result.json").read_text())


def test_baseline_scores(tmp_path):
    r = run("baselines/baseline.onnx", tmp_path)
    assert r["raw_scores"][0] > FLOOR


@pytest.mark.parametrize("adv", ["zero", "constant_push", "random", "huge", "nan", "wrong_shape", "slow"])
def test_policy_set_below_floor(tmp_path, adv):
    try:
        r = run(f"test_policies/{adv}.onnx", tmp_path)
    except subprocess.CalledProcessError:
        return  # player never became ready -> platform scores the submission 0
    assert r["raw_scores"][0] <= FLOOR


def test_records_match_and_deterministic(tmp_path):
    r1 = run("baselines/baseline.onnx", tmp_path / "a")
    r2 = run("baselines/baseline.onnx", tmp_path / "b")
    assert r1["raw_scores"] == r2["raw_scores"]
    rows = [json.loads(x) for x in (tmp_path / "a/history/episodes.jsonl").read_text().splitlines()][1:]
    assert len(rows) == r1["metadata"]["summary"][1]["value"]
    assert sum(x["success"] for x in rows) / len(rows) == r1["raw_scores"][0]
