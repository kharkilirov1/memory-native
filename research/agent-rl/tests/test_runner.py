"""Tests for run_episode using FakeAD."""

import random
from pathlib import Path

from deskenv.ad import FakeAD, _ScriptedStep
from deskenv.runner import run_episode


def _snap_ok(tree: dict, title: str = "note.txt") -> dict:
    return {
        "version": "2",
        "ok": True,
        "command": "snapshot",
        "data": {
            "app": "mousepad",
            "complete": True,
            "snapshot_id": "s1",
            "tree": tree,
            "window": {"id": "w-1", "title": title},
        },
    }


def _act_ok() -> dict:
    return {"ok": True, "command": "act"}


SIMPLE_TREE = {
    "role": "window",
    "name": "mousepad",
    "ref_id": "@s1:w1",
    "children": [
        {
            "role": "textfield",
            "name": "text area",
            "ref_id": "@s1:t1",
            "available_actions": ["Click", "Type"],
        },
    ],
}


def test_run_episode_done_immediately(tmp_path):
    """run_episode with a policy that says 'done' right away."""
    snap = _snap_ok(SIMPLE_TREE)
    ad = FakeAD([_ScriptedStep(snapshot=snap)])
    rng = random.Random(42)

    result = run_episode(
        session=None,
        ad=ad,
        task_kind="write_save",
        rng=rng,
        policy=lambda obs: 'ACTION: {"op": "done"}',
        max_steps=15,
    )
    assert "reward" in result
    assert "steps" in result
    assert "info" in result
    assert "transcript" in result
    assert result["steps"] == 1  # reset + 1 step
    assert result["info"]["done_reason"] == "done"


def test_run_episode_transcript(tmp_path):
    """Transcript should contain obs/action pairs."""
    snap = _snap_ok(SIMPLE_TREE)
    ad = FakeAD([
        _ScriptedStep(snapshot=snap),
        _ScriptedStep(snapshot=snap),
    ])
    rng = random.Random(42)

    result = run_episode(
        session=None,
        ad=ad,
        task_kind="write_save",
        rng=rng,
        policy=lambda obs: 'ACTION: {"op": "done"}',
        max_steps=15,
    )
    transcript = result["transcript"]
    assert len(transcript) >= 2
    assert transcript[0]["role"] == "obs"
    assert any(t["role"] == "action" for t in transcript)


def test_run_episode_missing_file(tmp_path):
    """No file created -> reward 0."""
    snap = _snap_ok(SIMPLE_TREE)
    ad = FakeAD([_ScriptedStep(snapshot=snap)])
    rng = random.Random(42)

    result = run_episode(
        session=None,
        ad=ad,
        task_kind="write_save",
        rng=rng,
        policy=lambda obs: 'ACTION: {"op": "done"}',
        max_steps=15,
    )
    assert result["reward"] == 0.0


def test_run_episode_max_steps(tmp_path):
    """Episode that hits max_steps."""
    snap = _snap_ok(SIMPLE_TREE)
    ad = FakeAD([
        _ScriptedStep(snapshot=snap),
        _ScriptedStep(act=_act_ok()),
        _ScriptedStep(snapshot=snap),
    ])
    rng = random.Random(42)

    result = run_episode(
        session=None,
        ad=ad,
        task_kind="write_save",
        rng=rng,
        policy=lambda obs: 'ACTION: {"op": "click", "id": 2}',
        max_steps=1,
    )
    assert result["info"]["done_reason"] == "max_steps"
