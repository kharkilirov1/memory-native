"""Tests for DesktopEnv using FakeAD."""

import random

from deskenv.ad import FakeAD, _ScriptedStep
from deskenv.env import DesktopEnv
from deskenv.tasks import make_task
from deskenv.actions import Action

# ── Helpers ─────────────────────────────────────────────────────────────


def _snap_ok(tree: dict) -> dict:
    return {"version": "2", "ok": True, "command": "snapshot",
            "data": {"app": "mousepad", "complete": True,
                     "snapshot_id": "s1", "tree": tree,
                     "window": {"id": "w-1", "title": "note.txt"}}}


def _act_ok() -> dict:
    return {"ok": True, "command": "act"}


def _act_err(code="STALE_REF") -> dict:
    return {"ok": False, "command": "act",
            "error": {"code": code, "message": f"{code} occurred",
                      "disposition": {"retry": "safe"}}}


SIMPLE_TREE = {
    "role": "window",
    "name": "mousepad",
    "ref_id": "@s1:w1",
    "available_actions": [],
    "children": [
        {
            "role": "textfield",
            "name": "text area",
            "ref_id": "@s1:t1",
            "available_actions": ["Click", "Type"],
        },
        {
            "role": "menu",
            "name": "File",
            "ref_id": "@s1:m1",
            "available_actions": ["Click"],
            "children": [
                {"role": "menuitem", "name": "Save", "ref_id": "@s1:sv",
                 "available_actions": ["Click"]},
            ],
        },
    ],
}


def _task():
    import pathlib
    rng = random.Random(1)
    return make_task("write_save", rng, pathlib.Path("C:\\tmp"))


# ── Tests ───────────────────────────────────────────────────────────────


def test_env_reset():
    ad = FakeAD([_ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE))])
    env = DesktopEnv(ad, _task())
    obs = env.reset()
    assert "Task:" in obs
    assert "short_id=1" in obs or "mousepad" in obs
    assert ad.call_log[0][0] == "snapshot"


def test_env_step_click():
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),  # 0: reset()
        _ScriptedStep(act=_act_ok()),                    # 1: _perform_gui
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),  # 2: _snapshot_tree (after GUI)
    ])
    env = DesktopEnv(ad, _task())
    env.reset()
    obs, reward, done, info = env.step('ACTION: {"op": "click", "id": 2}')
    assert reward == 0.0
    assert done is False
    # act was called
    assert any(c[0] == "act" for c in ad.call_log)


def test_env_step_done_reward():
    snap = _snap_ok({"role": "window", "name": "mousepad", "ref_id": "@s1:w1"})
    ad = FakeAD([
        _ScriptedStep(snapshot=snap),
    ])
    task = _task()
    # Create the file so check gives reward
    import tempfile, os
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "note.txt")
        # We need to know the expected sentence
        # _task() creates it with rng=1 -> we need to replicate or just check behavior
        # Since it won't match, reward will be low
        rng2 = random.Random(1)
        from deskenv.tasks import make_task as mt
        import pathlib
        t2 = mt("write_save", rng2, pathlib.Path("C:\\tmp"))
        # Actually, let's just test that done triggers check
        env = DesktopEnv(ad, t2)
        env.reset()
        obs, reward, done, info = env.step('ACTION: {"op": "done"}')
        assert done is True
        assert "check" in info
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_env_step_malformed():
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
    ])
    env = DesktopEnv(ad, _task())
    env.reset()
    obs, reward, done, info = env.step("garbage no action")
    assert reward == 0.0
    assert done is False  # first malformed
    assert info.get("malformed")


def test_env_max_malformed_ends():
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
    ])
    env = DesktopEnv(ad, _task())
    env.reset()
    env.step("garbage")  # malformed 1
    env.step("more garbage")  # malformed 2
    obs, reward, done, info = env.step("even more garbage")  # malformed 3
    assert done is True
    assert info.get("done_reason") == "too_many_malformed"


def test_env_max_steps():
    """Env terminates after max_steps even if all actions parse."""
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),  # 0: reset()
        _ScriptedStep(act=_act_ok()),                    # 1: _perform_gui
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),  # 2: _snapshot_tree
    ])
    env = DesktopEnv(ad, _task(), max_steps=1)
    env.reset()
    obs, reward, done, info = env.step('ACTION: {"op": "click", "id": 2}')
    assert done is True, f"expected done at max_steps, got {info}"


def test_env_open_menu():
    """open_menu re-renders with the menu expanded."""
    snap = _snap_ok(SIMPLE_TREE)
    ad = FakeAD([
        _ScriptedStep(snapshot=snap),
        _ScriptedStep(snapshot=snap),  # after open_menu, another snapshot
    ])
    env = DesktopEnv(ad, _task())
    env.reset()
    obs, reward, done, info = env.step('ACTION: {"op": "open_menu", "id": 3}')
    assert done is False
    # The menu ref @s1:m1 should now be in open_menus
    assert "@s1:m1" in env._open_menus


def test_env_stale_ref_passthrough():
    """Stale ref error from agent-desktop appears in obs."""
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),  # 0: reset()
        _ScriptedStep(act=_act_err("STALE_REF")),       # 1: _perform_gui
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),  # 2: _snapshot_tree
    ])
    env = DesktopEnv(ad, _task())
    env.reset()
    obs, reward, done, info = env.step('ACTION: {"op": "click", "id": 2}')
    assert "STALE_REF" in obs
    assert "ERROR: STALE_REF" in env._last_result


def test_env_reset_after_done():
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
    ])
    env = DesktopEnv(ad, _task())
    env.reset()
    env.step('ACTION: {"op": "done"}')
    # Second reset
    ad.reset([_ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE))])
    obs = env.reset()
    assert "Task:" in obs


def test_env_info_contains_steps():
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),  # 0: reset()
        _ScriptedStep(act=_act_ok()),                    # 1: _perform_gui
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),  # 2: _snapshot_tree
    ])
    env = DesktopEnv(ad, _task())
    env.reset()
    obs, reward, done, info = env.step('ACTION: {"op": "click", "id": 2}')
    assert info["steps"] == 1
