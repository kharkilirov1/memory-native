"""Tests for DesktopEnv using FakeAD."""

import random
from pathlib import Path

import pytest
from deskenv.ad import FakeAD, _ScriptedStep
from deskenv.env import DesktopEnv
from deskenv.tasks import make_task


# ── Helpers ─────────────────────────────────────────────────────────────


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


def _act_err(code="STALE_REF") -> dict:
    return {
        "ok": False,
        "command": "act",
        "error": {
            "code": code,
            "message": f"{code} occurred",
            "disposition": {"retry": "safe"},
        },
    }


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
                {
                    "role": "menuitem",
                    "name": "Save",
                    "ref_id": "@s1:sv",
                    "available_actions": ["Click"],
                },
            ],
        },
    ],
}

MENU_TREE = {
    "role": "window",
    "name": "mousepad",
    "ref_id": "@s1:w1",
    "available_actions": [],
    "children": [
        {
            "role": "menu",
            "name": "File",
            "ref_id": "@s1:m1",
            "available_actions": ["Click"],
            "children": [
                {
                    "role": "menuitem",
                    "name": "Save",
                    "ref_id": "@s1:sv",
                    "available_actions": ["Click"],
                },
            ],
        },
        {
            "role": "menu",
            "name": "Edit",
            "ref_id": "@s1:m2",
            "available_actions": ["Click"],
            "children": [
                {
                    "role": "menuitem",
                    "name": "Select All",
                    "ref_id": "@s1:sa",
                    "available_actions": ["Click"],
                },
                {
                    "role": "menu",
                    "name": "Convert",
                    "ref_id": "@s1:m3",
                    "available_actions": ["Click"],
                    "children": [
                        {
                            "role": "menuitem",
                            "name": "To Uppercase",
                            "ref_id": "@s1:uc",
                            "available_actions": ["Click"],
                        },
                    ],
                },
            ],
        },
    ],
}


def _task(workdir: Path | None = None):
    if workdir is None:
        workdir = Path("C:\\tmp")
    rng = random.Random(1)
    return make_task("write_save", rng, workdir)


# ── Tests ───────────────────────────────────────────────────────────────


def test_env_reset():
    ad = FakeAD([_ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE))])
    env = DesktopEnv(ad, _task(), workdir=Path("C:\\tmp"))
    obs = env.reset()
    assert "Task:" in obs
    assert "Window:" in obs  # title included
    assert "[1]" in obs or "mousepad" in obs
    assert ad.call_log[0][0] == "snapshot"


def test_env_step_click(tmp_path):
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),  # 0: reset
        _ScriptedStep(act=_act_ok()),                    # 1: _perform_gui
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),  # 2: _take_snapshot
    ])
    env = DesktopEnv(ad, _task(tmp_path), workdir=tmp_path)
    env.reset()
    obs, reward, done, info = env.step('ACTION: {"op": "click", "id": 2}')
    assert reward == 0.0
    assert done is False
    assert any(c[0] == "act" for c in ad.call_log)


def test_env_step_done_reward(tmp_path):
    snap = _snap_ok({"role": "window", "name": "mousepad", "ref_id": "@s1:w1"})
    ad = FakeAD([_ScriptedStep(snapshot=snap)])

    rng = random.Random(1)
    task = make_task("write_save", rng, tmp_path)
    # Extract expected sentence from instruction
    import re
    m = re.search(r'type exactly:\s+"(.+?)"', task.instruction)
    assert m
    (tmp_path / "note.txt").write_text(m.group(1), encoding="utf-8")

    env = DesktopEnv(ad, task, workdir=tmp_path)
    env.reset()
    obs, reward, done, info = env.step('ACTION: {"op": "done"}')
    assert done is True
    assert reward == 1.0
    assert "check" in info
    assert info.get("done_reason") == "done"


def test_env_step_malformed(tmp_path):
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
    ])
    env = DesktopEnv(ad, _task(tmp_path), workdir=tmp_path)
    env.reset()
    obs, reward, done, info = env.step("garbage no action")
    assert reward == 0.0
    assert done is False
    assert info.get("malformed")


def test_env_max_malformed_ends(tmp_path):
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
    ])
    env = DesktopEnv(ad, _task(tmp_path), workdir=tmp_path)
    env.reset()
    env.step("garbage")
    env.step("more garbage")
    obs, reward, done, info = env.step("even more garbage")
    assert done is True
    assert info.get("done_reason") == "too_many_malformed"


def test_env_max_steps(tmp_path):
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(act=_act_ok()),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
    ])
    env = DesktopEnv(ad, _task(tmp_path), workdir=tmp_path, max_steps=1)
    env.reset()
    obs, reward, done, info = env.step('ACTION: {"op": "click", "id": 2}')
    assert done is True
    assert info.get("done_reason") == "max_steps"


def _menu_open_snap(base_tree: dict, menu_ref: str, title: str = "note.txt") -> dict:
    """Return a snapshot where the named menu is open: state=selected, items onscreen."""
    import copy
    tree = copy.deepcopy(base_tree)

    def _open_menu(node: dict):
        if node.get("ref_id") == menu_ref:
            node.setdefault("states", []).append("selected")
        for child in node.get("children", []):
            _open_menu(child)
        # Remove offscreen from all descendants of the open menu
        if "states" in node and "offscreen" in node["states"]:
            node["states"] = [s for s in node["states"] if s != "offscreen"]

    _open_menu(tree)
    return _snap_ok(tree, title=title)


def test_env_open_menu(tmp_path):
    """open_menu is now a REAL GUI click; menu items appear because GTK shows them."""
    snap_closed = _snap_ok(MENU_TREE)
    snap_open = _menu_open_snap(MENU_TREE, "@s1:m1")
    ad = FakeAD([
        _ScriptedStep(snapshot=snap_closed),  # 0: reset
        _ScriptedStep(act=_act_ok()),          # 1: click menu (real GUI)
        _ScriptedStep(snapshot=snap_open),     # 2: snapshot shows menu open
    ])
    env = DesktopEnv(ad, _task(tmp_path), workdir=tmp_path)
    env.reset()
    obs, reward, done, info = env.step('ACTION: {"op": "open_menu", "id": 2}')
    assert done is False
    # The menu items appear because _menu_is_open detects states=["selected"] + onscreen children
    assert "Save" in obs


def test_env_close_menu(tmp_path):
    """close_menu is now press escape; menu collapses in snapshot."""
    snap_closed = _snap_ok(MENU_TREE)
    snap_open = _menu_open_snap(MENU_TREE, "@s1:m1")
    ad = FakeAD([
        _ScriptedStep(snapshot=snap_closed),  # 0: reset
        _ScriptedStep(act=_act_ok()),          # 1: click File (open)
        _ScriptedStep(snapshot=snap_open),     # 2: snapshot shows menu open
        _ScriptedStep(act=_act_ok()),          # 3: press escape (close)
        _ScriptedStep(snapshot=snap_closed),   # 4: snapshot shows menu closed
    ])
    env = DesktopEnv(ad, _task(tmp_path), workdir=tmp_path)
    env.reset()
    env.step('ACTION: {"op": "open_menu", "id": 2}')
    # After press escape the menu collapses
    obs, reward, done, info = env.step('ACTION: {"op": "close_menu", "id": 2}')
    assert "Save" not in obs


def test_env_submenu_persists_across_snapshots(tmp_path):
    """Menu opened by real click stays open after snapshot_id changes (detected by GTK state)."""
    tree_b = MENU_TREE.copy()
    tree_b["ref_id"] = "@s2:w1"
    for c in tree_b["children"]:
        _bump_ids(c)

    snap_closed = _snap_ok(MENU_TREE, title="*note.txt")
    snap_open_a = _menu_open_snap(MENU_TREE, "@s1:m1", title="*note.txt")
    snap_open_b = _menu_open_snap(tree_b, "@s2:m1", title="note.txt")

    ad = FakeAD([
        _ScriptedStep(snapshot=snap_closed),  # 0: reset
        _ScriptedStep(act=_act_ok()),          # 1: click File
        _ScriptedStep(snapshot=snap_open_a),   # 2: File open (a ids)
        _ScriptedStep(act=_act_ok()),          # 3: press ctrl+s
        _ScriptedStep(snapshot=snap_open_b),   # 4: new snapshot, different ids, File still open
    ])
    env = DesktopEnv(ad, _task(tmp_path), workdir=tmp_path)
    env.reset()
    env.step('ACTION: {"op": "open_menu", "id": 2}')
    obs, reward, done, info = env.step('ACTION: {"op": "press", "keys": "ctrl+s"}')
    # After snapshot_id change, _menu_is_open detects the open state via GTK states
    assert "Save" in obs


def _bump_ids(node: dict):
    """Mutate node's ref_id to simulate a new snapshot."""
    old = node.get("ref_id", "")
    if old.startswith("@"):
        node["ref_id"] = "@s2" + old[3:]
    for child in node.get("children", []):
        _bump_ids(child)


def test_env_stale_ref_passthrough(tmp_path):
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(act=_act_err("STALE_REF")),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
    ])
    env = DesktopEnv(ad, _task(tmp_path), workdir=tmp_path)
    env.reset()
    obs, reward, done, info = env.step('ACTION: {"op": "click", "id": 2}')
    assert "STALE_REF" in obs


def test_env_reset_after_done(tmp_path):
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
    ])
    env = DesktopEnv(ad, _task(tmp_path), workdir=tmp_path)
    env.reset()
    env.step('ACTION: {"op": "done"}')
    ad.reset([_ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE))])
    obs = env.reset()
    assert "Task:" in obs


def test_env_info_contains_steps(tmp_path):
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(act=_act_ok()),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
    ])
    env = DesktopEnv(ad, _task(tmp_path), workdir=tmp_path)
    env.reset()
    obs, reward, done, info = env.step('ACTION: {"op": "click", "id": 2}')
    assert info["steps"] == 1


# ── Regression tests for TASK_02 bugs ───────────────────────────────────


def test_bug1_no_app_on_ref_actions(tmp_path):
    """Bug 1: --app must NOT be passed to ref-taking actions."""
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(act=_act_ok()),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
    ])
    env = DesktopEnv(ad, _task(tmp_path), workdir=tmp_path)
    env.reset()
    env.step('ACTION: {"op": "click", "id": 2}')
    # The act call for click (non-press) should have app=None
    for entry in ad.call_log:
        if entry[0] == "act" and entry[1][0] == "click":
            assert entry[2].get("app") is None, "click should NOT receive --app"


def test_bug2_press_no_id_assertion(tmp_path):
    """Bug 2: press has no id; must not hit assert parsed.id is not None."""
    snap = _snap_ok(SIMPLE_TREE)
    ad = FakeAD([
        _ScriptedStep(snapshot=snap),
        _ScriptedStep(act=_act_ok()),
        _ScriptedStep(snapshot=snap),
    ])
    env = DesktopEnv(ad, _task(tmp_path), workdir=tmp_path)
    env.reset()
    obs, reward, done, info = env.step('ACTION: {"op": "press", "keys": "ctrl+s"}')
    assert "ERROR" not in obs
    # press should have been called with app=mousepad (the only op that does)
    for entry in ad.call_log:
        if entry[0] == "act" and entry[1][0] == "press":
            assert entry[2].get("app") == "mousepad"


def test_bug3_explicit_workdir(tmp_path):
    """Bug 3: DesktopEnv must take workdir and pass it to task.check()."""
    task = _task(tmp_path)
    env = DesktopEnv(FakeAD([_ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE))]),
                     task, workdir=tmp_path)
    env.reset()
    assert env._workdir == tmp_path
    # check exists and uses the workdir
    assert not hasattr(env, "_get_workdir")  # method deleted


def test_bug4_reward_at_max_steps(tmp_path):
    """Bug 4: reward must be computed at max_steps termination."""
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(act=_act_ok()),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
    ])
    # Create file first so check returns non-zero
    task = _task(tmp_path)
    import re
    m = re.search(r'type exactly:\s+"(.+?)"', task.instruction)
    assert m
    (tmp_path / "note.txt").write_text(m.group(1), encoding="utf-8")

    env = DesktopEnv(ad, task, workdir=tmp_path, max_steps=1)
    env.reset()
    obs, reward, done, info = env.step('ACTION: {"op": "click", "id": 2}')
    assert done is True
    assert info.get("done_reason") == "max_steps"
    assert reward > 0, "reward should be >0 at max_steps termination"


def test_bug4_reward_at_malformed_limit(tmp_path):
    """Bug 4: reward must be computed at too_many_malformed."""
    ad = FakeAD([
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
        _ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE)),
    ])
    env = DesktopEnv(ad, _task(tmp_path), workdir=tmp_path)
    env.reset()
    env.step("garbage")
    env.step("garbage")
    obs, reward, done, info = env.step("garbage")
    assert done is True
    assert info.get("done_reason") == "too_many_malformed"
    # reward should be computed (even if 0 because file doesn't exist)
    assert "check" in info


def test_bug5_open_menu_by_name_path(tmp_path):
    """open_menu survives snapshot_id changes: _menu_is_open detects GTK state."""
    tree_a = MENU_TREE
    tree_b = MENU_TREE.copy()
    tree_b["ref_id"] = "@s2:w1"
    for c in tree_b["children"]:
        _bump_ids(c)
    snap_closed = _snap_ok(tree_a)
    snap_open_a = _menu_open_snap(tree_a, "@s1:m1")
    snap_open_b = _menu_open_snap(tree_b, "@s2:m1")

    ad = FakeAD([
        _ScriptedStep(snapshot=snap_closed),  # 0: reset
        _ScriptedStep(act=_act_ok()),          # 1: click File (open)
        _ScriptedStep(snapshot=snap_open_a),   # 2: File open in snap_a
        _ScriptedStep(act=_act_ok()),          # 3: click on textfield
        _ScriptedStep(snapshot=snap_open_b),   # 4: new snapshot (different ids), File still open
    ])
    env = DesktopEnv(ad, _task(tmp_path), workdir=tmp_path)
    env.reset()
    env.step('ACTION: {"op": "open_menu", "id": 2}')
    obs, reward, done, info = env.step('ACTION: {"op": "click", "id": 1}')
    # _menu_is_open detects the open GTK state regardless of snapshot ids
    assert "Save" in obs


def test_bug6_no_double_obs_wrap(tmp_path):
    """Bug 6: on done, observation must not be double-wrapped."""
    snap = _snap_ok({"role": "window", "name": "w", "ref_id": "@w1"})
    ad = FakeAD([_ScriptedStep(snapshot=snap)])
    task = _task(tmp_path)
    env = DesktopEnv(ad, task, workdir=tmp_path)
    env.reset()
    obs, reward, done, info = env.step('ACTION: {"op": "done"}')
    # obs should start with "Task:" exactly once, not twice
    assert obs.count("Task:") == 1


def test_bug7_window_title_in_obs(tmp_path):
    """Bug 7: observation must include the window title."""
    ad = FakeAD([_ScriptedStep(snapshot=_snap_ok(SIMPLE_TREE, title="*note.txt - Mousepad"))])
    env = DesktopEnv(ad, _task(tmp_path), workdir=tmp_path)
    obs = env.reset()
    assert "Window: *note.txt - Mousepad" in obs
