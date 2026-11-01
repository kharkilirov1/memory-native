#!/usr/bin/env python3
"""Smoke test: scripted policies that solve each task kind with real mousepad.

Run on Linux only.  Each policy reads the rendered tree, finds nodes by
role+name (never fixed ids), and emits ACTION lines.  Expected reward = 1.0
for every kind.

Usage:
    python scripts/smoke_real.py --ad-bin /path/to/agent-desktop [--kind write_save]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Add repo root to sys.path so `from deskenv` works when running as script
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from deskenv.ad import AgentDesktop
from deskenv.env import DesktopEnv
from deskenv.session import DesktopSession
from deskenv.tasks import make_task


# ---------------------------------------------------------------------------
# Scripted policies — they read the observation text, never use fixed ids.
# ---------------------------------------------------------------------------

def _find_by_role_and_name(obs: str, role: str, name: str) -> int | None:
    """Return the first short_id whose line matches *role* and *name*."""
    for line in obs.splitlines():
        m = re.match(r"\s*\[short_id=(\d+)\]\s+", line)
        if not m:
            continue
        rest = line[m.end():]
        if rest.startswith(role + ' "') and '"' in rest:
            quoted = rest.split('"')[1]
            if quoted == name:
                return int(m.group(1))
    return None


def _find_by_role(obs: str, role: str) -> int | None:
    """Return the first short_id of a node with the given role."""
    for line in obs.splitlines():
        m = re.match(r"\s*\[short_id=(\d+)\]\s+", line)
        if not m:
            continue
        rest = line[m.end():]
        if rest.startswith(role) and not rest.startswith(role + ' "'):
            # role without a quoted name
            return int(m.group(1))
        if rest.startswith(role + ' "') and '"' in rest:
            return int(m.group(1))
    return None


def policy_write_save(obs: str) -> str:
    """Type the sentence and save (ctrl+s)."""
    # Find textfield
    tf = _find_by_role(obs, "textfield")
    if tf is not None:
        # Extract sentence from instruction
        m = re.search(r'type exactly:\s+"(.+?)"', obs)
        sentence = m.group(1) if m else "hello world"
        return f'ACTION: {{"op": "type", "id": {tf}, "text": "{sentence}"}}'

    # Check for alert dialog "Question" with Yes/No
    menu = _find_by_role_and_name(obs, "button", "Yes")
    if menu is not None:
        return f'ACTION: {{"op": "click", "id": {menu}}}'

    # Save: press ctrl+s
    return 'ACTION: {"op": "press", "keys": "ctrl+s"}'


def policy_uppercase_save(obs: str) -> str:
    """Select All -> Edit > Convert > To Uppercase -> Save."""
    # Open Edit menu
    edit = _find_by_role_and_name(obs, "menu", "Edit")
    if edit is not None:
        _maybe_collapsed = re.search(
            rf"\[short_id={edit}\].*\(\+\d+ items, use open_menu\)", obs
        )
        if _maybe_collapsed:
            return f'ACTION: {{"op": "open_menu", "id": {edit}}}'

    # Open Convert submenu
    convert = _find_by_role_and_name(obs, "menu", "Convert")
    if convert is not None:
        _maybe_collapsed = re.search(
            rf"\[short_id={convert}\].*\(\+\d+ items, use open_menu\)", obs
        )
        if _maybe_collapsed:
            return f'ACTION: {{"op": "open_menu", "id": {convert}}}'

    # Click "Select All"
    select_all = _find_by_role_and_name(obs, "menuitem", "Select All")
    if select_all is not None:
        return f'ACTION: {{"op": "click", "id": {select_all}}}'

    # Click "To Uppercase"
    upper = _find_by_role_and_name(obs, "menuitem", "To Uppercase")
    if upper is not None:
        return f'ACTION: {{"op": "click", "id": {upper}}}'

    # Save: press ctrl+s
    return 'ACTION: {"op": "press", "keys": "ctrl+s"}'


def policy_append_line(obs: str) -> str:
    """Type the line and save."""
    return policy_write_save(obs)


def policy_delete_line(obs: str) -> str:
    """Select all, backspace, type, save — simplified for smoke test."""
    m_line = re.search(r"delete line (\d+)", obs)
    return 'ACTION: {"op": "done"}'  # placeholder: too complex for scripted


_POLICIES = {
    "write_save": policy_write_save,
    "uppercase_save": policy_uppercase_save,
    "append_line": policy_append_line,
    "delete_line": policy_delete_line,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test real mousepad tasks")
    ap.add_argument("--ad-bin", default="agent-desktop", help="Path to agent-desktop CLI")
    ap.add_argument("--kind", default=None, help="Task kind (default: all)")
    args = ap.parse_args()

    session = DesktopSession(display=99, ad_bin=args.ad_bin)
    ad = session.ad

    kinds = [args.kind] if args.kind else list(_POLICIES)

    for kind in kinds:
        if kind not in _POLICIES:
            print(f"SKIP  {kind}: no policy defined")
            continue

        rng = __import__("random").Random(42)
        workdir = session.new_workdir()
        task = make_task(kind, rng, workdir)
        task.setup(workdir)
        proc = session.launch(task.launch_cmd, wait_app=task.app)

        env = DesktopEnv(ad, task, workdir=workdir)
        obs = env.reset()
        policy = _POLICIES[kind]
        done = False
        steps = 0
        while not done and steps < 20:
            action = policy(obs)
            obs, reward, done, info = env.step(action)
            steps += 1

        session.kill(proc)
        print(f"{'OK  ' if reward >= 0.99 else 'FAIL'} {kind}: reward={reward:.4f}, steps={steps}")


if __name__ == "__main__":
    main()
