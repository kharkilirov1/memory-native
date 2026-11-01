#!/usr/bin/env python3
"""Smoke test: scripted policies that solve each task kind with real mousepad.

Run on Linux only.  Each policy is a stateful class (step counter) that reads
the rendered tree, finds nodes by role+name, and emits ACTION lines.
Expected reward = 1.0 for every kind.

Usage:
    python scripts/smoke_real.py --ad-bin /path/to/agent-desktop [--kind write_save] [--verbose]
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
from deskenv.tasks import Task, make_task


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

def _find_by_role_and_name(obs: str, role: str, name: str) -> int | None:
    """Return the first ``[k]`` whose line matches *role* and *name*."""
    for line in obs.splitlines():
        m = re.match(r"\s*\[(\d+)\]\s+", line)
        if not m:
            continue
        rest = line[m.end():]
        if rest.startswith(role + ' "') and '"' in rest:
            quoted = rest.split('"')[1]
            if quoted == name:
                return int(m.group(1))
    return None


def _find_by_role(obs: str, role: str) -> int | None:
    """Return the first ``[k]`` with the given *role*."""
    for line in obs.splitlines():
        m = re.match(r"\s*\[(\d+)\]\s+", line)
        if not m:
            continue
        rest = line[m.end():]
        if rest.startswith(role + ' "') and '"' in rest:
            return int(m.group(1))
        if rest.startswith(role) and not rest.startswith(role + ' "'):
            return int(m.group(1))
    return None


def _is_collapsed(obs: str, k: int) -> bool:
    return bool(re.search(rf"\[{k}\].*\(+\d+ items, use open_menu\)", obs))


# ---------------------------------------------------------------------------
# Stateful policies
# ---------------------------------------------------------------------------

class _BasePolicy:
    """Base for stateful scripted policies."""

    def __init__(self, task: Task):
        self.task = task
        self._step = 0

    def __call__(self, obs: str) -> str:
        self._step += 1
        return self._act(obs)

    def _act(self, obs: str) -> str:
        raise NotImplementedError


class _WriteSavePolicy(_BasePolicy):
    """Type the sentence, ctrl+s, done."""

    def _act(self, obs: str) -> str:
        # 1: find textfield and type
        tf = _find_by_role(obs, "textfield")
        if tf is not None:
            m = re.search(r'type exactly:\s+"(.+?)"', self.task.instruction)
            sentence = m.group(1) if m else "hello world"
            return f'ACTION: {{"op": "type", "id": {tf}, "text": "{sentence}"}}'
        # 2: save
        return 'ACTION: {"op": "press", "keys": "ctrl+s"}'


class _UppercaseSavePolicy(_BasePolicy):
    """Menu path: Edit > Select All → Edit > Convert > To Uppercase → save.

    Fallback: ``set_value`` with the uppercased text.
    """

    def _act(self, obs: str) -> str:
        # Open Edit menu if collapsed
        edit = _find_by_role_and_name(obs, "menu", "Edit")
        if edit is not None and _is_collapsed(obs, edit):
            return f'ACTION: {{"op": "open_menu", "id": {edit}}}'

        # Open Convert submenu if collapsed
        convert = _find_by_role_and_name(obs, "menu", "Convert")
        if convert is not None and _is_collapsed(obs, convert):
            return f'ACTION: {{"op": "open_menu", "id": {convert}}}'

        # Click "Select All"
        select_all = _find_by_role_and_name(obs, "menuitem", "Select All")
        if select_all is not None:
            return f'ACTION: {{"op": "click", "id": {select_all}}}'

        # Click "To Uppercase"
        upper = _find_by_role_and_name(obs, "menuitem", "To Uppercase")
        if upper is not None:
            return f'ACTION: {{"op": "click", "id": {upper}}}'

        # Save
        return 'ACTION: {"op": "press", "keys": "ctrl+s"}'


class _AppendLinePolicy(_BasePolicy):
    """set_value the full content, ctrl+s, done."""

    def _act(self, obs: str) -> str:
        # Read the original file content from the workdir
        import os
        workdir = os.path.dirname(self.task.launch_cmd[1]) if len(self.task.launch_cmd) > 1 else "/tmp"
        path = Path(workdir) / "note.txt"
        if path.exists():
            orig = path.read_text(encoding="utf-8")
        else:
            orig = ""
        # Append the extra line from the instruction
        m = re.search(r"Append the following line at the end and save:\s*(.+)$", self.task.instruction)
        extra = m.group(1) if m else ""
        target = orig.rstrip("\n") + "\n" + extra + "\n" if extra else orig

        tf = _find_by_role(obs, "textfield")
        if tf is not None:
            return f'ACTION: {{"op": "set_value", "id": {tf}, "value": "{target}"}}'
        return 'ACTION: {"op": "press", "keys": "ctrl+s"}'


class _DeleteLinePolicy(_BasePolicy):
    """set_value the file content minus the deleted line, ctrl+s, done."""

    def _act(self, obs: str) -> str:
        import os
        workdir = os.path.dirname(self.task.launch_cmd[1]) if len(self.task.launch_cmd) > 1 else "/tmp"
        path = Path(workdir) / "note.txt"
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
        else:
            lines = []
        # Parse which line to delete from instruction
        m = re.search(r"Delete line (\d+)", self.task.instruction)
        k = int(m.group(1)) if m else 1
        target_lines = [l for i, l in enumerate(lines, 1) if i != k]
        target = "\n".join(target_lines) + ("\n" if lines else "")

        tf = _find_by_role(obs, "textfield")
        if tf is not None:
            return f'ACTION: {{"op": "set_value", "id": {tf}, "value": "{target}"}}'
        return 'ACTION: {"op": "press", "keys": "ctrl+s"}'


_POLICY_CLASSES = {
    "write_save": _WriteSavePolicy,
    "uppercase_save": _UppercaseSavePolicy,
    "append_line": _AppendLinePolicy,
    "delete_line": _DeleteLinePolicy,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test real mousepad tasks")
    ap.add_argument("--ad-bin", default="agent-desktop", help="Path to agent-desktop CLI")
    ap.add_argument("--kind", default=None, help="Task kind (default: all)")
    ap.add_argument("--verbose", action="store_true", help="Print every obs and action")
    args = ap.parse_args()

    session = DesktopSession(display=99, ad_bin=args.ad_bin)
    session.start()
    ad = session.ad

    kinds = [args.kind] if args.kind else list(_POLICY_CLASSES)
    overall_ok = True

    for kind in kinds:
        if kind not in _POLICY_CLASSES:
            print(f"SKIP  {kind}: no policy defined")
            continue

        rng = __import__("random").Random(42)
        workdir = session.new_workdir()
        task = make_task(kind, rng, workdir)
        task.setup(workdir)

        proc = None
        reward = 0.0
        info = {}
        try:
            proc = session.launch(task.launch_cmd, wait_app=task.app,
                                  workdir=workdir)
            env = DesktopEnv(ad, task, workdir=workdir)
            obs = env.reset()
            if args.verbose:
                print(f"--- {kind} step 0 ---")
                print(obs)
                print()

            policy = _POLICY_CLASSES[kind](task)
            done = False
            steps = 0
            while not done and steps < 20:
                action = policy(obs)
                obs, reward, done, info = env.step(action)
                steps += 1
                if args.verbose:
                    print(f"--- {kind} step {steps} ---")
                    print(f"ACTION: {action}")
                    print(obs)
                    print()
        finally:
            if proc is not None:
                session.kill(proc)

        ok = reward >= 0.99
        overall_ok = overall_ok and ok
        print(f"{'OK  ' if ok else 'FAIL'} {kind}: reward={reward:.4f}, steps={steps}")

    if not overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
