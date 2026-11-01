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
import json
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


def _act(op: str, **kw) -> str:
    return "ACTION: " + json.dumps({"op": op, **kw})


class _PlanPolicy(_BasePolicy):
    """Runs a fixed plan of steps; each step is (op, finder, extra) resolved against the CURRENT observation
    (ids change every snapshot), then 'done'. finder: ("role", name|None) or None for ops without an id."""

    plan: list = []

    def _act(self, obs: str) -> str:
        i = self._step - 1
        if i >= len(self.plan):
            return _act("done")
        op, finder, extra = self.plan[i]
        if finder is None:
            return _act(op, **extra)
        role, name = finder
        k = _find_by_role(obs, role) if name is None else _find_by_role_and_name(obs, role, name)
        if k is None:
            return _act("done")          # element missing: end the episode (reward tells the story)
        return _act(op, id=k, **extra)


def _content(task: Task) -> list[str]:
    path = Path(task.launch_cmd[1])
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


class _WriteSavePolicy(_PlanPolicy):
    """Type the sentence, ctrl+s, done."""

    def __init__(self, task: Task):
        super().__init__(task)
        m = re.search(r'type exactly:\s+"(.+?)"', task.instruction)
        self.plan = [("type", ("textfield", None), {"text": m.group(1) if m else ""}),
                     ("press", None, {"keys": "ctrl+s"})]


class _UppercaseSavePolicy(_PlanPolicy):
    """Menu path (exercises open_menu): Edit > Select All, Edit > Convert > To Uppercase, ctrl+s."""

    def __init__(self, task: Task):
        super().__init__(task)
        self.plan = [("open_menu", ("menu", "Edit"), {}),
                     ("click", ("menuitem", "Select All"), {}),
                     ("open_menu", ("menu", "Convert"), {}),
                     ("click", ("menuitem", "To Uppercase"), {}),
                     # after menu use the menubar keeps focus and swallows ctrl+s: save via the menu
                     ("open_menu", ("menu", "File"), {}),
                     ("click", ("menuitem", "Save"), {})]


class _AppendLinePolicy(_PlanPolicy):
    """set_value the full target content (read ONCE at construction), ctrl+s, done."""

    def __init__(self, task: Task):
        super().__init__(task)
        m = re.search(r"(?:line|text)[^:]*:\s*\"?(.+?)\"?\s*$", task.instruction)
        extra = m.group(1) if m else ""
        target = "\n".join(_content(task) + [extra]) + "\n"
        self.plan = [("set_value", ("textfield", None), {"value": target}),
                     ("press", None, {"keys": "ctrl+s"})]


class _DeleteLinePolicy(_PlanPolicy):
    """set_value the content minus line k (read ONCE), ctrl+s, done."""

    def __init__(self, task: Task):
        super().__init__(task)
        m = re.search(r"[Dd]elete line (?:number )?(\d+)", task.instruction)
        k = int(m.group(1)) if m else 1
        lines = [l for i, l in enumerate(_content(task), 1) if i != k]
        self.plan = [("set_value", ("textfield", None), {"value": "\n".join(lines) + "\n"}),
                     ("press", None, {"keys": "ctrl+s"})]


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
