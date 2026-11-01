"""AgentDesktop CLI wrapper and its fake counterpart for testing."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any


class ADError(Exception):
    """Raised on subprocess timeout or non-JSON output from agent-desktop."""

    def __init__(self, message: str, cmd: list[str] | None = None):
        self.cmd = cmd
        super().__init__(message)


@dataclass
class _ScriptedStep:
    """One scripted reply for the FakeAD state machine."""
    snapshot: dict | None = None
    act: dict | None = None
    error: str | None = None  # raises ADError with this message


class AgentDesktop:
    """Wrapper around the `agent-desktop` CLI.

    All calls go through subprocess (list args, no shell).  Returns the JSON
    envelope as a dict even when ``ok`` is ``false``.  Raises ``ADError`` only
    on timeout or unparseable output.
    """

    def __init__(self, bin_path: str, env: dict[str, str] | None = None,
                 timeout_s: int = 15):
        self.bin_path = bin_path
        self._env = env
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(self, app: str) -> dict[str, Any]:
        """``agent-desktop snapshot --app <app> -i --compact`` -> parsed JSON."""
        cmd = [self.bin_path, "snapshot", "--app", app, "-i", "--compact"]
        stdout = self._run(cmd)
        return self._parse(stdout, cmd)

    def act(self, cmd_name: str, *args: str, app: str | None = None) -> dict[str, Any]:
        """``agent-desktop <cmd> <args> [--app <app>]`` -> parsed JSON."""
        cmd = [self.bin_path, cmd_name, *args]
        if app is not None:
            cmd.extend(["--app", app])
        stdout = self._run(cmd)
        return self._parse(stdout, cmd)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, cmd: list[str]) -> str:
        try:
            cp = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=self._env,
            )
        except subprocess.TimeoutExpired:
            raise ADError(
                f"agent-desktop timed out after {self.timeout_s}s",
                cmd=cmd,
            )
        if cp.returncode != 0:
            raise ADError(
                f"agent-desktop exited with code {cp.returncode}: "
                f"{cp.stderr.strip() or cp.stdout.strip()}",
                cmd=cmd,
            )
        return cp.stdout.strip()

    @staticmethod
    def _parse(stdout: str, cmd: list[str]) -> dict[str, Any]:
        if not stdout:
            raise ADError("empty output from agent-desktop", cmd=cmd)
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ADError(
                f"non-JSON output from agent-desktop: {exc}",
                cmd=cmd,
            ) from exc


# ------------------------------------------------------------------
# Fake backend  (state-machine driven)
# ------------------------------------------------------------------


class FakeAD:
    """Scripted fake of ``AgentDesktop`` for unit tests.

    Behaviour is driven by a list of ``_ScriptedStep`` s fed via
    ``__init__``.  Each call to ``snapshot`` or ``act`` consumes the
    *next* step and returns its payload or raises its error.
    """

    def __init__(self, steps: list[_ScriptedStep] | None = None):
        self.steps = list(steps or [])
        self._pos = 0
        self.call_log: list[tuple[str, tuple, dict]] = []  # (method, args, kwargs)

    def snapshot(self, app: str) -> dict[str, Any]:
        self.call_log.append(("snapshot", (app,), {}))
        return self._next("snapshot")

    def act(self, cmd_name: str, *args: str, app: str | None = None) -> dict[str, Any]:
        self.call_log.append(("act", (cmd_name, *args), {"app": app}))
        return self._next("act")

    def _next(self, caller: str) -> dict[str, Any]:
        if self._pos >= len(self.steps):
            raise ADError(f"FakeAD: no more scripted steps for {caller}")
        step = self.steps[self._pos]
        self._pos += 1
        if step.error is not None:
            raise ADError(step.error)
        if caller == "snapshot" and step.snapshot is not None:
            return step.snapshot
        if caller == "act" and step.act is not None:
            return step.act
        raise ADError(f"FakeAD: step {self._pos - 1} has no response for {caller}")

    def reset(self, steps: list[_ScriptedStep]) -> None:
        """Re-arm the state machine with a new script."""
        self.steps = list(steps)
        self._pos = 0
        self.call_log.clear()
