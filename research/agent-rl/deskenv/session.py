"""Headless desktop session for running RL episodes on Linux.

Import-safe on Windows: the class definition is always available, but the
``start()`` method raises ``RuntimeError`` when the platform is not Linux.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .ad import AgentDesktop


def _check_linux() -> None:
    if os.name != "posix" or sys.platform != "linux":
        raise RuntimeError(
            "DesktopSession is Linux-only; cannot start background services "
            f"on {sys.platform}"
        )


class DesktopSession:
    """A headless Linux desktop (Xvfb + dbus + at-spi + openbox).

    Parameters
    ----------
    display:
        X display number (``:N``).
    workdir_root:
        Root directory for per-episode working directories.
    ad_bin:
        Path to the ``agent-desktop`` CLI binary.
    """

    def __init__(
        self,
        display: int = 99,
        workdir_root: str | Path = Path("/tmp/deskrl"),
        ad_bin: str = "agent-desktop",
    ):
        self.display = display
        self.workdir_root = Path(workdir_root)
        self._dbus_addr: str | None = None
        self._dbus_pid: int | None = None
        self.ad = AgentDesktop(ad_bin, env=self.env())

        self._procs: list[subprocess.Popen] = []
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background services if not already running.

        Raises ``RuntimeError`` on non-Linux platforms.
        """
        if self._running:
            return
        _check_linux()

        self.workdir_root.mkdir(parents=True, exist_ok=True)

        # 1. Xvfb
        xvfb = subprocess.Popen(
            [
                "Xvfb", f":{self.display}",
                "-screen", "0", "1280x800x24",
                "-nolisten", "tcp",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._procs.append(xvfb)

        # 2. dbus-daemon (session bus)
        dbus = subprocess.Popen(
            [
                "dbus-daemon", "--session", "--fork",
                "--print-address=1", "--print-pid=1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        stdout, _ = dbus.communicate(timeout=5)
        # --fork: the printed address/pid belong to the daemon child, not to `dbus`
        lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
        if len(lines) < 2:
            raise RuntimeError(f"dbus-daemon printed no address/pid: {stdout!r}")
        self._dbus_addr, self._dbus_pid = lines[0], int(lines[1])
        self.ad._env = self.env()  # agent-desktop must see the private session bus

        # 3. at-spi-bus-launcher
        atspi = subprocess.Popen(
            ["/usr/libexec/at-spi-bus-launcher", "--launch-immediately"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=self.env(),
        )
        self._procs.append(atspi)

        # 4. openbox (window manager — REQUIRED)
        openbox = subprocess.Popen(
            ["openbox"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=self.env(),
        )
        self._procs.append(openbox)

        self._running = True

    def stop(self) -> None:
        """Kill everything this session started."""
        for proc in reversed(self._procs):
            self._kill_proc(proc)
        self._procs.clear()
        if self._dbus_pid:
            try:
                os.kill(self._dbus_pid, 15)
            except ProcessLookupError:
                pass
            self._dbus_pid = self._dbus_addr = None
        self._running = False

    # ------------------------------------------------------------------
    # Per-episode helpers
    # ------------------------------------------------------------------

    def env(self) -> dict[str, str]:
        """Environment variables for children (DISPLAY, dbus, a11y)."""
        base = dict(os.environ)
        base.update({
            "DISPLAY": f":{self.display}",
            "NO_AT_BRIDGE": "0",
            "GTK_MODULES": "gail:atk-bridge",
            # in-memory GSettings: apps start from defaults every launch, so mousepad never
            # shows "previous session did not end normally - restore?" after we kill it
            "GSETTINGS_BACKEND": "memory",
            # per-session app data/config/cache (autosave files, recent lists) - no cross-run state
            "XDG_DATA_HOME": str(self.workdir_root / "xdg" / "data"),
            "XDG_CONFIG_HOME": str(self.workdir_root / "xdg" / "config"),
            "XDG_CACHE_HOME": str(self.workdir_root / "xdg" / "cache"),
        })
        if self._dbus_addr:
            base["DBUS_SESSION_BUS_ADDRESS"] = self._dbus_addr
        return base

    def launch(
        self,
        cmd: list[str],
        wait_app: str | None = None,
        timeout_s: int = 10,
        workdir: str | Path | None = None,
    ) -> subprocess.Popen:
        """Start an app with the session environment and optionally wait for it.

        If *workdir* is given, the app's stdout and stderr are redirected to
        ``<workdir>/app.log`` (inherited pipes keep callers hanging on Linux).
        Returns the ``Popen`` handle.
        """
        stdout = stderr = None
        if workdir is not None:
            log_path = Path(workdir) / "app.log"
            stdout = stderr = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, env=self.env(), stdout=stdout, stderr=stderr)
        if wait_app is not None:
            self._wait_for_app(wait_app, timeout_s)
        return proc

    def kill(self, proc: subprocess.Popen) -> None:
        """Terminate a process, wait briefly, then kill."""
        self._kill_proc(proc)

    def new_workdir(self) -> Path:
        """Create and return a fresh unique directory under ``workdir_root``."""
        self.workdir_root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(dir=str(self.workdir_root)))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _wait_for_app(self, app_name: str, timeout_s: int) -> None:
        deadline = time.monotonic() + timeout_s
        last_err = ""
        while time.monotonic() < deadline:
            try:
                snap = self.ad.snapshot(app_name)
                if snap.get("ok"):
                    return
                err = snap.get("error", {})
                code = err.get("code", "")
                if code not in ("APP_NOT_FOUND", "native_window_unproven"):
                    return
                last_err = code
            except Exception as exc:
                last_err = str(exc)
            time.sleep(0.3)
        raise TimeoutError(
            f"app {app_name!r} did not become available in {timeout_s}s "
            f"(last: {last_err})"
        )

    @staticmethod
    def _kill_proc(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2)
            except Exception:
                pass
