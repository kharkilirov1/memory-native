"""Tests for DesktopSession — import-safe on Windows."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Patch sys.platform to test Linux paths without being on Linux
if sys.platform != "linux":
    pass  # We test Linux-only methods via monkeypatch


def test_session_importable_on_windows():
    """The module must be importable everywhere (class defs available)."""
    from deskenv.session import DesktopSession, _check_linux

    # On non-Linux, _check_linux should raise
    if sys.platform != "linux":
        with pytest.raises(RuntimeError, match="Linux-only"):
            _check_linux()


def test_session_init_defaults():
    from deskenv.session import DesktopSession

    s = DesktopSession()
    assert s.display == 99
    assert s.workdir_root == Path("/tmp/deskrl")


def test_session_init_custom():
    from deskenv.session import DesktopSession

    s = DesktopSession(display=5, workdir_root="/tmp/test_rl", ad_bin="/usr/local/bin/agent-desktop")
    assert s.display == 5
    assert s.workdir_root == Path("/tmp/test_rl")


def test_session_env_has_display():
    from deskenv.session import DesktopSession

    s = DesktopSession(display=42)
    env = s.env()
    assert env["DISPLAY"] == ":42"
    assert env["NO_AT_BRIDGE"] == "0"
    assert "GTK_MODULES" in env
    assert "gail" in env["GTK_MODULES"]


@patch("deskenv.session.subprocess.Popen")
def test_session_start_calls_popen(mock_popen):
    """On Linux, start() should launch Xvfb, dbus, at-spi, openbox."""
    import os
    from deskenv.session import DesktopSession

    if os.name != "posix":
        pytest.skip("Linux-only test")

    # Mock the dbus-daemon output
    mock_proc = MagicMock()
    mock_proc.communicate.return_value = ("unix:abstract=/tmp/dbus-xxx\n12345\n", "")
    mock_popen.return_value = mock_proc

    s = DesktopSession(display=99)
    # Pretend we're on Linux
    with patch("deskenv.session.os.name", "posix"), \
         patch("deskenv.session.sys.platform", "linux"):
        s.start()
        assert s._running
        assert len(s._procs) == 4
        s.stop()


@patch("deskenv.session.DesktopSession._wait_for_app")
@patch("deskenv.session.subprocess.Popen")
def test_session_launch_with_wait(mock_popen, mock_wait):
    from deskenv.session import DesktopSession

    s = DesktopSession()
    proc = s.launch(["mousepad", "/tmp/test.txt"], wait_app="mousepad")
    assert proc is not None
    mock_wait.assert_called_once_with("mousepad", 10)


def test_session_new_workdir(tmp_path):
    from deskenv.session import DesktopSession

    s = DesktopSession(workdir_root=str(tmp_path))
    d = s.new_workdir()
    assert d.parent == tmp_path
    assert d.exists()


@patch("deskenv.session.subprocess.Popen")
def test_session_kill_proc(mock_popen):
    from deskenv.session import DesktopSession

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running
    DesktopSession._kill_proc(mock_proc)
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called_once()
