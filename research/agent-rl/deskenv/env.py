"""RL environment: DesktopEnv wraps AgentDesktop + a Task + action parsing."""

from __future__ import annotations

from typing import Any

from .actions import Action, parse_action
from .ad import AgentDesktop, FakeAD
from .render import render
from .tasks import Task


class DesktopEnv:
    """RL environment where an LLM agent controls a desktop app.

    Parameters
    ----------
    ad:
        Either a real ``AgentDesktop`` or a ``FakeAD`` for testing.
    task:
        The current ``Task`` to solve.
    max_steps:
        Maximum ``step()`` calls before forced termination.
    snapshot_timeout:
        Max seconds to wait for a snapshot subprocess.
    """

    def __init__(
        self,
        ad: AgentDesktop | FakeAD,
        task: Task,
        max_steps: int = 15,
    ):
        self._ad = ad
        self._task = task
        self.max_steps = max_steps

        # Internal state — cleared on reset
        self._current_obs: str = ""
        self._open_menus: tuple[str, ...] = ()
        self._last_result: str = ""
        self._steps = 0
        self._consecutive_malformed = 0
        self._done = False
        self._curr_refmap: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> str:
        """Set up the task's working directory and take an initial snapshot.

        Returns the initial observation text.
        """
        self._steps = 0
        self._consecutive_malformed = 0
        self._done = False
        self._open_menus = ()
        self._last_result = ""
        self._curr_refmap = {}

        # Snapshot the target app
        snap = self._ad.snapshot(self._task.app)
        tree = snap.get("data", {}).get("tree", snap)
        rendered, refmap = render(tree, open_menus=self._open_menus)
        self._curr_refmap = refmap
        self._current_obs = self._build_obs(rendered)
        return self._current_obs

    def step(self, action_text: str) -> tuple[str, float, bool, dict[str, Any]]:
        """Process one action from the agent.

        Returns
        -------
        obs_text, reward, done, info
        """
        if self._done:
            return self._current_obs, 0.0, True, {"info": "episode already done"}

        self._steps += 1
        info: dict[str, Any] = {"steps": self._steps}

        # ── parse ──────────────────────────────────────────────────────
        parsed = parse_action(action_text, self._curr_refmap)
        if isinstance(parsed, str):
            self._consecutive_malformed += 1
            self._last_result = f"ERROR: {parsed}"
            info["malformed"] = True
            info["n_malformed"] = self._consecutive_malformed

            # Check termination conditions
            done_now = self._check_done(info)
            rendered, _ = render(
                self._snapshot_tree(),
                open_menus=self._open_menus,
            )
            obs = self._build_obs(rendered)
            self._current_obs = obs
            return obs, 0.0, done_now, info

        # Reset malformed counter on successful parse
        self._consecutive_malformed = 0

        # ── env-only action: open_menu ─────────────────────────────────
        if parsed.op == "open_menu":
            assert parsed.id is not None  # guaranteed by parse_action
            self._open_menus = self._open_menus + (parsed.id,)
            snap_tree = self._snapshot_tree()
            rendered, refmap = render(snap_tree, open_menus=self._open_menus)
            self._curr_refmap = refmap
            self._last_result = f"ok: menu {parsed.id} expanded"
            obs = self._build_obs(rendered)
            self._current_obs = obs
            return obs, 0.0, False, info

        # ── done ────────────────────────────────────────────────────────
        if parsed.op == "done":
            reward, check_info = self._task.check(self._get_workdir())
            info["check"] = check_info
            self._done = True
            self._current_obs = self._build_obs(self._current_obs)
            return self._current_obs, reward, True, info

        # ── GUI actions ────────────────────────────────────────────────
        assert parsed.id is not None  # all remaining ops need an id
        gui_result = self._perform_gui(parsed)
        self._last_result = self._format_result(gui_result)

        # Take a fresh snapshot
        snap_tree = self._snapshot_tree()
        rendered, refmap = render(snap_tree, open_menus=self._open_menus)
        self._curr_refmap = refmap
        obs = self._build_obs(rendered)
        self._current_obs = obs

        done_now = self._check_done(info)
        return obs, 0.0, done_now, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot_tree(self) -> dict:
        snap = self._ad.snapshot(self._task.app)
        return snap.get("data", {}).get("tree", snap)

    def _perform_gui(self, act: Action) -> dict[str, Any]:
        """Call agent-desktop with the resolved command."""
        assert act.id is not None
        match act.op:
            case "click":
                return self._ad.act("click", act.id, app=self._task.app)
            case "double_click":
                return self._ad.act("double-click", act.id, app=self._task.app)
            case "type":
                return self._ad.act("type", act.id, act.text or "",
                                    app=self._task.app)
            case "set_value":
                return self._ad.act("set-value", act.id, act.value or "",
                                    app=self._task.app)
            case "clear":
                return self._ad.act("clear", act.id, app=self._task.app)
            case "toggle":
                return self._ad.act("toggle", act.id, app=self._task.app)
            case "select":
                return self._ad.act("select", act.id, app=self._task.app)
            case "press":
                return self._ad.act("press", act.keys or "",
                                    app=self._task.app)
            case _:
                return {"ok": False,
                        "error": {"code": "UNKNOWN_OP",
                                  "message": f"op {act.op} not mapped to CLI"}}

    @staticmethod
    def _format_result(result: dict[str, Any]) -> str:
        if result.get("ok"):
            return "ok"
        err = result.get("error", {})
        code = err.get("code", "UNKNOWN")
        msg = err.get("message", str(err))
        return f"ERROR: {code} — {msg}"

    def _build_obs(self, rendered: str) -> str:
        lines = [f"Task: {self._task.instruction}", "", rendered, ""]
        if self._last_result:
            lines.append(f"Last action: {self._last_result}")
            lines.append("")
        return "\n".join(lines)

    def _check_done(self, info: dict[str, Any]) -> bool:
        if self._steps >= self.max_steps:
            info["done_reason"] = "max_steps"
            self._done = True
            return True
        if self._consecutive_malformed >= 3:
            info["done_reason"] = "too_many_malformed"
            info["n_malformed"] = self._consecutive_malformed
            self._done = True
            return True
        return False

    @staticmethod
    def _get_workdir() -> Any:
        """Return a Path (or None) — not used in unit tests with FakeAD."""
        import pathlib
        return pathlib.Path.cwd()
