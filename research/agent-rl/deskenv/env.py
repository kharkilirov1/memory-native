"""RL environment: DesktopEnv wraps AgentDesktop + a Task + action parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .actions import Action, parse_action
from .ad import AgentDesktop, FakeAD
from .render import render, find_menu_name_path, resolve_menu_name_path
from .tasks import Task


class DesktopEnv:
    """RL environment where an LLM agent controls a desktop app.

    Parameters
    ----------
    ad:
        Either a real ``AgentDesktop`` or a ``FakeAD`` for testing.
    task:
        The current ``Task`` to solve.
    workdir:
        Working directory for this episode; passed to ``task.check()``.
    max_steps:
        Maximum ``step()`` calls before forced termination.
    """

    def __init__(
        self,
        ad: AgentDesktop | FakeAD,
        task: Task,
        workdir: str | Path,
        max_steps: int = 15,
    ):
        self._ad = ad
        self._task = task
        self._workdir = Path(workdir)
        self.max_steps = max_steps

        # Internal state — cleared on reset
        self._current_obs: str = ""
        self._open_menu_paths: list[tuple[str, ...]] = []
        self._last_result: str = ""
        self._steps = 0
        self._consecutive_malformed = 0
        self._done = False
        self._curr_refmap: dict[int, str] = {}
        self._window_title: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> str:
        """Take an initial snapshot and return the first observation."""
        self._steps = 0
        self._consecutive_malformed = 0
        self._done = False
        self._open_menu_paths = []
        self._last_result = ""
        self._curr_refmap = {}
        self._window_title = ""

        tree, title = self._take_snapshot()
        rendered, refmap = render(tree, open_menus=self._resolve_open_menus(tree))
        self._curr_refmap = refmap
        self._window_title = title
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
            return self._on_malformed(parsed, info)

        self._consecutive_malformed = 0

        # ── env-only: open_menu / close_menu ───────────────────────────
        # Menus are opened FOR REAL: items of a closed GTK menu are offscreen and agent-desktop
        # refuses to click them ("not actionable: visible"). open_menu = click on the menu (the
        # app shows it, its items become onscreen and render shows them); close_menu = escape.
        if parsed.op == "open_menu":
            parsed.op = "click"
        elif parsed.op == "close_menu":
            parsed.op, parsed.keys, parsed.id = "press", "escape", None

        # ── done ────────────────────────────────────────────────────────
        if parsed.op == "done":
            return self._on_done(info)

        # ── GUI actions ────────────────────────────────────────────────
        if parsed.op == "press":
            gui_result = self._perform_gui(parsed)
        else:
            assert parsed.id is not None, f"op {parsed.op} requires an id"
            gui_result = self._perform_gui(parsed)

        self._last_result = self._format_result(gui_result)

        # Take a fresh snapshot
        tree, title = self._take_snapshot()
        open_refs = self._resolve_open_menus(tree)
        rendered, refmap = render(tree, open_menus=open_refs)
        self._curr_refmap = refmap
        self._window_title = title
        self._current_obs = self._build_obs(rendered)

        done_now = self._check_done(info)
        if done_now:
            reward, check_info = self._task.check(self._workdir)
            info["check"] = check_info
            return self._current_obs, reward, done_now, info

        return self._current_obs, 0.0, False, info

    # ------------------------------------------------------------------
    # Handler helpers
    # ------------------------------------------------------------------

    def _on_malformed(
        self, error_msg: str, info: dict[str, Any]
    ) -> tuple[str, float, bool, dict[str, Any]]:
        self._consecutive_malformed += 1
        self._last_result = f"ERROR: {error_msg}"
        info["malformed"] = True
        info["n_malformed"] = self._consecutive_malformed

        tree, title = self._take_snapshot()
        open_refs = self._resolve_open_menus(tree)
        rendered, _ = render(tree, open_menus=open_refs)
        self._window_title = title
        obs = self._build_obs(rendered)
        self._current_obs = obs

        done_now = self._check_done(info)
        if done_now:
            reward, check_info = self._task.check(self._workdir)
            info["check"] = check_info
            return obs, reward, done_now, info

        return obs, 0.0, done_now, info

    def _handle_menu_op(
        self, act: Action, info: dict[str, Any]
    ) -> tuple[str, float, bool, dict[str, Any]]:
        assert act.id is not None
        # resolve the menu in the tree the agent SAW: a fresh snapshot has a new snapshot_id,
        # so act.id (a full ref of the previous snapshot) would not be found in it
        seen = getattr(self, "_last_tree", None) or {}
        path_seen = find_menu_name_path(seen, act.id)
        tree, title = self._take_snapshot()

        if act.op == "open_menu":
            path = path_seen
            if path is not None and path not in self._open_menu_paths:
                self._open_menu_paths.append(path)
            self._last_result = f"ok: menu expanded {path}"
        else:  # close_menu
            path = path_seen
            if path is not None and path in self._open_menu_paths:
                self._open_menu_paths.remove(path)
            self._last_result = f"ok: menu collapsed {path}"

        open_refs = self._resolve_open_menus(tree)
        rendered, refmap = render(tree, open_menus=open_refs)
        self._curr_refmap = refmap
        self._window_title = title
        obs = self._build_obs(rendered)
        self._current_obs = obs
        return obs, 0.0, False, info

    def _on_done(
        self, info: dict[str, Any]
    ) -> tuple[str, float, bool, dict[str, Any]]:
        reward, check_info = self._task.check(self._workdir)
        info["check"] = check_info
        info["done_reason"] = "done"
        self._done = True
        # _current_obs is already built; do not wrap again
        return self._current_obs, reward, True, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _take_snapshot(self) -> tuple[dict, str]:
        snap = self._ad.snapshot(self._task.app)
        data = snap.get("data", {})
        tree = data.get("tree", snap)
        title = data.get("window", {}).get("title", "")
        self._last_tree = tree   # the tree the agent's short ids refer to (refs are per-snapshot)
        return tree, title

    def _resolve_open_menus(self, tree: dict) -> tuple[str, ...]:
        """Convert stored name-paths to current snapshot ref_ids."""
        refs: list[str] = []
        for path in self._open_menu_paths:
            ref = resolve_menu_name_path(tree, path)
            if ref is not None:
                refs.append(ref)
        return tuple(refs)

    def _perform_gui(self, act: Action) -> dict[str, Any]:
        """Call agent-desktop with the resolved command.

        Only ``press`` receives ``--app``; ref-taking actions do NOT.
        """
        match act.op:
            case "click":
                return self._ad.act("click", act.id)  # type: ignore[arg-type]
            case "double_click":
                return self._ad.act("double-click", act.id)  # type: ignore[arg-type]
            case "type":
                return self._ad.act("type", act.id, act.text or "")  # type: ignore[arg-type]
            case "set_value":
                return self._ad.act("set-value", act.id, act.value or "")  # type: ignore[arg-type]
            case "clear":
                return self._ad.act("clear", act.id)  # type: ignore[arg-type]
            case "toggle":
                return self._ad.act("toggle", act.id)  # type: ignore[arg-type]
            case "select":
                return self._ad.act("select", act.id)  # type: ignore[arg-type]
            case "press":
                return self._ad.act("press", act.keys or "", app=self._task.app)
            case _:
                return {
                    "ok": False,
                    "error": {
                        "code": "UNKNOWN_OP",
                        "message": f"op {act.op} not mapped to CLI",
                    },
                }

    @staticmethod
    def _format_result(result: dict[str, Any]) -> str:
        if result.get("ok"):
            return "ok"
        err = result.get("error", {})
        code = err.get("code", "UNKNOWN")
        msg = err.get("message", str(err))
        return f"ERROR: {code} \u2014 {msg}"

    def _build_obs(self, rendered: str) -> str:
        lines = [f"Task: {self._task.instruction}", ""]
        if self._window_title:
            lines.append(f"Window: {self._window_title}")
            lines.append("")
        lines.append(rendered)
        lines.append("")
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
