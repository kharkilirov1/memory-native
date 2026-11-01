# TASK 02 — fixes from a review on a real Linux desktop + runner (same rules: only research/agent-rl/, no git, pytest at the end)

## Bugs to fix in deskenv (add a regression test for each)
1. `env.py` passes `--app` to ref actions. Verified: `agent-desktop click [OPTIONS] <REF>` has NO `--app` (nor do
   double-click/type/set-value/clear/toggle/select). Only `press` takes `--app`. Pass `app=` ONLY for press.
2. `press` has no id, but `step()` does `assert parsed.id is not None` before `_perform_gui` -> AssertionError.
   Handle press without id.
3. `task.check()` is called with `Path.cwd()` (`_get_workdir`). Make `DesktopEnv(ad, task, workdir, max_steps=15)` take the
   workdir explicitly and pass it to check. Delete `_get_workdir`.
4. Reward must be computed at EVERY episode end (done, max_steps, too_many_malformed), not only on `done`.
   info["done_reason"] in {"done","max_steps","too_many_malformed"}.
5. `open_menus` stores full refs, but every snapshot has a NEW snapshot_id, so after the next snapshot the opened menu
   collapses again. Key opened menus by their NAME PATH (e.g. ("Edit","Convert")) and re-resolve on each render.
   Opening a submenu inside an opened menu must work. Add `close_menu{id}` op (env-only) as the inverse.
6. On `done`, `_build_obs(self._current_obs)` wraps the observation twice. Fix.
7. Observation should include the window title (`data.window.title`) — it shows the file name and a leading `*` when
   unsaved, which is the main progress signal in mousepad.

## New: `deskenv/session.py` — headless desktop session (Linux only; import must not fail on Windows)
`class DesktopSession(display=99, workdir_root=Path("/tmp/deskrl"))`:
- `start()`: if not running, start `Xvfb :N -screen 0 1280x800x24 -nolisten tcp`; a private `dbus-daemon --session --fork
  --print-address=1 --print-pid=1`; `/usr/libexec/at-spi-bus-launcher --launch-immediately`; `openbox` (a window manager
  is REQUIRED: without it agent-desktop fails with ACTION_NOT_SUPPORTED native_window_unproven). Env for children:
  DISPLAY, DBUS_SESSION_BUS_ADDRESS, NO_AT_BRIDGE=0, GTK_MODULES=gail:atk-bridge. `env()` returns that dict
  (also pass it to AgentDesktop).
- `launch(cmd, wait_app=name, timeout_s=10)`: Popen the app with that env, then poll `ad.snapshot(name)` until ok
  (APP_NOT_FOUND / native_window_unproven while starting -> keep polling). Return the Popen.
- `kill(proc)`: terminate, wait 3 s, kill. `stop()`: kill everything it started (keep pids).
- `new_workdir()`: fresh unique dir under workdir_root.
`deskenv/runner.py`: `run_episode(session, ad, task_kind, rng, policy, max_steps=15) -> dict` — new workdir, task setup,
launch app, env reset, loop `policy(obs) -> action_text` until done, kill app, return {reward, steps, info, transcript}.
`policy` is any callable (the RL loop will pass the LLM).

## New: `scripts/smoke_real.py` (run on Linux by me, not by you)
For each task kind, a SCRIPTED policy that solves it with the real mousepad UI using names, not fixed ids: it reads the
observation text, finds the line by role+name (e.g. button "Yes", textfield, menuitem "Save" under menu "File",
"Select All", "To Uppercase"), and emits the ACTION. Print reward per kind; expected 1.0 each. Add a `--kind` flag and
`--ad-bin` (path to agent-desktop). Mousepad facts: textfield has role "textfield" (one per window); menus File/Edit/
Search/View/Document/Help; Edit>Convert>To Uppercase; File>Save; ctrl+s also saves; a new file prompts an alert
"Question" with buttons No/Yes.

Reply with a short report: changed/new files, pytest summary line, open doubts.
