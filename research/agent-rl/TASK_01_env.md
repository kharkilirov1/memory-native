# TASK 01 — desktop RL environment (code only, Python 3.10+, no new deps beyond stdlib + pytest)

Goal: an RL environment where an LLM operates Linux desktop apps through the accessibility tree
via the `agent-desktop` CLI (Vercel Labs, v1.0.2) — NO screenshots. Will run on Linux (Xvfb +
openbox + at-spi2); you are on Windows, so everything must be unit-testable with a FAKE backend.

Put all code in `research/agent-rl/deskenv/` (package) + tests in `research/agent-rl/tests/`.
Do NOT touch anything outside `research/agent-rl/`. No git commit/push. Run only `pytest research/agent-rl/tests -q`.

## Facts about agent-desktop (verified on Linux)
- CLI: `agent-desktop snapshot --app mousepad -i --compact` -> one JSON line:
  `{"version":"2.5","ok":true,"command":"snapshot","data":{"app":"mousepad","complete":true,"ref_count":271,
   "snapshot_id":"s38i6wd7dxtgr7","tree":{...},"window":{"id":"w-...","title":"*/tmp/work/note.txt - Mousepad"}}}`
- tree node: `{"role":"menu","name":"File","ref_id":"@s38i6wd7dxtgr7:e2","available_actions":["Click","Select"],
   "states":["focused"],"children":[...]}` (ref_id/name/actions/states/children all optional).
  Roles seen: window, group, menu, menuitem, toolbar, button, tablist, tab, scrollarea, textfield, scrollbar, alert.
  Menu items are present in the tree even when the menu is closed (mousepad: 229 menuitems of 271 refs).
- error: `{"ok":false,"command":"snapshot","error":{"code":"ACTION_NOT_SUPPORTED","message":"...","details":{"kind":"..."},
   "disposition":{"delivery":"not_delivered","retry":"safe"}}}`; codes include STALE_REF, AMBIGUOUS_TARGET, INVALID_ARGS.
- actions (ref = full `@<snapshot_id>:eN`): `click REF`, `double-click REF`, `type REF TEXT`, `set-value REF VALUE`,
  `clear REF`, `focus REF`, `toggle REF`, `check REF`, `uncheck REF`, `select REF`, `expand REF`, `collapse REF`,
  `scroll REF`, `press COMBO --app APP` (COMBO like `ctrl+s`, `return`, `escape`). Each prints one JSON envelope.
- A modal dialog appears as a separate window (role "alert", name "Question"); snapshot of the app returns it.

## Modules
1. `ad.py` — `class AgentDesktop(bin_path, env=None, timeout_s=15)`: `snapshot(app) -> dict`, `act(cmd, *args, app=None) -> dict`
   via subprocess (list args, no shell), parse the JSON envelope, raise nothing on ok:false (return it), raise `ADError` on
   timeout / non-JSON. `class FakeAD` with the same interface driven by a scripted state machine (for tests).
2. `render.py` — `render(tree, max_lines=150, max_name=60, open_menus=()) -> (text, refmap)`. Text: one line per node that
   has a ref or a name, indented by depth, format `[k] role "name" (actions) {states}` where k is a SHORT id 1..N;
   `refmap[k] = full ref_id`. Menus: by default show each top-level `menu` as one line
   `[k] menu "File" (+17 items, use open_menu)` and hide its items; with `open_menus={full_ref_of_menu}` show that menu's
   items. Strip trailing spaces in names (mousepad pads menu names). Drop unnamed non-interactive containers but keep
   their children (flatten). Truncate to max_lines with a final line `... (N more)`. Deterministic output.
3. `actions.py` — parse ONE model action from text. Format: the model ends its reply with a single line
   `ACTION: {"op": "...", ...}` (JSON). ops: click{id}, double_click{id}, type{id,text}, set_value{id,value}, clear{id},
   toggle{id}, select{id}, press{keys}, open_menu{id} (env-only: re-render with that menu expanded, no GUI action),
   done{} (agent claims finished). Return a dataclass or an error string for malformed input (unknown op, missing field,
   bad JSON, id not in refmap).
4. `env.py` — `class DesktopEnv(ad, task, max_steps=15)`: `reset() -> obs_text`, `step(action_text) -> (obs_text, reward, done, info)`.
   obs_text = task instruction + rendered tree of the target app + last action result (ok / error code + message, short).
   Map short ids through the CURRENT refmap; after any GUI action take a fresh snapshot. If the snapshot shows a modal
   alert, render that. Episode ends on `done`, on max_steps, or on 3 consecutive malformed actions. reward is given only
   at the end: `task.check()` (float in [0,1]); malformed action -> no GUI call, error in obs. info: steps, n_errors,
   n_malformed, check details.
5. `tasks.py` — `class Task` (instruction:str, app:str, launch_cmd:list[str], setup(workdir), check(workdir)->(float, dict)).
   Generator `make_task(kind, rng, workdir)` with kinds (mousepad):
   - `write_save`: file `<w>/note.txt` does NOT exist; mousepad opens it ("create?" dialog -> Yes); instruction: type an
     exact random sentence (5-8 words from a fixed word list) and save. check: file content.strip() == sentence -> 1.0;
     file exists but wrong -> 0.2 * (token overlap ratio); missing -> 0.
   - `uppercase_save`: file with 3 random lines exists; instruction: convert ALL text to uppercase and save
     (doable via Edit>Select All, Edit>Convert>To Uppercase, File>Save). check exact.
   - `append_line`: append a given line at the end and save. check last line exact and earlier lines unchanged.
   - `delete_line`: delete line number k (1-based) and save. check exact.
   launch_cmd e.g. `["mousepad", path]`; the env runner (not you) launches it — just expose it.
6. `tests/` — FakeAD-based tests for render (menus collapsed/expanded, truncation, refmap), actions parsing (all ops +
   errors), env loop (reset/step/done/max_steps/malformed limit/stale ref error passthrough), each task's check() on
   correct/wrong/missing files (use tmp_path). Aim for ~30 tests, all fast.

Keep code compact and readable (type hints, short docstrings). When done, reply with: file list + line counts, pytest
summary line, and anything you were unsure about. Do not write long explanations.
