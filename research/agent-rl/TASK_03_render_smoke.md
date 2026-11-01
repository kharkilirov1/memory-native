# TASK 03 — render fixes + working smoke policies (same rules: only research/agent-rl/, no git, pytest at the end)

I (Claude) already fixed and applied (keep them, they are in the working tree): ad.py returns the JSON envelope on
non-zero exit; session.py passes DBUS_SESSION_BUS_ADDRESS to apps and agent-desktop, uses GSETTINGS_BACKEND=memory and
per-session XDG_* dirs (this removed mousepad's "previous session did not end normally - restore?" dialog; a NEW file
now opens directly with NO dialog); smoke_real.py calls session.start().

## Real observation now (uppercase_save, at reset) — note the bugs
```
[short_id=1] window "/tmp/deskrl/tmp31s8i7r6/note.txt - Mousepad"
  [short_id=2] menu (+249 items, use open_menu)
  [short_id=3] button "New" (click,SetF) {offscreen}
  ...
  [short_id=18] tablist (sel)
    [short_id=19] tab (sel) {selected}
      [short_id=20] scrollarea (SetF,scr)
        [short_id=21] textfield (SetV,Type,SetF) {focused}
        [short_id=22] scrollbar (SetV) {offscreen}
```
1. The menubar is an UNNAMED `menu` whose children are the named menus File/Edit/Search/View/Document/Help. Treat an
   unnamed menu (menubar) as a transparent container: show its named child menus, each collapsed as
   `[k] menu "Edit" (+N items, open_menu)`. open_menu on "Edit" must show its items, incl. submenu "Convert" collapsed;
   open_menu on "Convert" shows "To Uppercase" etc. Name paths like ("Edit","Convert").
2. Line format must be `[k] role "name" (actions) {states}` exactly as in TASK_01 — `[21]`, not `[short_id=21]`.
3. Hide `{offscreen}` toolbar buttons by default (they cannot be clicked headless: agent-desktop refuses
   "Target is not actionable: visible (live offscreen state is true)"); keep `{offscreen}` nodes only if they have
   actionable children. Keep the textfield.
4. Show the textfield's current text (agent-desktop puts it in node `value` if present; also check `text`/`description`
   keys) truncated to 300 chars, so the agent can see the content.
5. `session.launch()`: redirect the app's stdout/stderr to `<workdir>/app.log` (inherited pipes keep callers hanging).

## smoke_real.py policies must be STATEFUL (closure/class with a step counter) and END with `done`
Verified on the real UI: `type` into the textfield works; `set_value` on the textfield works (replaces all text);
`press ctrl+s` saves (title loses the leading `*`); `focus` is refused by policy (don't use it).
- write_save: type the sentence into the textfield, press ctrl+s, done.
- uppercase_save (menu path, exercises open_menu): open_menu Edit -> click "Select All"; open_menu Edit -> open_menu
  Convert -> click "To Uppercase"; press ctrl+s; done. Fallback if a step errors: set_value with the uppercased text.
- append_line / delete_line: set_value the textfield to the target content (from the task's own fields), ctrl+s, done.
Wrap the episode loop in try/finally so the app and the session are always killed. Print the observation and action
of every step with `--verbose`.

Reply with the short report (files changed, pytest line, doubts).
