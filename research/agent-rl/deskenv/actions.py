"""Parse one model action from text into a structured command."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any


@dataclass
class Action:
    """A parsed, validated action from the model."""
    op: str
    id: str | None = None
    text: str | None = None
    value: str | None = None
    keys: str | None = None


# Known operations and their required fields
_OPS_REQUIRED: dict[str, set[str]] = {
    "click": {"id"},
    "double_click": {"id"},
    "type": {"id", "text"},
    "set_value": {"id", "value"},
    "clear": {"id"},
    "toggle": {"id"},
    "select": {"id"},
    "press": {"keys"},
    "open_menu": {"id"},
    "close_menu": {"id"},
    "done": set(),
}

_OPS_VALID = set(_OPS_REQUIRED)


def parse_action(text: str, refmap: dict[int, str]) -> Action | str:
    """Parse the **last** ``ACTION: <json>`` line in *text*.

    Parameters
    ----------
    text:
        Model reply.  The action line must be the very last line.
    refmap:
        Current short-id to full-ref-id mapping from ``render()``.

    Returns
    -------
    Action:
        On success, a validated action.
    str:
        A human-readable error message on failure.
    """
    # Find the last ACTION: line
    lines = text.strip().splitlines()
    action_line = None
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("ACTION:"):
            action_line = line[len("ACTION:"):].strip()
            break

    if action_line is None:
        return 'no ACTION: line found \u2014 end your reply with `ACTION: {"op": "..."}`'

    # Parse JSON
    try:
        raw: dict[str, Any] = json.loads(action_line)
    except json.JSONDecodeError as exc:
        return f"invalid JSON in ACTION: {exc}"

    if not isinstance(raw, dict):
        return "ACTION value must be a JSON object"

    # Validate op
    op = raw.get("op")
    if not isinstance(op, str) or op not in _OPS_VALID:
        valid = ", ".join(sorted(_OPS_VALID))
        return f"unknown op {op!r}; valid ops: {valid}"

    # Check required fields
    required = _OPS_REQUIRED[op]
    missing = required - set(raw.keys())
    if missing:
        return f"op {op!r} missing required field(s): {', '.join(sorted(missing))}"

    # For ops that take an id, map short id to full ref
    id_field = "id" if "id" in required else None
    resolved_id: str | None = None
    if id_field is not None:
        raw_id = raw.get("id")
        if raw_id is None:
            return f"op {op!r}: id is required"
        try:
            short = int(raw_id)
        except (ValueError, TypeError):
            return f"op {op!r}: id must be an integer (short id), got {raw_id!r}"
        resolved_id = refmap.get(short)
        if resolved_id is None:
            return f"op {op!r}: short id {short} not found in current tree"

    # Build result
    kw = dict(raw)
    kw.pop("op", None)
    if resolved_id is not None:
        kw["id"] = resolved_id
    # Extra keys not in Action fields are silently ignored
    valid_field_names = {f.name for f in fields(Action)}
    kw = {k: v for k, v in kw.items() if k in valid_field_names}
    return Action(op=op, **kw)
