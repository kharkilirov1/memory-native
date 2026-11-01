"""Render an accessibility tree as indented text for the LLM, and menu path helpers."""

from __future__ import annotations

from typing import Any

_SHORT_ACTIONS = {
    "Click": "click",
    "Select": "sel",
    "Press": "press",
    "Type": "type",
    "Focus": "focus",
    "Toggle": "toggle",
    "Check": "chk",
    "Uncheck": "unchk",
    "Expand": "exp",
    "Collapse": "col",
    "Clear": "clr",
    "Scroll": "scr",
}


def _compact_actions(actions: list[str] | None) -> str:
    if not actions:
        return ""
    abbrs = [_SHORT_ACTIONS.get(a, a[:4]) for a in actions]
    return f" ({','.join(abbrs)})"


def _states_str(states: list[str] | None) -> str:
    if not states:
        return ""
    return " {" + ",".join(states) + "}"


def _name_str(name: str | None, max_name: int) -> str:
    if not name:
        return ""
    stripped = name.rstrip()
    if len(stripped) > max_name:
        stripped = stripped[: max_name - 1] + "\u2026"
    return f' "{stripped}"'


def _count_descendants(node: dict[str, Any]) -> int:
    """Count all descendant *nodes* (including non-interactive, recursively)."""
    total = 0
    for child in node.get("children") or []:
        total += 1 + _count_descendants(child)
    return total


# Roles that are never interactive enough to warrant a line; their children
# get flattened up.
_FLATTEN_ROLES = frozenset({"group", "scrollarea", "scrollbar"})


def _flatten(
    node: dict[str, Any],
    depth: int,
    open_menus: set[str],
    max_name: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Walk the tree and produce (depth, node) pairs, flattening as needed."""
    rows: list[tuple[int, dict[str, Any]]] = []
    role = (node.get("role") or "").lower()
    name = node.get("name") or ""
    ref = node.get("ref_id") or ""
    children = node.get("children") or []

    # ── menu collapsing ────────────────────────────────────────────────
    if role == "menu" and ref not in open_menus:
        count = _count_descendants(node)
        # Represent the collapsed menu as a single line
        collapsed = dict(node)
        collapsed["_collapsed_count"] = count
        rows.append((depth, collapsed))
        return rows  # don't descend

    # ── flatten containers ─────────────────────────────────────────────
    if role in _FLATTEN_ROLES and not ref and not name:
        for child in children:
            rows.extend(_flatten(child, depth, open_menus, max_name))
        return rows

    # ── determine whether this node gets its own line ───────────────────
    has_ref = bool(ref)
    has_name = bool(name)
    interactive = bool(node.get("available_actions"))

    if not has_ref and not has_name and not interactive:
        # Unnamed non-interactive container – flatten children up
        for child in children:
            rows.extend(_flatten(child, depth, open_menus, max_name))
        return rows

    rows.append((depth, node))
    for child in children:
        rows.extend(_flatten(child, depth + 1, open_menus, max_name))
    return rows


def render(
    tree: dict[str, Any],
    max_lines: int = 150,
    max_name: int = 60,
    open_menus: tuple[str, ...] = (),
) -> tuple[str, dict[int, str]]:
    """Produce indented text and a short-id to ref_id map.

    Parameters
    ----------
    tree:
        Root accessibility node.
    max_lines:
        Maximum output lines; excess is truncated.
    max_name:
        Maximum displayed name length.
    open_menus:
        ``ref_id``s of menus whose children should be shown.

    Returns
    -------
    text:
        Indented text representation.
    refmap:
        ``{short_id: full_ref_id}`` for every node that has a ``ref_id``.
    """
    open_set = set(open_menus)
    flat = _flatten(tree, 0, open_set, max_name)

    lines: list[str] = []
    refmap: dict[int, str] = {}
    short_id = 0

    for depth, node in flat:
        role = (node.get("role") or "?").lower()
        name = node.get("name") or ""
        ref = node.get("ref_id") or ""
        actions = node.get("available_actions") or []
        states = node.get("states") or []

        coll_count = node.get("_collapsed_count")
        if coll_count is not None:
            short_id += 1
            indent = "  " * depth
            name_display = _name_str(name, max_name)
            body = f"[short_id={short_id}] {role}{name_display} (+{coll_count} items, use open_menu)"
            lines.append(f"{indent}{body}")
            if ref:
                refmap[short_id] = ref
            continue

        short_id += 1
        indent = "  " * depth
        name_display = _name_str(name, max_name)
        actions_display = _compact_actions(actions)
        states_display = _states_str(states)
        body = (
            f"[short_id={short_id}] {role}"
            f"{name_display}{actions_display}{states_display}"
        )
        lines.append(f"{indent}{body}")
        if ref:
            refmap[short_id] = ref

    # Truncation
    if len(lines) > max_lines:
        extra = len(lines) - max_lines
        lines = lines[:max_lines]
        lines.append(f"\u2026 ({extra} more)")

    return "\n".join(lines), refmap


# ---------------------------------------------------------------------------
# Menu name-path helpers  (cross-snapshot stable)
# ---------------------------------------------------------------------------


def resolve_menu_name_path(
    tree: dict[str, Any], path: tuple[str, ...]
) -> str | None:
    """Walk *tree* following *path* and return the ref_id of the deepest node.

    *path* is a sequence of stripped menu names, e.g. ``("File",)`` or
    ``("Edit", "Convert")``.  Returns ``None`` if any step is not found.
    """
    current = tree
    for name in path:
        found: dict[str, Any] | None = None
        for child in current.get("children") or []:
            if child.get("name", "").rstrip() == name:
                found = child
                break
        if found is None:
            return None
        current = found
    return current.get("ref_id")


def find_menu_name_path(
    tree: dict[str, Any], target_ref_id: str
) -> tuple[str, ...] | None:
    """Walk *tree* depth-first and return the name-path to the node with ref_id.

    Returns ``()`` for the root node, ``("File",)`` for a top-level menu,
    ``("Edit", "Convert")`` for a submenu, or ``None`` when not found.
    Non-string names are skipped (the path segment is an empty string).
    """
    return _find_path_dfs(tree, target_ref_id, ())


def _find_path_dfs(
    node: dict[str, Any], target: str, cur_path: tuple[str, ...]
) -> tuple[str, ...] | None:
    if node.get("ref_id") == target:
        return cur_path
    for child in node.get("children") or []:
        child_name = (child.get("name") or "").rstrip()
        next_path = cur_path + (child_name,) if child_name else cur_path
        result = _find_path_dfs(child, target, next_path)
        if result is not None:
            return result
    return None
