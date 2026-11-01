"""Tests for action parsing."""

from deskenv.actions import parse_action, Action


REFS = {1: "@s1:w1", 2: "@s1:e1", 3: "@s1:e10", 42: "@big:btn"}


def test_parse_click():
    result = parse_action('ACTION: {"op": "click", "id": 1}', REFS)
    assert isinstance(result, Action)
    assert result.op == "click"
    assert result.id == "@s1:w1"


def test_parse_double_click():
    result = parse_action('ACTION: {"op": "double_click", "id": 2}', REFS)
    assert isinstance(result, Action)
    assert result.op == "double_click"
    assert result.id == "@s1:e1"


def test_parse_type():
    result = parse_action('ACTION: {"op": "type", "id": 3, "text": "hello world"}', REFS)
    assert isinstance(result, Action)
    assert result.op == "type"
    assert result.id == "@s1:e10"
    assert result.text == "hello world"


def test_parse_set_value():
    result = parse_action('ACTION: {"op": "set_value", "id": 42, "value": "new text"}', REFS)
    assert isinstance(result, Action)
    assert result.op == "set_value"
    assert result.id == "@big:btn"
    assert result.value == "new text"


def test_parse_clear():
    result = parse_action('ACTION: {"op": "clear", "id": 1}', REFS)
    assert isinstance(result, Action)
    assert result.op == "clear"


def test_parse_toggle():
    result = parse_action('ACTION: {"op": "toggle", "id": 2}', REFS)
    assert isinstance(result, Action)
    assert result.op == "toggle"


def test_parse_select():
    result = parse_action('ACTION: {"op": "select", "id": 2}', REFS)
    assert isinstance(result, Action)
    assert result.op == "select"


def test_parse_press():
    result = parse_action('ACTION: {"op": "press", "keys": "ctrl+s"}', REFS)
    assert isinstance(result, Action)
    assert result.op == "press"
    assert result.keys == "ctrl+s"


def test_parse_open_menu():
    result = parse_action('ACTION: {"op": "open_menu", "id": 2}', REFS)
    assert isinstance(result, Action)
    assert result.op == "open_menu"
    assert result.id == "@s1:e1"


def test_parse_done():
    result = parse_action('ACTION: {"op": "done"}', REFS)
    assert isinstance(result, Action)
    assert result.op == "done"
    assert result.id is None


def test_parse_no_action_line():
    result = parse_action("just some text", REFS)
    assert isinstance(result, str)
    assert "no ACTION: line" in result


def test_parse_invalid_json():
    result = parse_action('ACTION: {bad json}', REFS)
    assert isinstance(result, str)
    assert "invalid JSON" in result


def test_parse_unknown_op():
    result = parse_action('ACTION: {"op": "fly"}', REFS)
    assert isinstance(result, str)
    assert "unknown op" in result


def test_parse_missing_required_field():
    result = parse_action('ACTION: {"op": "click"}', REFS)  # missing id
    assert isinstance(result, str)
    assert "missing required" in result.lower()


def test_parse_id_not_in_refmap():
    result = parse_action('ACTION: {"op": "click", "id": 999}', REFS)
    assert isinstance(result, str)
    assert "not found" in result


def test_parse_id_not_int():
    result = parse_action('ACTION: {"op": "click", "id": "abc"}', REFS)
    assert isinstance(result, str)
    assert "must be an integer" in result


def test_parse_action_trailing_text():
    """Action line can have text after it; we take the last ACTION: line."""
    result = parse_action(
        "Some reasoning...\nACTION: {\"op\": \"done\"}\n", REFS
    )
    assert isinstance(result, Action)
    assert result.op == "done"


def test_parse_multiple_action_lines():
    """Only the last ACTION: line counts."""
    result = parse_action(
        'ACTION: {"op": "click", "id": 1}\nACTION: {"op": "done"}', REFS
    )
    assert isinstance(result, Action)
    assert result.op == "done"


def test_parse_extra_fields_ignored():
    """Extra keys in the JSON object should be ignored (not cause error)."""
    result = parse_action(
        'ACTION: {"op": "done", "extra": "stuff", "unused": 42}', REFS
    )
    assert isinstance(result, Action)
    assert result.op == "done"
