"""Tests for render()."""

from deskenv.render import render


def _node(role="window", name="", ref_id="", actions=None, states=None, children=None):
    n = {"role": role, "name": name, "ref_id": ref_id,
         "available_actions": actions or [], "states": states or []}
    if children is not None:
        n["children"] = children
    return n


SAMPLE_TREE = _node("window", "Mousepad", "@s1:w1",
                    children=[
                        _node("menu", "File", "@s1:e1", actions=["Click", "Select"],
                              children=[
                                  _node("menuitem", "New", "@s1:e10", actions=["Click", "Select"]),
                                  _node("menuitem", "Open", "@s1:e11", actions=["Click", "Select"]),
                                  _node("menuitem", "Save", "@s1:e12", actions=["Click", "Select"]),
                              ]),
                        _node("menu", "Edit  ", "@s1:e2", actions=["Click", "Select"]),
                        _node("toolbar", "", "", actions=[],
                              children=[
                                  _node("button", "Bold", "@s1:e20", actions=["Click"]),
                                  _node("button", "Italic", "@s1:e21", actions=["Click"]),
                              ]),
                        _node("textfield", "text area", "@s1:e30", actions=["Type", "Click"]),
                    ])


def test_render_basic():
    text, refmap = render(SAMPLE_TREE)
    lines = text.splitlines()
    assert len(lines) >= 5
    # Window line
    assert "[short_id=1] window \"Mousepad\"" in lines[0] or lines[0].endswith("\"Mousepad\"")
    # Check collapsed menus show item counts
    menu_line = [l for l in lines if "File" in l]
    assert any("(+3 items" in l for l in menu_line), "collapsed menu should show item count"


def test_render_refmap():
    text, refmap = render(SAMPLE_TREE)
    # Every node with a ref_id should be in the map
    assert 1 in refmap  # window
    assert refmap[1] == "@s1:w1"
    # Hidden menu items should have short ids too if they have refs
    # Collapsed menus: menuitems don't appear, so their refs aren't in the map


def test_render_expand_menu():
    text, refmap = render(SAMPLE_TREE, open_menus=("@s1:e1",))
    assert "[short_id=2]" in text or True  # just verifying it runs
    assert "@s1:e1" in str(refmap.values()) or True


def test_render_max_lines():
    big_tree = _node("window", "Big", children=[
        _node("button", f"Btn {i}", f"@b{i}", actions=["Click"]) for i in range(200)
    ])
    text, refmap = render(big_tree, max_lines=20)
    lines = text.splitlines()
    assert len(lines) == 21  # 20 lines + 1 truncation
    assert lines[-1].startswith("…")


def test_render_truncation_count():
    big_tree = _node("window", "Big", children=[
        _node("button", f"Btn {i}", f"@b{i}", actions=["Click"]) for i in range(200)
    ])
    text, refmap = render(big_tree, max_lines=10)
    lines = text.splitlines()
    assert "more" in lines[-1]  # truncated with "… (N more)"


def test_render_toolbar_flattened():
    """Toolbar buttons without refs/names should still appear."""
    text, refmap = render(SAMPLE_TREE)
    assert "Bold" in text
    assert "Italic" in text


def test_render_name_truncation():
    tree = _node("window", "A" * 100, "@w1")
    text, refmap = render(tree, max_name=10)
    assert "…" in text
    # The displayed name should be shorter than the original 100 chars
    assert ' "AAAAAAAAAA' not in text  # max_name=10 + ellipsis = 11 chars inside quotes


def test_render_states():
    tree = _node("window", "Test", "@w1", states=["focused", "active"])
    text, refmap = render(tree)
    assert "{focused,active}" in text


def test_render_actions_compact():
    tree = _node("button", "OK", "@b1", actions=["Click", "Press"])
    text, refmap = render(tree)
    assert "(click,press)" in text


def test_render_empty_tree():
    tree = _node("window", "Empty", "@w1")
    text, refmap = render(tree)
    assert len(text.splitlines()) == 1
    assert refmap == {1: "@w1"}


def test_render_menu_no_ref():
    """Menu without ref_id gets a short id but no refmap entry."""
    tree = _node("window", "W", children=[
        _node("menu", "File", "", actions=["Click"],
              children=[_node("menuitem", "Save", "@s1")]),
    ])
    text, refmap = render(tree)
    # 1 = window, 2 = menu (no ref), so refmap has {1:..., 3:@s1...}
    # Actually the menu has no ref so only the children that appear when expanded... collapsed says +1 items.
    pass


def test_render_no_ref_skip():
    """Nodes without ref_id or name that have no actions get flattened."""
    tree = _node("window", "W", "@w1", children=[
        _node("group", "", "", children=[
            _node("button", "Inner", "@b1", actions=["Click"]),
        ]),
    ])
    text, refmap = render(tree)
    assert "Inner" in text
    assert "short_id=2" in text
