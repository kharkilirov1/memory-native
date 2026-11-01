"""Tests for render()."""

from deskenv.render import render


def _node(role="window", name="", ref_id="", actions=None, states=None,
          children=None, **kw):
    n = {"role": role, "name": name, "ref_id": ref_id,
         "available_actions": actions or [], "states": states or []}
    n.update(kw)
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
    # Window line uses new [k] format
    assert "[1] window \"Mousepad\"" in lines[0]
    # Check collapsed menus show item counts
    menu_line = [l for l in lines if "File" in l]
    assert any("(+3 items" in l for l in menu_line), "collapsed menu should show item count"


def test_render_refmap():
    text, refmap = render(SAMPLE_TREE)
    assert 1 in refmap
    assert refmap[1] == "@s1:w1"


def test_render_expand_menu():
    text, refmap = render(SAMPLE_TREE, open_menus=("@s1:e1",))
    assert "[2]" in text
    assert "@s1:e1" in str(refmap.values())


def test_render_max_lines():
    big_tree = _node("window", "Big", children=[
        _node("button", f"Btn {i}", f"@b{i}", actions=["Click"]) for i in range(200)
    ])
    text, refmap = render(big_tree, max_lines=20)
    lines = text.splitlines()
    assert len(lines) == 21  # 20 lines + 1 truncation
    assert lines[-1].startswith("\u2026")


def test_render_truncation_count():
    big_tree = _node("window", "Big", children=[
        _node("button", f"Btn {i}", f"@b{i}", actions=["Click"]) for i in range(200)
    ])
    text, refmap = render(big_tree, max_lines=10)
    lines = text.splitlines()
    assert "more" in lines[-1]


def test_render_toolbar_flattened():
    text, refmap = render(SAMPLE_TREE)
    assert "Bold" in text
    assert "Italic" in text


def test_render_name_truncation():
    tree = _node("window", "A" * 100, "@w1")
    text, refmap = render(tree, max_name=10)
    assert "\u2026" in text
    assert ' "AAAAAAAAAA' not in text


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


def test_render_no_ref_skip():
    """Nodes without ref_id or name that have no actions get flattened."""
    tree = _node("window", "W", "@w1", children=[
        _node("group", "", "", children=[
            _node("button", "Inner", "@b1", actions=["Click"]),
        ]),
    ])
    text, refmap = render(tree)
    assert "Inner" in text
    assert "[2]" in text


# ── TASK_03 regression tests ────────────────────────────────────────────

def test_bug1_unnamed_menu_flattened():
    """Bug 1: unnamed menu (menubar) is a transparent container.

    Its named child menus should appear as individual collapsed menus.
    """
    tree = _node("window", "W", "@w1", children=[
        _node("menu", "", "", actions=["Click"], children=[
            _node("menu", "File", "@f1", actions=["Click"],
                  children=[_node("menuitem", "Save", "@s1", actions=["Click"])]),
            _node("menu", "Edit", "@e1", actions=["Click"],
                  children=[_node("menuitem", "Undo", "@u1", actions=["Click"])]),
        ]),
        _node("textfield", "text", "@t1", actions=["Type"]),
    ])
    text, refmap = render(tree)
    # The unnamed menu should not get its own line
    assert "menu (+" not in text.splitlines()[1]  # not one big collapsed menu
    # Named child menus should appear
    assert '"File"' in text
    assert '"Edit"' in text
    assert '"Save"' not in text  # collapsed
    assert '"Undo"' not in text
    # Textfield still visible
    assert "[2]" in text or "[3]" in text


def test_bug2_bracket_format():
    """Bug 2: line format is [k] not [short_id=k]."""
    tree = _node("window", "W", "@w1")
    text, refmap = render(tree)
    assert "[short_id=" not in text
    assert "[1]" in text


def test_bug3_offscreen_hidden():
    """Bug 3: offscreen nodes without actionable children are hidden."""
    tree = _node("window", "W", "@w1", children=[
        _node("button", "OffBtn", "@b1", actions=["Click"],
              states=["offscreen"]),
        _node("button", "Visible", "@b2", actions=["Click"]),
    ])
    text, refmap = render(tree)
    assert "OffBtn" not in text
    assert "Visible" in text


def test_bug3_offscreen_kept_with_actionable_children():
    """Bug 3: offscreen nodes WITH actionable children are kept."""
    tree = _node("window", "W", "@w1", children=[
        _node("group", "grp", "@g1", states=["offscreen"], children=[
            _node("button", "Inner", "@b1", actions=["Click"]),
        ]),
    ])
    text, refmap = render(tree)
    assert "Inner" in text


def test_bug4_textfield_value():
    """Bug 4: textfield shows its value truncated to 300 chars."""
    tree = _node("textfield", "edit", "@t1", actions=["Type"],
                 value="Hello World")
    text, refmap = render(tree)
    assert 'text="Hello World"' in text


def test_bug4_textfield_value_truncated():
    """Bug 4: long values are truncated."""
    long_val = "A" * 500
    tree = _node("textfield", "edit", "@t1", actions=["Type"],
                 value=long_val)
    text, refmap = render(tree)
    assert "\u2026" in text  # ellipsis truncation marker


def test_bug4_textfield_fallback_text_key():
    """Bug 4: textfield uses 'text' key when 'value' is absent."""
    tree = _node("textfield", "edit", "@t1", actions=["Type"],
                 text="from text key")
    text, refmap = render(tree)
    assert 'text="from text key"' in text


def test_menu_is_open_selected():
    """_menu_is_open: a menu with states=[selected] and onscreen items is expanded."""
    tree = _node("window", "W", "@w1", children=[
        _node("menu", "File", "@f1", states=["selected"],
              children=[
                  _node("menuitem", "Save", "@s1", actions=["Click"]),
              ]),
        _node("menu", "Edit", "@e1",
              children=[
                  _node("menuitem", "Undo", "@u1", actions=["Click"]),
              ]),
    ])
    text, refmap = render(tree)
    # File menu (selected + onscreen children) is expanded: shows "Save"
    assert "Save" in text
    # Edit menu (no selected state, children are offscreen) is collapsed
    assert 'menu "Edit" (+1 items, use open_menu)' in text


def test_menu_is_open_closed_menu():
    """_menu_is_open: a menu without selected state stays collapsed even if items are onscreen."""
    tree = _node("window", "W", "@w1", children=[
        _node("menu", "File", "@f1",
              children=[
                  _node("menuitem", "Save", "@s1", actions=["Click"]),
              ]),
    ])
    text, refmap = render(tree)
    # Without selected/expanded state, the menu is collapsed regardless of children
    assert 'menu "File" (+1 items, use open_menu)' in text
    assert "Save" not in text
