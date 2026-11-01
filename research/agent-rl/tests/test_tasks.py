"""Tests for task check() functions using tmp_path."""

import random

from deskenv.tasks import make_task, Task


def test_write_save_check_exact(tmp_path):
    rng = random.Random(42)
    task = make_task("write_save", rng, tmp_path)
    # Read the expected sentence from the task's instruction
    instruction = task.instruction
    # Instruction format: 'Open mousepad, type exactly: "SENTENCE" then save to note.txt'
    import re
    m = re.search(r'type exactly:\s+"(.+?)"', instruction)
    assert m, f"could not extract sentence from {instruction}"
    expected = m.group(1)
    (tmp_path / "note.txt").write_text(expected, encoding="utf-8")
    reward, info = task.check(tmp_path)
    assert reward == 1.0, f"expected exact match, got {info}"


def test_write_save_check_wrong(tmp_path):
    rng = random.Random(42)
    task = make_task("write_save", rng, tmp_path)
    (tmp_path / "note.txt").write_text(
        "Wrong sentence entirely.", encoding="utf-8"
    )
    reward, info = task.check(tmp_path)
    assert reward < 1.0, "wrong file should not get full reward"


def test_write_save_check_missing(tmp_path):
    rng = random.Random(42)
    task = make_task("write_save", rng, tmp_path)
    reward, info = task.check(tmp_path)
    assert reward == 0.0
    assert "does not exist" in info["reason"]


def test_uppercase_save_check_exact(tmp_path):
    rng = random.Random(7)
    task = make_task("uppercase_save", rng, tmp_path)
    # Setup creates the original file
    task.setup(tmp_path)
    # Now simulate correct uppercase result
    expected = "\n".join([
        line.upper()
        for line in (tmp_path / "note.txt").read_text().strip().splitlines()
    ]) + "\n"
    (tmp_path / "note.txt").write_text(expected, encoding="utf-8")
    reward, info = task.check(tmp_path)
    assert reward == 1.0, f"expected uppercase exact, got {info}"


def test_uppercase_save_check_not_uppercase(tmp_path):
    rng = random.Random(7)
    task = make_task("uppercase_save", rng, tmp_path)
    task.setup(tmp_path)  # creates file with lowercase lines
    reward, info = task.check(tmp_path)
    assert reward == 0.0, "lowercase file should get 0 reward"


def test_append_line_check_exact(tmp_path):
    rng = random.Random(3)
    task = make_task("append_line", rng, tmp_path)
    task.setup(tmp_path)
    # Read original, append the extra line
    orig = (tmp_path / "note.txt").read_text(encoding="utf-8")
    extra_line = task.instruction.split(": ")[-1]
    (tmp_path / "note.txt").write_text(
        orig.rstrip("\n") + "\n" + extra_line + "\n",
        encoding="utf-8",
    )
    reward, info = task.check(tmp_path)
    assert reward == 1.0, f"expected exact append, got {info}"


def test_append_line_check_partial(tmp_path):
    rng = random.Random(3)
    task = make_task("append_line", rng, tmp_path)
    task.setup(tmp_path)
    orig_lines = (tmp_path / "note.txt").read_text().strip().splitlines()
    # keep original lines but put wrong appended text
    (tmp_path / "note.txt").write_text(
        "\n".join(orig_lines) + "\nWRONG\n", encoding="utf-8"
    )
    reward, info = task.check(tmp_path)
    assert reward > 0, "original lines intact should get partial reward"
    assert reward < 1.0


def test_append_line_missing(tmp_path):
    rng = random.Random(3)
    task = make_task("append_line", rng, tmp_path)
    reward, info = task.check(tmp_path)
    assert reward == 0.0
    assert "does not exist" in info["reason"]


def test_delete_line_check_exact(tmp_path):
    rng = random.Random(5)
    task = make_task("delete_line", rng, tmp_path)
    task.setup(tmp_path)
    lines = (tmp_path / "note.txt").read_text().strip().splitlines()
    # Task deleted line k; we reconstruct what the result should be
    # Extract k from instruction
    import re
    m = re.search(r"line (\d+)", task.instruction)
    assert m
    k = int(m.group(1))
    expected_lines = [l for i, l in enumerate(lines, 1) if i != k]
    expected = "\n".join(expected_lines) + "\n"
    (tmp_path / "note.txt").write_text(expected, encoding="utf-8")
    reward, info = task.check(tmp_path)
    assert reward == 1.0, f"expected exact delete, got {info}"


def test_delete_line_check_wrong(tmp_path):
    rng = random.Random(5)
    task = make_task("delete_line", rng, tmp_path)
    task.setup(tmp_path)
    # Write something different
    (tmp_path / "note.txt").write_text("completely different\n", encoding="utf-8")
    reward, info = task.check(tmp_path)
    assert reward == 0.0


def test_unknown_task_kind(tmp_path):
    import random
    rng = random.Random(0)
    try:
        make_task("nonexistent", rng, tmp_path)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "unknown task kind" in str(e)


def test_task_attributes(tmp_path):
    import random
    rng = random.Random(42)
    task = make_task("write_save", rng, tmp_path)
    assert task.app == "mousepad"
    assert task.launch_cmd[0] == "mousepad"
    assert "note.txt" in str(task.launch_cmd)
