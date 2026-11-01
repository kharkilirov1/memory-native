"""Task definitions for the desktop RL environment.

Each task describes what the agent should accomplish, how to set up the
working directory before launch, and how to check success after the agent
declares ``done``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Word pool for random sentences
_WORDS = [
    "apple", "bird", "cloud", "dream", "eagle", "flame", "grape", "horse",
    "ivory", "jewel", "knife", "lemon", "mango", "night", "ocean", "pearl",
    "queen", "river", "stone", "tiger", "umbra", "vapor", "wheat", "yacht",
    "zebra", "amber", "bloom", "crane", "delta", "ember",
]


def _random_sentence(rng: random.Random, min_words: int = 5,
                     max_words: int = 8) -> str:
    n = rng.randint(min_words, max_words)
    words = [rng.choice(_WORDS) for _ in range(n)]
    words[0] = words[0].capitalize()
    return " ".join(words) + "."


def _token_overlap(a: str, b: str) -> float:
    """Simple token overlap ratio (Jaccard on whitespace-split tokens)."""
    ta = set(a.strip().lower().split())
    tb = set(b.strip().lower().split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


_CHECK: Callable[[Path], tuple[float, dict[str, Any]]]
_CHECK_FN = Callable[[Path], tuple[float, dict[str, Any]]]


@dataclass
class Task:
    """A single desktop task for the RL environment.

    Attributes
    ----------
    instruction:
        Human-readable instruction shown to the agent.
    app:
        Target app name (e.g. ``"mousepad"``).
    launch_cmd:
        Command list to start the app (e.g. ``["mousepad", "<workdir>/note.txt"]``).
    setup:
        Called with the working directory before the episode.  Creates/removes
        files as needed.
    check:
        Called with the working directory after the agent says ``done``.
        Returns ``(reward_float_in_0_1, info_dict)``.
    """

    instruction: str
    app: str
    launch_cmd: list[str]
    setup: Callable[[Path], None] = lambda w: None
    check: _CHECK_FN = lambda w: (0.0, {"reason": "check not implemented"})


def make_task(kind: str, rng: random.Random,
              workdir: Path) -> Task:
    """Generate a task instance of the given *kind*.

    Parameters
    ----------
    kind:
        One of ``"write_save"``, ``"uppercase_save"``, ``"append_line"``,
        ``"delete_line"``.
    rng:
        Seeded random number generator for reproducibility.
    workdir:
        Working directory for the episode.  Must already exist.
    """
    match kind:
        case "write_save":
            return _make_write_save(rng, workdir)
        case "uppercase_save":
            return _make_uppercase_save(rng, workdir)
        case "append_line":
            return _make_append_line(rng, workdir)
        case "delete_line":
            return _make_delete_line(rng, workdir)
        case _:
            raise ValueError(f"unknown task kind: {kind!r}")


# ------------------------------------------------------------------
# write_save
# ------------------------------------------------------------------

def _make_write_save(rng: random.Random, workdir: Path) -> Task:
    sentence = _random_sentence(rng)
    path = workdir / "note.txt"

    def setup(w: Path) -> None:
        # File must NOT exist before the task (mousepad shows a dialog)
        p = w / "note.txt"
        if p.exists():
            p.unlink()

    def check(w: Path) -> tuple[float, dict[str, Any]]:
        p = w / "note.txt"
        if not p.exists():
            return 0.0, {"reason": "file does not exist"}
        content = p.read_text(encoding="utf-8").strip()
        if content == sentence:
            return 1.0, {"reason": "exact match"}
        overlap = _token_overlap(content, sentence)
        return 0.2 * overlap, {
            "reason": "content mismatch",
            "expected": sentence,
            "got": content,
            "token_overlap": round(overlap, 4),
        }

    return Task(
        instruction=f'Open mousepad, type exactly: "{sentence}" then save to note.txt',
        app="mousepad",
        launch_cmd=["mousepad", str(path)],
        setup=setup,
        check=check,
    )


# ------------------------------------------------------------------
# uppercase_save
# ------------------------------------------------------------------

def _random_lines(rng: random.Random, n: int = 3) -> list[str]:
    return [_random_sentence(rng, 4, 7) for _ in range(n)]


def _make_uppercase_save(rng: random.Random, workdir: Path) -> Task:
    lines = _random_lines(rng)
    path = workdir / "note.txt"

    def setup(w: Path) -> None:
        w.joinpath("note.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    expected = "\n".join(l.upper() for l in lines) + "\n"

    def check(w: Path) -> tuple[float, dict[str, Any]]:
        p = w / "note.txt"
        if not p.exists():
            return 0.0, {"reason": "file does not exist"}
        content = p.read_text(encoding="utf-8")
        if content == expected:
            return 1.0, {"reason": "exact match (uppercase)"}
        return 0.0, {
            "reason": "content not fully uppercased",
            "expected": expected,
            "got": content,
        }

    return Task(
        instruction="Convert ALL text to uppercase using Edit>Select All, Edit>Convert>To Uppercase, then File>Save",
        app="mousepad",
        launch_cmd=["mousepad", str(path)],
        setup=setup,
        check=check,
    )


# ------------------------------------------------------------------
# append_line
# ------------------------------------------------------------------

def _make_append_line(rng: random.Random, workdir: Path) -> Task:
    lines = _random_lines(rng)
    extra = _random_sentence(rng, 4, 6)
    path = workdir / "note.txt"

    def setup(w: Path) -> None:
        w.joinpath("note.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    expected = "\n".join(lines) + "\n" + extra + "\n"

    def check(w: Path) -> tuple[float, dict[str, Any]]:
        p = w / "note.txt"
        if not p.exists():
            return 0.0, {"reason": "file does not exist"}
        content = p.read_text(encoding="utf-8")
        if content == expected:
            return 1.0, {"reason": "exact match (appended)"}
        # Partial: original lines intact?
        orig = content.splitlines()
        if len(orig) >= len(lines) and orig[:len(lines)] == lines:
            return 0.5, {
                "reason": "original lines intact, but appended text differs",
                "expected_line": extra,
                "got_last": orig[-1] if orig else "",
            }
        return 0.0, {
            "reason": "content changed unexpectedly",
            "expected": expected,
            "got": content,
        }

    return Task(
        instruction=f"Append the following line at the end and save: {extra}",
        app="mousepad",
        launch_cmd=["mousepad", str(path)],
        setup=setup,
        check=check,
    )


# ------------------------------------------------------------------
# delete_line
# ------------------------------------------------------------------

def _make_delete_line(rng: random.Random, workdir: Path) -> Task:
    lines = _random_lines(rng, 4)
    delete_k = rng.randint(1, len(lines))
    path = workdir / "note.txt"

    def setup(w: Path) -> None:
        w.joinpath("note.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    expected_lines = [l for i, l in enumerate(lines) if i != delete_k - 1]
    expected = "\n".join(expected_lines) + "\n"

    def check(w: Path) -> tuple[float, dict[str, Any]]:
        p = w / "note.txt"
        if not p.exists():
            return 0.0, {"reason": "file does not exist"}
        content = p.read_text(encoding="utf-8")
        if content == expected:
            return 1.0, {"reason": "exact match (deleted line)"}
        return 0.0, {
            "reason": "content mismatch after delete",
            "expected": expected,
            "got": content,
        }

    return Task(
        instruction=f"Delete line {delete_k} and save",
        app="mousepad",
        launch_cmd=["mousepad", str(path)],
        setup=setup,
        check=check,
    )
