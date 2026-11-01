"""Run one RL episode end to end with a real or fake desktop session."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any, Callable

from .ad import AgentDesktop, FakeAD
from .env import DesktopEnv
from .session import DesktopSession
from .tasks import make_task

Policy = Callable[[str], str]
"""Signature: ``policy(obs_text: str) -> action_text``."""


def run_episode(
    session: DesktopSession | None,
    ad: AgentDesktop | FakeAD,
    task_kind: str,
    rng: random.Random,
    policy: Policy,
    max_steps: int = 15,
) -> dict[str, Any]:
    """Run one episode and return the result dict.

    Parameters
    ----------
    session:
        A running ``DesktopSession`` (or ``None`` for FakeAD testing).
    ad:
        The agent-desktop wrapper (real or fake).
    task_kind:
        One of the task kinds accepted by ``make_task``.
    rng:
        Seeded RNG for reproducibility.
    policy:
        Callable that maps observation text to an action string.
    max_steps:
        Maximum steps before forced termination.

    Returns
    -------
    dict with keys ``reward``, ``steps``, ``info``, ``transcript``.
    """
    # Create workdir
    if session is not None:
        workdir = session.new_workdir()
    else:
        workdir = Path(tempfile.mkdtemp())

    task = make_task(task_kind, rng, workdir)
    task.setup(workdir)

    # Launch the app
    launched = None
    if session is not None:
        launched = session.launch(task.launch_cmd, wait_app=task.app)

    env = DesktopEnv(ad, task, workdir=workdir, max_steps=max_steps)
    obs = env.reset()
    transcript: list[dict[str, Any]] = [{"role": "obs", "text": obs}]

    done = False
    while not done:
        action = policy(obs)
        transcript.append({"role": "action", "text": action})
        obs, reward, done, info = env.step(action)
        transcript.append({"role": "obs", "text": obs})

    # Cleanup
    if launched is not None:
        session.kill(launched)  # type: ignore[union-attr]

    return {
        "reward": reward,
        "steps": info.get("steps", 0),
        "info": info,
        "transcript": transcript,
    }
