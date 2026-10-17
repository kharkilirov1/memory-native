"""`--seed` must reproduce a whole gate run, not just the batch sampler.

Regression for the observed inconsistency: identical
``memory-native-charlm --config micro --steps 60 --device cpu --seed 0``
invocations reported dense val 2.9905 and 3.0529, because ``--seed`` only seeded
the batch generator while model init and the counter layers' stochastic rounding
used the global torch RNG.
"""
import torch

from memory_native import cli

ARGV = [
    "--config", "micro", "--steps", "2", "--batch", "4", "--eval-iters", "1",
    "--kinds", "dense,counter_rms", "--device", "cpu", "--seed",
]


def _stub_corpus(monkeypatch):
    """Offline corpus: no download, no cache, deterministic tensor."""
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, 32, (4096,), generator=g)
    monkeypatch.setattr(
        cli, "load_corpus", lambda device, path, download=True: (data, data, 32))


def _run_gate(capsys, seed):
    """Run the gate and return only the deterministic FINAL block.

    Per-kind lines carry wall-clock tok/s, so they are excluded on purpose.
    ``charlm_main`` replaces ``CONFIGS[name]`` with a vocab-corrected copy; the
    original object is restored so the test does not leak state into others.
    """
    original = cli.CONFIGS["micro"]
    try:
        cli.charlm_main(ARGV + [str(seed)])
    finally:
        cli.CONFIGS["micro"] = original
    out = capsys.readouterr().out
    return out[out.index("FINAL val loss:"):]


def test_same_seed_reproduces_the_gate(monkeypatch, capsys):
    _stub_corpus(monkeypatch)
    assert _run_gate(capsys, seed=7) == _run_gate(capsys, seed=7)


def test_different_seed_changes_the_run(monkeypatch, capsys):
    _stub_corpus(monkeypatch)
    assert _run_gate(capsys, seed=7) != _run_gate(capsys, seed=8)
