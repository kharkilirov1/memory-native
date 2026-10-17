"""Import and execute both public runners, so publishing errors cannot hide."""
import importlib.util
from pathlib import Path
import math
import pytest
import torch


def load_runner(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode', ['native', 'qat_adam', 'frozen'])
def test_matrix_runner_two_steps(mode):
    runner = load_runner('binary_factor_witness')
    result = runner.run(seed=3, mode=mode, method='svd', lr=.001, steps=2)
    assert math.isfinite(result['test'])
    assert result['state']['persistent_total'] > 0
    if mode == 'native':
        assert result['updates'] == [2, 2]
    elif mode == 'frozen':
        assert result['updates'] == [0, 0]


@pytest.mark.parametrize('mode', ['native', 'qat_adam', 'frozen'])
def test_text_runner_converted_forward_and_backward(mode):
    runner = load_runner('binary_factor_text')
    torch.manual_seed(51)
    teacher = runner.ByteGPT()
    model, layers, _ = runner.converted(teacher, mode, 'svd', .001, 51)
    x = torch.randint(0, 256, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, 256)
    logits.square().mean().backward()
    assert len(layers) == 4
    for layer in layers:
        assert layer.h.grad is not None
        if mode in ('native', 'frozen'):
            expected = 1 if mode == 'native' else 0
            assert int(layer.left.steps) == int(layer.right.steps) == expected
