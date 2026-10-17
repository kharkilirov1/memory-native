"""The 6-bit code (0.75 B/param) is a lower bound; pin the accounted overheads.

Companion to `scripts/state_accounting.py` and `research/results/STATE_ACCOUNTING.md`.
"""
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SPEC = importlib.util.spec_from_file_location(
    "state_accounting", Path(__file__).resolve().parents[1] / "scripts" / "state_accounting.py")
state_accounting = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(state_accounting)

from memory_native.group_scale_packed import _salient_index_dtype  # noqa: E402


def _rows(acc):
    return {name: size for name, _shape, _dtype, size, _persistent in acc["rows"]}


def test_packed_body_is_exactly_six_bits_per_param():
    acc = state_accounting.accounting(512, 512, group=128, C=11, salient_fraction=0.02)
    n = acc["params"]
    assert acc["ideal_bytes"] == int(n * 0.75)
    assert acc["state_bytes"] == n * 3 // 4          # 4 codes / 3 bytes, no padding


def test_overhead_is_group_metadata_plus_the_salient_channel():
    acc = state_accounting.accounting(512, 512, group=128, C=11, salient_fraction=0.02)
    n = acc["params"]
    assert 0.75 < acc["persistent_bytes"] / n < 1.0
    assert acc["persistent_bytes"] < acc["resident_bytes"]
    assert acc["resident_bytes"] / n < 1.2
    rows = _rows(acc)
    # below 2**31 positions the runtime mirror uses the same int32 width as the index
    assert rows["_salient_perm_flat"] == rows["salient_idx"]
    assert rows["salient_val"] == rows["salient_idx"] // 2   # fp16 vs int32


def test_salient_index_dtype_rule():
    assert _salient_index_dtype(2 ** 31) is torch.int32       # max index 2**31-1 fits
    assert _salient_index_dtype(2 ** 31 + 1) is torch.int64   # 27B-class flat index


def test_layer_mirror_dtype_follows_its_own_size():
    """The rule is exercised on a real layer without allocating 2**31 coefficients."""
    layer = state_accounting.PackedGroupScaleCounterLinear(
        64, 64, group=64, C=11, kernel_mode="gemm")
    assert layer._salient_perm_flat.dtype is torch.int32
    assert _salient_index_dtype(layer.out_features * layer.in_features) is torch.int32


def test_without_salient_only_group_metadata_remains():
    acc = state_accounting.accounting(512, 512, group=128, C=11, salient_fraction=0.0)
    n = acc["params"]
    rows = _rows(acc)
    assert rows["salient_idx"] == 0
    assert rows["salient_val"] == 0
    assert acc["persistent_bytes"] / n < 0.8     # 0.75 + groups scales/RMS/perm


def test_strict_scratch_is_a_transient_not_part_of_the_state():
    acc = state_accounting.accounting(512, 512, group=128, C=11, salient_fraction=0.02)
    assert acc["strict_scratch_bytes"] > 0
    assert acc["strict_scratch_bytes"] < acc["persistent_bytes"]
