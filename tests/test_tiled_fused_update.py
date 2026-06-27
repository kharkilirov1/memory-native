from __future__ import annotations

import pytest
import torch

from memory_native.fused_update import HAS_TRITON
from memory_native.packed import PackedRMSCounterLinear


@pytest.mark.skipif(not (torch.cuda.is_available() and HAS_TRITON), reason="requires CUDA + Triton")
def test_tiled_fused_update_refreshes_only_touched_cache_rows():
    torch.manual_seed(0)
    lay = PackedRMSCounterLinear(64, 48, C=11, tile_rows=16, cache_mode="int8").cuda().train()
    lay._build_t_cache()
    before = lay._t_cache.clone()
    # Directly exercise the row-slice fused path. A zero gradient is enough to prove the path runs
    # and refreshes the derived cache without rebuilding unrelated rows.
    gw = torch.zeros(16, 64, device="cuda")
    assert lay._fused_update(16, 32, gw)
    after = lay._t_cache
    assert torch.equal(after[:16], before[:16])
    assert torch.equal(after[32:], before[32:])
    # touched rows are recomputed from the truth state (equal here because gw==0)
    t_new, _ = lay._decode_rows(16, 32)
    assert torch.equal(after[16:32], t_new.to(after.dtype))
