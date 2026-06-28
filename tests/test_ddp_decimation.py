"""Data-parallel state consistency for the counter update -- the claim with zero prior coverage.

The counter has no Parameter .grad; the optimizer IS the in-place state update, applied after the
weight-gradient is all-reduced. So "every replica holds byte-identical packed state after the same
steps" is a real invariant that nothing tested. Two regimes exercise it, with different failure
modes that were both uncovered:

  * proxy RMS: the denominator is built from this rank's RAW grad_out/x (not the averaged grad_w),
    so without an extra all-reduce it differs per rank -> states silently DIVERGE. (This is the
    real bug; exact/lagged read the stat from the already-averaged grad_w and are fine.)
  * decimation: the update fires every _dec_period steps; the period is chosen from the flip-rate.
    Because flips are counted on the AVERAGED grad_w, the period stays in lockstep across ranks on
    its own -- this guards that it stays bit-identical (and does not deadlock) when it engages.

Runs on CPU via gloo (no GPU). Skipped if multiprocessing-spawn / distributed isn't usable here.
"""
import multiprocessing as mp

import pytest
import torch


def _worker(rank, world, sync_file, tmp_state, mode):
    import torch.distributed as dist
    from memory_native import PackedRMSCounterLinear

    dist.init_process_group("gloo", rank=rank, world_size=world,
                            init_method=f"file://{sync_file}")
    torch.manual_seed(0)  # identical model init + phase-aligned SR RNG on every rank
    if mode == "proxy":
        lay = PackedRMSCounterLinear(16, 12, C=11, lr=0.03, lr_scale=1e-3,
                                     rms_mode="proxy").train()
        steps, teacher = 60, None
    else:  # decimation, with a fixed teacher so the layer stabilizes and the period actually grows
        lay = PackedRMSCounterLinear(16, 12, C=11, lr=0.02, lr_scale=1e-3,
                                     decimate_updates=True).train()
        steps = 150
        teacher = (torch.randint(-1, 2, (12, 16)).float() * 0.25)
    g = torch.Generator().manual_seed(100 + rank)  # per-rank-DIFFERENT data
    max_period = 1
    for _ in range(steps):
        x = torch.randn(8, 16, generator=g)
        y = (x @ teacher.t()) if teacher is not None else torch.randn(8, 12, generator=g)
        (lay(x) - y).pow(2).mean().backward()
        max_period = max(max_period, lay._dec_period)
    if rank == 0:
        torch.save(lay.state.clone(), tmp_state)
    dist.barrier()
    ref = torch.load(tmp_state)
    assert torch.equal(lay.state, ref), f"rank {rank} state diverged from rank 0 (mode={mode})"
    if mode == "decimation":
        assert max_period > 1, "decimation never engaged -> test is vacuous"
    dist.barrier()
    dist.destroy_process_group()


def _run(mode, tmp_path):
    if not getattr(torch.distributed, "is_available", lambda: False)():
        pytest.skip("torch.distributed unavailable")
    ctx = mp.get_context("spawn")
    world = 2
    sync_file = str(tmp_path / f"pg_{mode}")
    tmp_state = str(tmp_path / f"rank0_{mode}.pt")
    procs = [ctx.Process(target=_worker, args=(r, world, sync_file, tmp_state, mode))
             for r in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=150)         # hard timeout: a regressed collective would hang here
    alive = [p for p in procs if p.is_alive()]
    if alive:
        for p in alive:
            p.terminate()
        pytest.fail(f"DDP {mode} deadlocked (asymmetric collective)")
    for p in procs:
        assert p.exitcode == 0, f"worker exited {p.exitcode} (state diverged or error), mode={mode}"


def test_ddp_proxy_rms_states_stay_bit_identical(tmp_path):
    """Catches the real bug: proxy denominator must be reduced or ranks diverge."""
    _run("proxy", tmp_path)


def test_ddp_decimation_engages_and_stays_bit_identical(tmp_path):
    """Decimation period grows (>1) yet ranks stay in lockstep and bit-identical."""
    _run("decimation", tmp_path)
