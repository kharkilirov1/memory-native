"""M9 — Multi-Token Prediction (MTP).

Pins:
  (a) shapes: n_pred heads produce n_pred logits tensors, each (b, t, vocab).
  (b) the loss correctly shifts targets per head (head j targets = inputs shifted by j+1, i.e.
      next-token targets shifted by j) AND masks positions past the end (no wraparound).
  (c) n_pred==1 reduces EXACTLY to the standard single-head next-token loss (parity).
  (d) training reduces loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_native.mtp import MultiTokenHead, mtp_loss, shift_targets, IGNORE_INDEX


def test_n_pred_heads_produce_n_pred_logits():
    torch.manual_seed(0)
    b, t, d, vocab = 2, 7, 16, 11
    head = MultiTokenHead(d, vocab, n_pred=4)
    h = torch.randn(b, t, d)
    logits_list = head(h)
    assert isinstance(logits_list, list)
    assert len(logits_list) == 4
    for lg in logits_list:
        assert lg.shape == (b, t, vocab)


def test_shift_targets_per_head_and_boundary_mask():
    """Head j's target at position i is targets[i+j]; the last j positions are masked (not wrapped).
    targets here are the standard next-token targets (targets[i] == idx[i+1])."""
    targets = torch.arange(1, 7).reshape(1, 6)            # [[1,2,3,4,5,6]]
    # j=0 unchanged
    assert torch.equal(shift_targets(targets, 0), targets)
    # j=1: shift left by 1, last 1 masked
    exp1 = torch.tensor([[2, 3, 4, 5, 6, IGNORE_INDEX]])
    assert torch.equal(shift_targets(targets, 1), exp1)
    # j=3: shift left by 3, last 3 masked
    exp3 = torch.tensor([[4, 5, 6, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]])
    assert torch.equal(shift_targets(targets, 3), exp3)
    # NO wraparound: masked entries are exactly IGNORE_INDEX, never a token id
    s3 = shift_targets(targets, 3)
    assert (s3[:, -3:] == IGNORE_INDEX).all()
    assert (s3[:, :3] >= 0).all()


def test_boundary_mask_count_per_head():
    """Head j has exactly j masked (invalid) positions per row near the end, and (t-j) valid ones.
    This is the boundary-masking correctness pin: fewer valid positions near the sequence end."""
    b, t = 3, 8
    targets = torch.randint(0, 20, (b, t))
    for j in range(t + 2):
        s = shift_targets(targets, j)
        assert s.shape == (b, t)
        n_masked = int((s == IGNORE_INDEX).sum())
        expected_masked = min(j, t) * b
        assert n_masked == expected_masked, f"j={j}: masked {n_masked} != {expected_masked}"
        # valid (unmasked) positions equal the original targets shifted left by j
        if j < t:
            assert torch.equal(s[:, : t - j], targets[:, j:])


def test_n_pred_1_parity_with_standard_next_token_loss():
    """n_pred==1 must reduce EXACTLY to F.cross_entropy(logits, targets) — no position masked."""
    torch.manual_seed(0)
    b, t, d, vocab = 4, 9, 16, 13
    head = MultiTokenHead(d, vocab, n_pred=1)
    h = torch.randn(b, t, d)
    targets = torch.randint(0, vocab, (b, t))

    mean_loss, per_head = head.loss(h, targets)
    logits = head(h)[0]
    ref = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))

    assert len(per_head) == 1
    assert torch.allclose(mean_loss, ref, atol=1e-6)
    assert torch.allclose(per_head[0], ref, atol=1e-6)


def test_mtp_loss_equals_mean_of_per_head_manual():
    """mtp_loss is the mean over heads of the boundary-masked per-head CE — check head-by-head."""
    torch.manual_seed(1)
    b, t, d, vocab = 2, 6, 8, 10
    n_pred = 3
    head = MultiTokenHead(d, vocab, n_pred=n_pred)
    h = torch.randn(b, t, d)
    targets = torch.randint(0, vocab, (b, t))

    mean_loss, per_head = mtp_loss(head(h), targets)
    logits = head(h)
    manual = []
    for j in range(n_pred):
        tgt = shift_targets(targets, j)
        manual.append(F.cross_entropy(
            logits[j].reshape(-1, vocab), tgt.reshape(-1), ignore_index=IGNORE_INDEX))
    assert len(per_head) == n_pred
    for got, exp in zip(per_head, manual):
        assert torch.allclose(got, exp, atol=1e-6)
    assert torch.allclose(mean_loss, torch.stack(manual).mean(), atol=1e-6)


def test_all_masked_head_when_j_ge_t():
    """If j >= t every target is past the end -> fully masked (and CE on a fully-ignored tensor
    must not crash the per-head computation in practice we keep n_pred < t, but pin the helper)."""
    targets = torch.randint(0, 5, (2, 4))
    s = shift_targets(targets, 4)
    assert (s == IGNORE_INDEX).all()
    s2 = shift_targets(targets, 10)
    assert (s2 == IGNORE_INDEX).all()


def test_training_reduces_loss():
    """A tiny optimisable head over a fixed hidden state must drive the MTP loss down."""
    torch.manual_seed(0)
    b, t, d, vocab = 8, 12, 24, 17
    head = MultiTokenHead(d, vocab, n_pred=3)
    h = torch.randn(b, t, d)
    targets = torch.randint(0, vocab, (b, t))
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)

    loss0, _ = head.loss(h, targets)
    for _ in range(200):
        loss, _ = head.loss(h, targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    loss1, _ = head.loss(h, targets)
    assert loss1.item() < loss0.item() - 0.1


def test_tied_aux_heads_add_no_params():
    """With tie_embedding, every head shares the embedding weight -> only biases (here none)."""
    d, vocab = 16, 20
    emb = nn.Embedding(vocab, d)
    primary = nn.Linear(d, vocab, bias=False)
    primary.weight = emb.weight
    head = MultiTokenHead(d, vocab, n_pred=4, primary_head=primary,
                          embedding=emb, tie_embedding=True)
    head_params = {id(p) for p in head.parameters()}
    # the only weight should be the shared embedding weight
    assert id(emb.weight) in head_params
    # no aux head introduces an independent weight tensor
    for lin in head.aux:
        assert lin.weight is emb.weight
