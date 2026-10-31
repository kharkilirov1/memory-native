"""Mechanism-level tests of PackedNA/MXNLinear inside a tiny pre-LN transformer block.

Replaces the old single-seed char-LM "guessing game": every test here is deterministic
(fixed seeds, deterministic algorithms, eager mode), CPU-fast, and asserts a property a
broken implementation would fail.  Each property has a null twin that proves the check
can fail (a deliberately broken variant or the legacy ``gap_mode="min"``).

Block: LN -> causal single-head attention (qkv, proj) -> +res -> LN -> MLP (fc, out) -> +res,
every Linear bias-free.  LayerNorms are frozen identity so the only learnable state is
the four packed layers, which update inside their own backward (no optimizer).

Formats: "wide_c3" = PackedNA5 ladder {0,.5,3} C=3; "e2m1_c5" = PackedNA6 ladder
{0,.5,1,6} C=5.  Measurement forwards run under torch.no_grad() (exactly one forward
per backward is enforced by MXNLinear).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mx6.layer import MXNLinear  # noqa: E402
from mx6.packed_na import PackedNA  # noqa: E402

D, T, B = 32, 8, 8
LIN = ("qkv", "proj", "fc", "out")
FMT_SPECS = {"wide_c3": ([0.0, 0.5, 3.0], 3, 1.5),
             "e2m1_c5": ([0.0, 0.5, 1.0, 6.0], 5, 3.0)}
FMTS = list(FMT_SPECS)


@pytest.fixture(autouse=True)
def _deterministic():
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(prev)


# ── architecture ──────────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, make, causal: bool = True):
        super().__init__()
        self.causal = causal
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv, self.proj = make(D, 3 * D), make(D, D)
        self.fc, self.out = make(D, 4 * D), make(4 * D, D)
        for ln in (self.ln1, self.ln2):
            for p in ln.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        t = x.shape[1]
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        if self.causal:
            mask = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float("-inf"))
        x = x + self.proj(att.softmax(dim=-1) @ v)
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


def make_fmt(name: str, gap_mode: str = "directional") -> PackedNA:
    mags, C, _ = FMT_SPECS[name]
    return PackedNA(mags, C, gap_mode=gap_mode)


def mxn_block(name: str, seed: int, lr: float = 0.02, gap_mode: str = "directional",
              causal: bool = True) -> Block:
    fmt, hr = make_fmt(name, gap_mode), FMT_SPECS[name][2]
    torch.manual_seed(seed)
    return Block(lambda i, o: MXNLinear(i, o, fmt, lr=lr, scale_headroom=hr), causal)


def dense_twin(blk: Block) -> Block:
    """Dense nn.Linear block whose weights are the decoded packed weights."""
    t = Block(lambda i, o: nn.Linear(i, o, bias=False), blk.causal)
    with torch.no_grad():
        for n in LIN:
            getattr(t, n).weight.copy_(getattr(blk, n).weight())
    return t


def set_codes(layer: MXNLinear, codes: torch.Tensor) -> None:
    layer._state.copy_(layer.fmt.pack_state(codes.to(torch.uint8)))
    layer._codes_cache = None
    layer._weight_cache = None


def bump_code(layer: MXNLinear, r: int = 0, c: int = 0) -> None:
    """Move ONE code to the adjacent rung (the broken-implementation stand-in)."""
    fmt, codes = layer.fmt, layer.codes.to(torch.long).clone()
    mag = codes[r, c] // fmt.acc_levels
    mag = mag + 1 if mag < fmt.n_vals - 1 else mag - 1
    codes[r, c] = mag * fmt.acc_levels + (codes[r, c] % fmt.acc_levels)
    set_codes(layer, codes)


def make_pair(name: str, seed: int, lr: float = 0.02, gap_mode: str = "directional",
              boost: int = 2):
    """(student, dense_teacher).  The teacher's weights lie exactly on the ladder x scale
    grid (so the student can represent the solution); the student shares the teacher's
    block scales but starts from different random codes."""
    teacher_q = mxn_block(name, 1000 + seed, gap_mode="directional")
    student = mxn_block(name, 2000 + seed, lr=lr, gap_mode=gap_mode)
    for n in LIN:
        tl, sl = getattr(teacher_q, n), getattr(student, n)
        with torch.no_grad():
            tl._scale.add_(boost)                   # x2**boost: stronger target function
            tl._weight_cache = None
            sl._scale.copy_(tl._scale)
            b = 1.0 / sl.in_features ** 0.5
            w0 = torch.empty(sl.out_features, sl.in_features).uniform_(-b, b) * 2.0 ** boost
            s = sl.fmt.scale_floats(sl._scale).repeat_interleave(32, dim=1)
            idx = sl.fmt.round_idx(w0 / s)
            set_codes(sl, sl.fmt.encode(idx, torch.zeros_like(idx)))
    return student, dense_twin(teacher_q)


def eval_batch(seed: int = 999):
    return torch.randn(B, T, D, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def mse(student, teacher, x) -> float:
    return float(F.mse_loss(student(x), teacher(x)))


def train(student, teacher, steps: int, seed: int, flip: bool = False) -> None:
    """MSE regression on fresh seeded batches; updates happen inside backward."""
    torch.manual_seed(seed)
    if flip:
        flip_grads(student)
    gen = torch.Generator().manual_seed(seed + 7)
    student.train()
    for _ in range(steps):
        x = torch.randn(B, T, D, generator=gen)
        with torch.no_grad():
            y = teacher(x)
        F.mse_loss(student(x), y).backward()


def flip_grads(blk: Block) -> None:
    """Null twin: feed the layers the NEGATED weight gradient."""
    for n in LIN:
        layer = getattr(blk, n)
        orig = layer.apply_update_
        layer.apply_update_ = lambda g, _o=orig: _o(-g)


def outer_fraction(blk: Block) -> float:
    hit = tot = 0
    for n in LIN:
        layer = getattr(blk, n)
        v = layer.fmt.value_float(layer.codes).abs()
        hit += int((v == float(layer.fmt.signed.abs().max())).sum())
        tot += v.numel()
    return hit / tot


# ── 1. forward parity ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", FMTS)
def test_forward_parity_with_decoded_dense(name):
    """Packed block == dense block carrying the decoded weights; accumulator is invisible."""
    blk = mxn_block(name, 0)
    ref = dense_twin(blk)
    x = eval_batch(1)
    with torch.no_grad():
        err = float((blk(x) - ref(x)).abs().max())
        assert err < 1e-6, err

        # the accumulator residue (low part of the code) must not reach the forward pass
        layer = blk.fc
        fmt = layer.fmt
        mag = layer.codes.to(torch.long) // fmt.acc_levels
        set_codes(layer, mag * fmt.acc_levels + 2 * (fmt.C - 1))      # accumulator saturated +
        assert not torch.equal(layer.codes % fmt.acc_levels, torch.full_like(layer.codes, fmt.C - 1))
        assert float((blk(x) - ref(x)).abs().max()) < 1e-6

        # null twin: ONE code moved one rung -> the comparison must notice
        bump_code(blk.qkv)
        bad = float((blk(x) - ref(x)).abs().max())
    print(f"[fwd {name}] clean err={err:.2e}  one-code-perturbed err={bad:.2e}")
    assert bad > 1e-4, bad


# ── 2. backward parity ────────────────────────────────────────────────────────

def _backward_grads(blk, x, y, record_updates: bool):
    """Run one forward/backward; returns dx, per-layer grad_out and (packed only) grad_w."""
    outs, gw = {}, {}
    for n in LIN:
        layer = getattr(blk, n)
        def _hook(m, i, o, _n=n):
            o.register_hook(lambda g: outs.__setitem__(_n, g.clone()))
        layer.register_forward_hook(_hook)
        if record_updates:
            orig = layer.apply_update_
            layer.apply_update_ = lambda g, _o=orig, _n=n: (gw.__setitem__(_n, g.clone()), _o(g))
    x = x.clone().requires_grad_(True)
    F.mse_loss(blk(x), y).backward()
    return x.grad, outs, gw


def _rel(a, b) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@pytest.mark.parametrize("name", FMTS)
def test_backward_parity_with_decoded_dense(name):
    """dx, grad_out of every layer and the weight gradient handed to apply_update_ match
    the dense reference (built from the pre-update decoded weights)."""
    blk = mxn_block(name, 0)
    ref = dense_twin(blk)               # built BEFORE backward: updates happen inside it
    x, y = eval_batch(2), eval_batch(3)
    blk.train()
    dx, outs, gw = _backward_grads(blk, x, y, record_updates=True)
    dx_r, outs_r, _ = _backward_grads(ref, x, y, record_updates=False)
    errs = {"dx": _rel(dx, dx_r)}
    for n in LIN:
        errs[f"gout_{n}"] = _rel(outs[n], outs_r[n])
        errs[f"gw_{n}"] = _rel(gw[n], getattr(ref, n).weight.grad)
    print(f"[bwd {name}] max rel err = {max(errs.values()):.2e}")
    assert max(errs.values()) < 1e-4, errs
    assert all(not getattr(blk, n)._pending for n in LIN)      # exactly one fwd per bwd

    # null twin: reference with ONE code perturbed -> mismatch must be detected
    blk2 = mxn_block(name, 0)
    ref2 = dense_twin(blk2)
    with torch.no_grad():
        w = ref2.fc.weight
        w[0, :8] += 1.0 * float(blk2.fc.fmt.scale_floats(blk2.fc._scale).max())
    blk2.train()
    dx2, _, gw2 = _backward_grads(blk2, x, y, record_updates=True)
    dx2_r, _, _ = _backward_grads(ref2, x, y, record_updates=False)
    bad = _rel(dx2, dx2_r)
    print(f"[bwd {name}] perturbed-reference dx rel err = {bad:.2e}")
    assert bad > 1e-3, bad


# ── 3. update direction ───────────────────────────────────────────────────────

SEEDS = (0, 1, 2)
LR = 0.005


@pytest.mark.parametrize("name", FMTS)
def test_one_update_step_lowers_loss_and_flipped_gradient_raises_it(name):
    x = eval_batch(4)
    r_ok, r_flip = [], []
    for seed in SEEDS:
        st, te = make_pair(name, seed, lr=0.05)
        l0 = mse(st, te, x)
        train(st, te, 1, seed)
        r_ok.append(mse(st, te, x) / l0)
        st, te = make_pair(name, seed, lr=0.05)
        train(st, te, 1, seed, flip=True)
        r_flip.append(mse(st, te, x) / l0)
    print(f"[dir {name}] after/before: normal={[round(r, 3) for r in r_ok]} "
          f"flipped={[round(r, 3) for r in r_flip]}")
    assert all(r < 1.0 for r in r_ok), r_ok
    assert all(r > 1.0 for r in r_flip), r_flip


# ── 4. teacher-student with a known (exactly representable) solution ──────────

TS_STEPS = 50
TS_RATIO_MAX = {"wide_c3": 0.75, "e2m1_c5": 0.60}     # calibrated, see report
_cache: dict = {}


def trained(name: str, gap_mode: str, seed: int, lr: float = LR, steps: int = TS_STEPS):
    key = (name, gap_mode, seed, lr, steps)
    if key not in _cache:
        st, te = make_pair(name, seed, lr=lr, gap_mode=gap_mode)
        x = eval_batch()
        l0 = mse(st, te, x)
        train(st, te, steps, seed)
        _cache[key] = (mse(st, te, x) / l0, outer_fraction(st))
    return _cache[key]


@pytest.mark.parametrize("name", FMTS)
def test_teacher_student_recovers_teacher(name):
    ratios = [trained(name, "directional", s)[0] for s in SEEDS]
    frozen = [trained(name, "directional", s, lr=0.0, steps=10)[0] for s in SEEDS]
    print(f"[ts {name}] mse_final/mse_init: trained={[round(r, 3) for r in ratios]} "
          f"frozen(lr=0)={[round(r, 4) for r in frozen]}")
    assert all(r < TS_RATIO_MAX[name] for r in ratios), ratios
    assert all(r > 0.999 for r in frozen), frozen          # null twin: nothing learns at lr=0


# ── 5. attention-specific ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name", FMTS)
def test_causal_mask_with_quantized_qkv(name):
    """Perturbing tokens >= t0 must leave outputs < t0 bit-for-bit (to fp noise) intact."""
    t0 = 5
    x = eval_batch(5)
    x2 = x.clone()
    x2[:, t0:] += torch.randn(B, T - t0, D, generator=torch.Generator().manual_seed(6))
    with torch.no_grad():
        blk = mxn_block(name, 0, causal=True)
        a, b = blk(x), blk(x2)
        past, future = float((a - b)[:, :t0].abs().max()), float((a - b)[:, t0:].abs().max())
        # null twin: identical codes, mask removed -> the past MUST change
        nc = mxn_block(name, 0, causal=False)
        leak = float((nc(x) - nc(x2))[:, :t0].abs().max())
    print(f"[causal {name}] past diff={past:.1e} future diff={future:.2e} | no-mask past diff={leak:.2e}")
    assert past < 1e-5 and future > 1e-2          # mask holds and the test can see changes
    assert leak > 1e-3                            # null twin: without the mask it leaks


class CopyLM(nn.Module):
    """Embeddings + packed block + dense head: predict the token K positions back."""

    def __init__(self, blk: Block, vocab: int):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, D), nn.Embedding(T, D)
        self.blk, self.head = blk, nn.Linear(D, vocab)

    def forward(self, ids):
        return self.head(self.blk(self.tok(ids) + self.pos(torch.arange(ids.shape[1]))))


def copy_accuracy(name: str, seed: int, lr: float, steps: int, V: int = 4, K: int = 1) -> float:
    torch.manual_seed(seed)
    model = CopyLM(mxn_block(name, seed, lr=lr), V)
    dense_params = [p for n, p in model.named_parameters() if not n.startswith("blk.")]
    opt = torch.optim.Adam(dense_params, lr=1e-2)
    gen = torch.Generator().manual_seed(seed + 5)
    model.train()
    for _ in range(steps):
        ids = torch.randint(0, V, (32, T), generator=gen)
        loss = F.cross_entropy(model(ids)[:, K:].reshape(-1, V), ids[:, :-K].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    ev = torch.randint(0, V, (256, T), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        pred = model(ev).argmax(-1)
    return float((pred[:, K:] == ev[:, :-K]).float().mean())


COPY_STEPS = 60       # acc 1.0 for all seeds at 60 steps (0.62-0.95 at 40, so not a lucky margin)


@pytest.mark.parametrize("name", FMTS)
def test_quantized_attention_learns_lookback_copy(name):
    """y[t] = token[t-1] needs attention to the previous position (chance = 0.25).
    Null twin: same model with the packed block frozen (lr=0) stays far below."""
    accs = [copy_accuracy(name, s, 0.01, COPY_STEPS) for s in SEEDS]
    frozen = copy_accuracy(name, 0, 0.0, COPY_STEPS)
    print(f"[copy {name}] acc={[round(a, 3) for a in accs]} frozen-block acc={frozen:.3f}")
    assert all(a >= 0.95 for a in accs), accs
    assert frozen < 0.60, frozen


# ── 6. ladder occupancy under real gradients ──────────────────────────────────

OUTER_CAP = {"wide_c3": 0.02, "e2m1_c5": 0.02}          # measured 0.002 / 0.000 (legacy: 0.06-0.08)


@pytest.mark.parametrize("name", FMTS)
def test_outer_rung_occupancy_stays_bounded_vs_legacy_min(name):
    d = [trained(name, "directional", s)[1] for s in SEEDS]
    m = [trained(name, "min", s)[1] for s in SEEDS[:2]]
    print(f"[outer {name}] directional={[round(v, 3) for v in d]} legacy-min={[round(v, 3) for v in m]}")
    assert all(v < OUTER_CAP[name] and v < 0.60 for v in d), d
    # null twin: the biased legacy walk piles mass on the outer rung in a transformer too
    assert all(v > 3 * max(d) and v > 0.03 for v in m), (d, m)
