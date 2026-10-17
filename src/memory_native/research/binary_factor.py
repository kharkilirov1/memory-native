"""Research CPU binary factors with packed finite-state training.

Independent implementation from equations, not copied LittleBit source.  The
Scale-Binary-Scale-Binary-Scale architecture and Joint-ITQ initialization are
attributed to LittleBit / LittleBit-2 (arXiv:2506.13771, 2603.00042).  This module
is NOT an implementation of the authors' complete QAT/distillation recipe.

A factor retains six bits per coefficient (64 states), a row RMS, and an update
index.  State z in [0,63] reads as sign(z-31.5); latent q=(z-31.5)/16 is used only
for the update.  There is no persistent floating-point factor/master matrix.
Shared h/l/g scales are ordinary Parameters and need an external optimizer.
Forward decodes factor signs; no full [out,in] weight or its gradient is formed.
This is eager, first-order, CPU research code, not a fused GPU kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn

__all__ = ["BinaryFactorInit", "initialize_binary_factors", "NativeBinaryFactorLinear",
           "QATBinaryFactorLinear", "PackedBinaryLinear", "pack6", "unpack6",
           "row_rms_step", "tensor_state_bytes"]
_QUANTUM = 1.0 / 16.0
_MAX_LATENT = 31.5 / 16.0


def _integer(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_cpu(x: torch.Tensor, name: str) -> None:
    if x.device.type != "cpu" or x.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must be a float32/float64 CPU tensor")
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} must be finite")


def pack6(codes: torch.Tensor) -> torch.Tensor:
    """Losslessly pack four integer codes into three bytes, with no padding."""
    if codes.ndim != 2 or codes.shape[1] == 0 or codes.shape[1] % 4:
        raise ValueError("codes must be a nonempty matrix with width divisible by four")
    if codes.dtype not in (torch.uint8, torch.int16, torch.int32, torch.int64):
        raise ValueError("codes must be integer tensors")
    if bool(((codes < 0) | (codes > 63)).any()):
        raise ValueError("six-bit codes must lie in [0,63]")
    a = codes.to(torch.int32).reshape(codes.shape[0], -1, 4)
    word = a[..., 0] | (a[..., 1] << 6) | (a[..., 2] << 12) | (a[..., 3] << 18)
    return torch.stack((word & 255, (word >> 8) & 255, (word >> 16) & 255), -1).to(torch.uint8).flatten(1)


def unpack6(packed: torch.Tensor) -> torch.Tensor:
    if packed.ndim != 2 or packed.dtype != torch.uint8 or packed.shape[1] == 0 or packed.shape[1] % 3:
        raise ValueError("packed must be a uint8 matrix with three-byte groups")
    b = packed.to(torch.int32).reshape(packed.shape[0], -1, 3)
    word = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16)
    return torch.stack(tuple((word >> shift) & 63 for shift in (0, 6, 12, 18)), -1).to(torch.uint8).flatten(1)


def _sign(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def _codes(latent: torch.Tensor) -> torch.Tensor:
    # The half-integer lattice excludes zero. Preserve the original sign even
    # for subnormal values that round to exactly 31.5 in the index conversion.
    z = (latent.double() / _QUANTUM + 31.5).round().clamp(0, 63)
    z = torch.where(latent >= 0, z.clamp_min(32), z.clamp_max(31))
    return z.to(torch.uint8)


def _latent(z: torch.Tensor) -> torch.Tensor:
    return (z.float() - 31.5) * _QUANTUM


@dataclass
class BinaryFactorInit:
    """Temporary initializer. Donor and SVD matrices are NOT retained by layers."""
    left_latent: torch.Tensor  # [N,r], dimensionless
    right_latent: torch.Tensor  # [r,K], dimensionless
    row_scale: torch.Tensor
    latent_scale: torch.Tensor
    column_scale: torch.Tensor
    metadata: dict[str, Any]

    def dense_weight(self) -> torch.Tensor:
        """Diagnostic only; not used in native forward/backward."""
        return (self.row_scale[:, None] * _sign(self.left_latent) * self.latent_scale) @ (_sign(self.right_latent) * self.column_scale)


def _magnitude_factors(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Best rank-one magnitude envelope, independently evaluated from an SVD.
    u, s, vh = torch.linalg.svd(a.abs(), full_matrices=False)
    root = s[0].clamp_min(0).sqrt()
    return (u[:, 0].abs() * root).clamp_min(1e-12), (vh[0].abs() * root).clamp_min(1e-12)


@torch.no_grad()
def initialize_binary_factors(weight: torch.Tensor, rank: int, *, method: str = "joint_itq",
                              iterations: int = 30, seed: int = 0,
                              guarded: bool = False, calibration: torch.Tensor | None = None) -> BinaryFactorInit:
    """SVD -> optional internal rotation -> binary signs and three scale vectors.

    ITQ minimizes a factor-space surrogate, not end-to-end loss. ``guarded``
    chooses among identity, seeded random rotation, and final ITQ using weight
    reconstruction (or supplied calibration activations). It guarantees only
    not worsening this finite initializer-selection objective, not held-out loss.
    Complete SVD / temporary dense scoring are initialization-only CPU work.
    No test data may be passed as calibration.
    """
    _finite_cpu(weight, "weight")
    rank = _integer("rank", rank)
    iterations = _integer("iterations", iterations, 0)
    seed = _integer("seed", seed, 0)
    if weight.ndim != 2 or rank > min(weight.shape) or weight.shape[1] % 4 or rank % 4:
        raise ValueError("expected weight[N,K], rank<=min(N,K), and K/rank divisible by four")
    if method not in {"svd", "random", "joint_itq"}:
        raise ValueError("method must be svd, random or joint_itq")
    if guarded and method != "joint_itq":
        raise ValueError("guarded selection is available only with joint_itq")
    w = weight.detach().double()
    if calibration is not None:
        _finite_cpu(calibration, "calibration")
        if calibration.ndim != 2 or calibration.shape[1] != w.shape[1] or not calibration.shape[0]:
            raise ValueError("calibration must have shape [nonempty rows,K]")
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    left = u[:, :rank] * s[:rank].sqrt()
    right_t = vh[:rank].T * s[:rank].sqrt()
    eye = torch.eye(rank, dtype=w.dtype)
    generator = torch.Generator().manual_seed(seed)
    rotation, triangular = torch.linalg.qr(torch.randn(rank, rank, generator=generator, dtype=w.dtype))
    rotation = rotation * _sign(triangular.diagonal())
    candidates = [("svd", eye)]
    history = []
    if method != "svd":
        candidates.append(("random", rotation.clone()))
    if method == "joint_itq":
        joint = torch.cat((left, right_t))
        for _ in range(iterations):
            bits = _sign(joint @ rotation)
            a, _, bh = torch.linalg.svd(joint.T @ bits, full_matrices=False)
            rotation = a @ bh
            history.append(float((_sign(joint @ rotation) - joint @ rotation).square().sum()))
        candidates.append(("joint_itq", rotation))
    scored = []
    inits = []
    for name, r in candidates:
        l = left @ r
        v = (right_t @ r).T
        h, lu = _magnitude_factors(l)
        lv, g = _magnitude_factors(v)
        ql = l / (h[:, None] * lu)
        qr = v / (lv[:, None] * g)
        init = BinaryFactorInit(_latent(_codes(ql)), _latent(_codes(qr)), h.float(), (lu*lv).float(), g.float(), {})
        error = init.dense_weight().double() - w
        if calibration is not None:
            residual = calibration.double() @ error.T
            denominator = (calibration.double() @ w.T).square().sum()
        else:
            residual, denominator = error, w.square().sum()
        scored.append({"name": name, "relative_mse": float(residual.square().sum()/denominator.clamp_min(1e-30))})
        inits.append(init)
    pick = min(range(len(scored)), key=lambda j: scored[j]["relative_mse"]) if guarded else len(scored)-1
    init = inits[pick]
    init.metadata = {"method": method, "selected": scored[pick]["name"], "guarded": guarded,
                     "rank": rank, "iterations": iterations, "seed": seed,
                     "selection_metric": "calibration_output" if calibration is not None else "weight_frobenius",
                     "candidates": scored, "itq_objective": history,
                     "truncation_relative_mse": float(s[rank:].square().sum()/s.square().sum().clamp_min(1e-30))}
    return init


class _CounterApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bank, tap):
        if bank._outstanding_forward:
            raise RuntimeError("counter reuse/gradient accumulation requires a separate scheduler")
        bank._validate_input(x)
        bank._outstanding_forward = True
        ctx.bank = bank
        ctx.controls = bank._controls()
        ctx.save_for_backward(x, bank.codes, bank.rms)
        return x @ bank.visible(x.dtype).T

    @staticmethod
    def backward(ctx, go):
        if torch.is_grad_enabled():
            raise RuntimeError("higher-order gradients are unsupported")
        bank = ctx.bank
        x, _, _ = ctx.saved_tensors  # also validates in-place version counters
        if bank._controls() != ctx.controls:
            raise RuntimeError("counter controls/state changed between forward and backward")
        _finite_cpu(go, "gradient")
        go2, x2 = go.reshape(-1, go.shape[-1]), x.reshape(-1, x.shape[-1])
        # The input derivative is computed BEFORE either buffer is changed.
        gx = (go2 @ bank.visible(go.dtype)).reshape_as(x)
        g = go2.T @ x2
        bank.update_from_gradient(g)
        bank._outstanding_forward = False
        return gx, None, None


class PackedBinaryLinear(nn.Module):
    """Six-bit binary sign propensity; row-RMS statistics are additional state."""
    def __init__(self, latent: torch.Tensor, *, lr: float = .003, beta: float = .9,
                 eps: float = 1e-3, seed: int = 0):
        super().__init__()
        _finite_cpu(latent, "latent")
        if latent.ndim != 2 or min(latent.shape) < 1 or latent.shape[1] % 4:
            raise ValueError("latent must be a nonempty matrix with width divisible by four")
        if not math.isfinite(lr) or lr < 0 or not math.isfinite(beta) or not 0 <= beta < 1 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("invalid optimizer controls")
        self.out_features, self.in_features = latent.shape
        self.lr, self.beta, self.eps = float(lr), float(beta), float(eps)
        self.seed = _integer("seed", seed, 0)
        self.update_enabled, self._outstanding_forward = True, False
        self.register_buffer("codes", pack6(_codes(latent)))
        self.register_buffer("rms", torch.zeros((self.out_features, 1)))
        self.register_buffer("steps", torch.zeros((), dtype=torch.int64))
        self.register_buffer("flips", torch.zeros((), dtype=torch.int64))

    def _validate_input(self, x):
        _finite_cpu(x, "input")
        if self.codes.device.type != "cpu" or self.rms.dtype != torch.float32:
            raise ValueError("counter buffers must stay on CPU with fp32 RMS")
        if x.ndim < 1 or x.shape[-1] != self.in_features or x.numel() == 0:
            raise ValueError("invalid input shape")

    def _controls(self):
        return self.lr, self.beta, self.eps, self.seed, int(self.steps), self.training, self.update_enabled

    def visible(self, dtype=torch.float32):
        return torch.where(unpack6(self.codes) >= 32, 1., -1.).to(dtype)

    def latent(self):
        return _latent(unpack6(self.codes))

    @torch.no_grad()
    def update_from_gradient(self, grad):
        _finite_cpu(grad, "factor gradient")
        if tuple(grad.shape) != (self.out_features, self.in_features):
            raise ValueError("factor gradient shape mismatch")
        if not math.isfinite(self.lr) or self.lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        g = grad.float()
        old = unpack6(self.codes)
        rms = self.beta*self.rms + (1-self.beta)*g.square().mean(1, keepdim=True)
        position = old.float() - (self.lr/_QUANTUM)*g/rms.sqrt().clamp_min(self.eps)
        if not bool(torch.isfinite(position).all() & torch.isfinite(rms).all()):
            raise FloatingPointError("nonfinite proposal; bank not modified")
        low = position.floor()
        gen = torch.Generator().manual_seed((self.seed + 1_000_003*int(self.steps)) % (2**63-1))
        new = (low + (torch.rand(position.shape, generator=gen) < (position-low))).clamp(0,63).to(torch.uint8)
        packed = pack6(new)
        flips = ((new >= 32) != (old >= 32)).sum()
        self.rms.copy_(rms)
        self.codes.copy_(packed)
        self.flips.add_(flips)
        self.steps.add_(1)

    def forward(self, x):
        self._validate_input(x)
        if self.training and self.update_enabled and torch.is_grad_enabled():
            tap = torch.zeros((), requires_grad=True, dtype=x.dtype)
            return _CounterApply.apply(x, self, tap)
        return x @ self.visible(x.dtype).T

    def get_extra_state(self):
        if self._outstanding_forward:
            raise RuntimeError("checkpoint only between complete steps")
        return {"version": 1, "shape": (self.out_features,self.in_features),
                "lr": self.lr, "beta": self.beta, "eps": self.eps, "seed": self.seed,
                "update_enabled": self.update_enabled}

    def set_extra_state(self, state):
        self.lr, self.beta, self.eps, self.seed = state["lr"], state["beta"], state["eps"], state["seed"]
        self.update_enabled = state["update_enabled"]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self._outstanding_forward:
            raise RuntimeError("cannot load into a pending counter")
        state = state_dict.get(prefix+"_extra_state")
        if not isinstance(state,dict) or state.get("version") != 1 or state.get("shape") != (self.out_features,self.in_features):
            raise RuntimeError("missing or incompatible binary counter metadata")
        for key in ("lr","beta","eps"):
            if key not in state or not isinstance(state[key],(int,float)) or not math.isfinite(state[key]):
                raise RuntimeError("invalid counter metadata")
        if state["lr"] < 0 or not 0 <= state["beta"] < 1 or state["eps"] <= 0 or type(state.get("seed")) is not int or state["seed"]<0 or type(state.get("update_enabled")) is not bool:
            raise RuntimeError("invalid counter metadata")
        for name, target in (("codes", self.codes),("rms",self.rms),("steps",self.steps),("flips",self.flips)):
            value = state_dict.get(prefix+name)
            if not torch.is_tensor(value) or value.shape != target.shape or value.dtype != target.dtype:
                raise RuntimeError(f"invalid {name} buffer")
            if name != "codes" and (not bool(torch.isfinite(value).all()) or bool((value<0).any())):
                raise RuntimeError(f"invalid {name} buffer values")
        super()._load_from_state_dict(state_dict,prefix,*args,**kwargs)


class NativeBinaryFactorLinear(nn.Module):
    """Two factor matmuls, 6-bit factor state, three shared trainable scales.

    No full matrix W in forward/backward. Temporary decoded signs and factor
    gradients remain. Bias and h/l/g use conventional autograd/optimizer state.
    """
    def __init__(self, init: BinaryFactorInit, *, lr: float = .003, seed: int = 0, bias: torch.Tensor | None = None):
        super().__init__()
        self.left = PackedBinaryLinear(init.left_latent, lr=lr, seed=seed)
        self.right = PackedBinaryLinear(init.right_latent, lr=lr, seed=seed+65537)
        self.in_features, self.out_features = self.right.in_features, self.left.out_features
        self.rank = self.left.in_features
        self.h = nn.Parameter(init.row_scale.clone())
        self.ell = nn.Parameter(init.latent_scale.clone())
        self.g = nn.Parameter(init.column_scale.clone())
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None
        self._validate_scales()

    def _validate_scales(self):
        if self.right.out_features != self.rank or self.h.shape != (self.out_features,) or self.ell.shape != (self.rank,) or self.g.shape != (self.in_features,):
            raise ValueError("initializer dimension mismatch")
        if self.bias is not None and self.bias.shape != (self.out_features,):
            raise ValueError("bias dimension mismatch")
        for t in (self.h,self.ell,self.g):
            _finite_cpu(t,"scale")

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, *, lr: float = .003, seed: int = 0, **init_options):
        init = initialize_binary_factors(linear.weight, rank, seed=seed, **init_options)
        return cls(init, lr=lr, seed=seed, bias=linear.bias)

    def forward(self,x):
        y = self.left(self.right(x*self.g)*self.ell)*self.h
        return y if self.bias is None else y+self.bias

    def set_lr(self, lr: float):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        if self.left._outstanding_forward or self.right._outstanding_forward:
            raise RuntimeError("set lr only between steps")
        self.left.lr = self.right.lr = float(lr)

    @torch.no_grad()
    def dense_weight(self):
        return (self.h[:,None]*self.left.visible()*self.ell) @ (self.right.visible()*self.g)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, factor_bits=6"


class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q):
        ctx.save_for_backward(q)
        return _sign(q)

    @staticmethod
    def backward(ctx,g):
        (q,) = ctx.saved_tensors
        return g*(q.abs()<=_MAX_LATENT)


class QATBinaryFactorLinear(nn.Module):
    """Matched architecture/control with continuous latent factors and clipped STE.

    This is NOT the full LittleBit-2 SmoothSign/KD recipe. Factors start from
    exactly the same six-bit-decoded propensities as the native control.
    """
    def __init__(self, init: BinaryFactorInit, *, bias: torch.Tensor | None = None):
        super().__init__()
        self.left = nn.Parameter(_latent(_codes(init.left_latent)))
        self.right = nn.Parameter(_latent(_codes(init.right_latent)))
        self.h = nn.Parameter(init.row_scale.clone())
        self.ell = nn.Parameter(init.latent_scale.clone())
        self.g = nn.Parameter(init.column_scale.clone())
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None
        self.in_features, self.out_features, self.rank = self.right.shape[1], self.left.shape[0], self.left.shape[1]
        self.register_buffer("rms_left",torch.zeros((self.out_features,1)))
        self.register_buffer("rms_right",torch.zeros((self.rank,1)))

    def forward(self,x):
        y=(((x*self.g) @ _SignSTE.apply(self.right).T)*self.ell) @ _SignSTE.apply(self.left).T
        y=y*self.h
        return y if self.bias is None else y+self.bias

    @torch.no_grad()
    def clamp_latents(self):
        self.left.clamp_(-_MAX_LATENT,_MAX_LATENT)
        self.right.clamp_(-_MAX_LATENT,_MAX_LATENT)

    def scale_parameters(self):
        return [self.h,self.ell,self.g]+([] if self.bias is None else [self.bias])


@torch.no_grad()
def row_rms_step(model: QATBinaryFactorLinear, lr: float, beta: float = .9, eps: float = 1e-3):
    """Continuous control for the same row-RMS rule, without state rounding."""
    if not math.isfinite(lr) or lr < 0:
        raise ValueError("invalid lr")
    for parameter, rms in ((model.left,model.rms_left),(model.right,model.rms_right)):
        if parameter.grad is None:
            raise RuntimeError("missing QAT factor gradient")
        rms.mul_(beta).add_(parameter.grad.square().mean(1,keepdim=True),alpha=1-beta)
        parameter.add_(parameter.grad/rms.sqrt().clamp_min(eps),alpha=-lr)
    model.clamp_latents()


def tensor_state_bytes(model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict[str,int]:
    """Persistent tensor accounting, NOT allocator peak/RSS; grads are separate."""
    params=list(model.parameters()); buffers=list(model.buffers())
    opt = [] if optimizer is None else [x for s in optimizer.state.values() for x in s.values() if torch.is_tensor(x)]
    def count(values):
        seen=set();total=0
        for x in values:
            key=(x.device,x.untyped_storage().data_ptr())
            if key not in seen:
                total+=x.untyped_storage().nbytes();seen.add(key)
        return total
    return {"parameters":count(params),"buffers":count(buffers),"optimizer":count(opt),
            "retained_gradients":count([p.grad for p in params if p.grad is not None]),
            "persistent_total":count(params+buffers+opt)}


def _pack_signs(signs: torch.Tensor) -> torch.Tensor:
    bits=(signs>=0).to(torch.int64)
    pad=(-bits.shape[1])%8
    if pad:
        bits=torch.nn.functional.pad(bits,(0,pad))
    bits=bits.reshape(bits.shape[0],-1,8)
    shifts=torch.arange(8)
    return (bits<<shifts).sum(-1).to(torch.uint8)


def _unpack_signs(packed: torch.Tensor,width: int) -> torch.Tensor:
    bits=((packed.to(torch.int64)[...,None]>>torch.arange(8))&1).flatten(1)[:,:width]
    return bits.float()*2-1


class BinaryFactorInference(nn.Module):
    """One-bit factor artifact; reference decoding, not a fast inference kernel."""
    def __init__(self, artifact: dict[str, Any]):
        super().__init__()
        if artifact.get("format") != "native_binary_factor_inference_v1":
            raise ValueError("unsupported inference artifact")
        self.in_features=_integer("in_features",artifact["in_features"])
        self.out_features=_integer("out_features",artifact["out_features"])
        self.rank=_integer("rank",artifact["rank"])
        expected={"left_bits":(self.out_features,math.ceil(self.rank/8)),
                  "right_bits":(self.rank,math.ceil(self.in_features/8)),
                  "h":(self.out_features,),"ell":(self.rank,),"g":(self.in_features,)}
        for name,shape in expected.items():
            value=artifact[name]
            if not torch.is_tensor(value) or tuple(value.shape)!=shape or value.device.type!="cpu":
                raise ValueError(f"invalid {name}")
            if name.endswith("bits"):
                if value.dtype!=torch.uint8:
                    raise ValueError("packed signs require uint8")
            else:
                _finite_cpu(value,name)
            self.register_buffer(name,value.detach().clone())
        b=artifact.get("bias")
        if b is not None:
            _finite_cpu(b,"bias")
            if b.shape!=(self.out_features,):
                raise ValueError("invalid bias shape")
        self.register_buffer("bias",None if b is None else b.detach().clone())

    def forward(self,x):
        _finite_cpu(x,"input")
        if x.shape[-1]!=self.in_features:
            raise ValueError("input width mismatch")
        right=_unpack_signs(self.right_bits,self.in_features).to(x.dtype)
        left=_unpack_signs(self.left_bits,self.rank).to(x.dtype)
        y=(((x*self.g)@right.T)*self.ell)@left.T
        y=y*self.h
        return y if self.bias is None else y+self.bias


@torch.no_grad()
def export_binary_factors(layer: NativeBinaryFactorLinear) -> dict[str, Any]:
    """Inference-only export: does not claim to resume the removed counters."""
    if layer.left._outstanding_forward or layer.right._outstanding_forward:
        raise RuntimeError("export only between complete steps")
    result={"format":"native_binary_factor_inference_v1","in_features":layer.in_features,
            "out_features":layer.out_features,"rank":layer.rank,
            "left_bits":_pack_signs(layer.left.visible()),"right_bits":_pack_signs(layer.right.visible()),
            "h":layer.h.detach().clone(),"ell":layer.ell.detach().clone(),"g":layer.g.detach().clone(),
            "bias":None if layer.bias is None else layer.bias.detach().clone()}
    return result

__all__ += ["BinaryFactorInference", "export_binary_factors"]
