"""memory-native — finite-state counter synapses + reversible activations, in pure PyTorch.

A training method that attacks all four memory pools of training (params, optimizer state,
gradients, activations) with two independent levers, implemented on stock PyTorch (CPU/CUDA),
with no custom engine:

  * CompactCounterLinear / RMSCounterLinear -- a ternary weight whose optimizer lives inside
    a per-synapse finite-state automaton; the update is fused into backward (no FP master, no
    Adam moments, no full gradient buffer).
  * ReversibleCouplingBlock -- activations are recomputed in backward instead of stored.

See README for what is measured vs what still needs the Triton/CUDA packed kernel.
"""
from .counter import (
    C_DEFAULT,
    CompactCounterLinear,
    RMSCounterLinear,
    decode_state,
    encode_state,
    stochastic_round,
    ternary_gradient_unbiased,
)
from .reversible import ReversibleCouplingBlock, ReversibleSequential
from .packed import PackedRMSCounterLinear, pack_codes, unpack_codes
from .triton_counter import HAS_TRITON, TritonCounterLinear, triton_decode_matmul
from .baselines import TernaryQATLinear, make_linear
from .models import CONFIGS, GPT, GPTConfig
from .memory import compare_training_peak, fmt_bytes, memory_report, peak_training_memory
from .optimizers import GaLoreAdamW, LoMo, available_optimizers, build_optimizer
from .data import get_batch, load_corpus, synthetic_corpus

__version__ = "0.1.0"

__all__ = [
    "C_DEFAULT",
    "CompactCounterLinear",
    "RMSCounterLinear",
    "PackedRMSCounterLinear",
    "pack_codes",
    "unpack_codes",
    "TritonCounterLinear",
    "triton_decode_matmul",
    "HAS_TRITON",
    "encode_state",
    "decode_state",
    "stochastic_round",
    "ternary_gradient_unbiased",
    "ReversibleCouplingBlock",
    "ReversibleSequential",
    "TernaryQATLinear",
    "make_linear",
    "GPT",
    "GPTConfig",
    "CONFIGS",
    "memory_report",
    "peak_training_memory",
    "compare_training_peak",
    "fmt_bytes",
    "build_optimizer",
    "available_optimizers",
    "GaLoreAdamW",
    "LoMo",
    "load_corpus",
    "get_batch",
    "synthetic_corpus",
    "__version__",
]
