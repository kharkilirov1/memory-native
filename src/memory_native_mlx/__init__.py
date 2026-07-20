"""memory-native on MLX — Apple-silicon (Metal) port of the counter-synapse method.

Same finite-state 6-bit synapse, same deterministic hash-SR update as the CUDA/Triton
path, expressed in MLX so it trains on a MacBook's unified memory. The pure-MLX ops run
on any MLX backend (Metal on macOS, CPU on Linux — which is how this port is CI-tested
against the PyTorch reference); the fused Metal kernel engages automatically on Apple
GPUs. See docs/MLX_PORT.md for the design mapping and validation status.
"""
from .counter import (
    C_DEFAULT,
    RMSCounterLinear,
    counter_update_hashsr,
    decode_state,
    encode_state,
    hash_u32,
    uniform01,
)
from .bonsai import (
    group_counter_from_dense,
    group_counter_from_quantized,
    ternary_to_mlx_quant,
    to_mlx_quantized,
)
from .group_scale import GroupScaleCounterLinear
from .packed import PackedRMSCounterLinear, pack_codes, unpack_codes
from .reversible import ReversibleCouplingBlock, ReversibleSequence, ReversibleSequential

__all__ = [
    "C_DEFAULT",
    "encode_state",
    "decode_state",
    "hash_u32",
    "uniform01",
    "counter_update_hashsr",
    "RMSCounterLinear",
    "PackedRMSCounterLinear",
    "GroupScaleCounterLinear",
    "group_counter_from_dense",
    "group_counter_from_quantized",
    "ternary_to_mlx_quant",
    "to_mlx_quantized",
    "pack_codes",
    "unpack_codes",
    "ReversibleCouplingBlock",
    "ReversibleSequential",
    "ReversibleSequence",
]
