"""Opt-in, address-independent stochastic-rounding streams.

Stream identity comes from a stable module path and a user seed, never Python's
randomized hash(), a memory address, or a distributed rank. Exact replay requires
the same backend, tiling and update schedule. This is a small deterministic PRNG,
not a claim of statistically independent streams or a cryptographic generator.
"""
from __future__ import annotations

import hashlib
import operator

import torch


class RoundingStreamMixin:
    def _init_rounding_stream(self) -> None:
        self._rounding_salt = None
        self._rounding_step = 0

    def _next_rounding_seed(self, legacy_seed: int | None = None) -> int | None:
        if self._rounding_salt is None:
            return legacy_seed
        # The odd multiplier permutes uint32 step positions before mixing the salt.
        seed = (self._rounding_salt ^ (self._rounding_step * 0x9E3779B9)) & 0xFFFFFFFF
        self._rounding_step += 1
        return seed

    def _round_counter(self, value: torch.Tensor, row_offset: int = 0) -> torch.Tensor:
        from .counter import stochastic_round
        from .fused_update import hash_u32, uniform01

        seed = self._next_rounding_seed()
        if seed is None:
            return stochastic_round(value)
        elem = torch.arange(value.numel(), device=value.device, dtype=torch.int64)
        elem = elem.reshape_as(value) + row_offset * self.in_features
        rnd = uniform01(seed ^ hash_u32(elem))
        floor = value.floor()
        return floor + (rnd < value - floor).to(value.dtype)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self._rounding_salt is not None:
            destination[prefix + "rounding_stream"] = torch.tensor(
                [self._rounding_salt, self._rounding_step], dtype=torch.int64)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "rounding_stream"
        self._init_rounding_stream()
        if key in state_dict:
            self._rounding_salt, self._rounding_step = map(int, state_dict[key].tolist())
            state_dict = state_dict.copy()
            del state_dict[key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


def configure_rounding_streams(model: torch.nn.Module, seed: int = 0) -> dict[str, int]:
    """Assign and reset streams for supported counters using their named_modules paths.

    Call once after model construction/conversion, BEFORE loading a checkpoint.
    Checkpoints restore identities and positions; old checkpoints restore legacy RNG.
    Replicas must have the same paths and seed. Global torch RNG users (activation
    quantization, ternary pulses, dropout, etc.) are outside this helper's scope.
    """
    seed = operator.index(seed)
    streams = {}
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, RoundingStreamMixin):
            digest = hashlib.sha256(f"memory-native:sr:v1:{seed}:{name}".encode()).digest()
            salt = int.from_bytes(digest[:4], "little")
            if salt in streams.values():
                raise ValueError("rounding stream ID collision; choose a different seed")
            streams[name] = salt
            layers.append((module, salt))
    for module, salt in layers:
        module._rounding_salt = salt
        module._rounding_step = 0
    return streams
