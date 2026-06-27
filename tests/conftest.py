"""Pin CPU threads for the test suite.

The stochastic-quantization tests average hundreds of small tensor ops; with the OpenMP/MKL
thread pools left at their default size some runtime/environment combinations spend all their
time in thread contention and appear to hang (observed in review). One thread makes the suite
fast and deterministic everywhere, mirroring `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

torch.set_num_threads(1)
