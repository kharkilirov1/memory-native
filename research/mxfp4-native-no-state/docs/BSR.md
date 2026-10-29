# Balanced stochastic rounding (BSR)

BSR is a no-persistent-state experiment for the MXFP4 E2M1 ladder. It groups coordinates
by sign and adjacent quantization gap, then uses randomized cumulative rounding within each
group.

For each coordinate between levels l and h, p=(x-l)/(h-l). The randomized cumulative
scheme preserves the stochastic-rounding marginal E[q]=x (within the representable range)
while bounding the aggregate rounding error of a sign/gap group by one local gap.

This is not a proof of lower per-coordinate variance or better language-model quality.
The tiny-char-LM A/B is only two seeds and the measured average improvement is 0.0011
bits/char, too small to claim a real gain. Keep it opt-in pending longer, multi-seed runs.

Run tests from this directory with `pytest tests/test_bsr.py`.
