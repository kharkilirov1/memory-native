# Project notes

The upstream experimental archive describes MXFP4 E2M1 values with one E8M0 scale per 32
weights and an optional compact update accumulator. The design goal is to avoid FP32 master
weights and Adam moments for the MXFP4 weight path.

The archive's own evidence is limited to small char-LM experiments. It does not establish
quality retention or training throughput for 12B models. The 6.25-bit/parameter figure is
an ideal persistent-state target for the compact three-state path, not a guarantee of
total training memory. Eager decoding, gradients, activations, and temporary workspaces
must be measured separately.

See `NO_STATE_EXPERIMENTS.md` for the tested no-extra-state optimizer candidates and the
decision not to promote them to defaults.
