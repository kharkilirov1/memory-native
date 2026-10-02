# Preflight for the archived external campaign

The strict-v3 release, its runner/package and teacher cache are external. A public
clone cannot launch or validate that workflow. The public cached-KD workflow is
[notebooks/cached_kd_public.ipynb](../notebooks/cached_kd_public.ipynb).

Install `.[dev,donor,campaign]` before running the archived preflight. Its default
launch mode requires local release code and tests, a local donor checker, an
explicit cache validator, donor/cache/corpus directories and a complete resolved
runtime environment. It runs the supplied strict tests and validators. It never
downloads replacement code. The local policy is authoritative unless a policy
URL is explicitly supplied.

```sh
python scripts/preflight_consistency.py \
  --project /path/to/external-release \
  --policy /path/to/memory-native/production/qwen38_27b_recovery_3k.yaml \
  --donor /path/to/donor --data /path/to/corpus --cache /path/to/cache \
  --cache-validator /path/to/external-release/scripts/validate_kd_cache_v3.py \
  --expected-env-json /path/to/resolved_env.json
```

The validator receives `CACHE --expected-topk K --expected-steps N` plus
`--model-index INDEX --data-manifest FILE`. A sharded donor index is required
by this archived validator interface. It must
verify the private strict-v3 tensor and loss format, including tail buckets and
saved normalization. The public schema-v2 top-K cache is a different format.
A resolved environment snapshot is authoritative; missing entries are not filled
from unrelated shell variables. `MODEL`, `DATA_DIR` and `CACHE` must resolve to the
directories actually checked. The snapshot also supplies `GRAD_CKPT`,
`SAVE_BEST`, `FP_LR` and `NUM_BLOCKS` alongside all policy-mapped values.

The checker adds a **new content-evidence contract**. This is not a claim that the
archived cache originally contained these fields. An evidence adapter requires
access to the original donor/corpus/cache and their verified build provenance;
relabeling an unidentified old cache is insufficient. Its manifest must contain
`identities.model.files` and `identities.data.files`, with relative filenames,
byte lengths and SHA-256 digests; either representation may be a mapping or
list of `{path, bytes, sha256}` records. The data identity also records
`domain_order`; `shards` records each cache filename and SHA-256 digest. Metadata
checks cannot replace the private-format validator.

`--inspection-only` reads available metadata without executing external code.
It returns `INCOMPLETE` (exit 3) when evidence is missing, or `INSPECTION COMPLETE`
(exit 0); neither is launch approval. Launch failures return exit 2.

Passing these structural checks does not establish model quality, memory fit or
implementation of proposed schedules/proxy evaluation/full training resume. The
400-step archived cache also needs documented reuse semantics for a 3000-step
run. The original policy's unsupported gradient accumulation request is now
`grad_accum: 1`, one counter update per backward. Warm-quality evaluation and
acceptance gates remain mandatory in the external implementation.
