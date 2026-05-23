# Local Service Handoff

This document is the shortest path to wiring the benchmark into a separate local
leaderboard service.

## Pin These Inputs

### Benchmark Repo

Use a pinned commit SHA from:

```text
https://github.com/ryanznie/sundai-protein-folding-bench
```

Do not integrate against a moving `main` branch reference.

### Bundle

Current local bundle root:

```text
/Users/ryanznie/Desktop/Important/Work/Sundai/bundles/public_lb_v1
```

Treat the bundle as read-only and immutable.

## Runtime Contract

The worker should mount:

- `/input` -> bundle root
- `/output` -> writable run output directory

The runner should execute:

```bash
bash benchmark.sh
```

with:

```text
INPUT_DIR=/input
OUTPUT_DIR=/output
TIMEOUT_SEC=600
```

## Submission Contract

For v1, the benchmark expects a submission payload containing:

```text
submission/
  train.py
  config.json
```

The benchmark runner owns:

- benchmark.py
- scorer.py
- service-side validation
- final score extraction

## Current Bundle Status

`public_lb_v1` currently includes:

- `checkpoints/simplefold_100M.ckpt`
- train processed structures and tokenized samples for:
  - `8cny_A`
  - `8i85_A`
- val processed structure and tokenized sample for:
  - `7ftv_A`
- test FASTAs and public references for:
  - `7ftv_A`
  - `8cny_A`
  - `8g8r_A`
  - `8i85_A`

## Important Known Gap

The bundle does **not** yet include precomputed cached `ESM-3B` features.

That means:

- service integration can start now
- but the benchmark is not yet fully optimized for the tightest feedback loop

## Recommended Worker Behavior

1. Unpack submission into a workspace.
2. Mount the pinned bundle at `/input`.
3. Mount an empty output dir at `/output`.
4. Run `bash benchmark.sh`.
5. Read:
   - `/output/results.json`
   - `/output/summary.json`
6. Persist logs and scores.

## Suggested Metadata To Store

- `benchmark_repo`
- `benchmark_commit_sha`
- `bundle_version`
- `bundle_uri`
- `runtime_sec`
- `valid`
- per-target metrics

## Recommended v1 Defaults

- fixed model track: `SimpleFold-100M`
- one GPU
- no internet during execution
- public starter bundle only
- final ranking by:
  1. `mean_tm_score` descending
  2. `runtime_sec` ascending
