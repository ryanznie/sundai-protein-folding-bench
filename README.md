# Sundai Protein Folding Bench

`sundai-protein-folding-bench` is a Kaggle-style GPU benchmark for
budget-constrained protein folding fine-tuning.

Participants do not submit full repos. They submit a small code artifact,
centered on `submission/train.py`, which is executed inside a fixed benchmark
environment.

## Track

The default track is:

- fixed base model: `SimpleFold-100M`
- fixed preprocessed train / validation / test bundle
- fixed precomputed ESM features
- fixed GPU budget
- fixed wall-clock timeout
- fixed output schema

The benchmark runner owns:

- environment setup
- timeout enforcement
- output validation
- scoring

Participants only change:

- `submission/train.py`
- optionally `submission/config.json`

## Competition Objective

Given:

- a fixed starting checkpoint
- fixed preprocessed inputs
- fixed precomputed ESM features
- a strict runtime budget

produce the best hidden-set structure predictions.

Recommended ranking:

1. `mean_tm_score` descending
2. `total_runtime_sec` ascending

For public development, the scorer can also expose faster proxy metrics:

- `mean_ca_rmsd`
- `mean_gdt_ts_like`
- `coverage`

## What Gets Mounted At Runtime

The benchmark runner assumes a bundle mounted at `/input` with this structure:

```text
/input/
  manifest.json
  train/
    manifest.json
    samples/
      <target_id>.pt
  val/
    manifest.json
    samples/
      <target_id>.pt
  test/
    manifest.json
    samples/
      <target_id>.pt
  checkpoints/
    simplefold_100M.ckpt
```

Each sample payload may contain:

- tokenized / cropped model inputs
- `record` metadata
- `esm_s` precomputed ESM features
- optional reference structure path for public-dev scoring

## Required Submission Outputs

`submission/train.py` must write predictions to:

```text
/output/predictions/<target_id>_sampled_0.cif
```

Optional outputs:

- `/output/checkpoint.pt`
- `/output/metrics.json`
- `/output/logs/*.log`

## Invalid Submission Conditions

A run is invalid if:

- it exceeds the timeout
- it crashes
- it writes malformed outputs
- it misses one or more required targets
- it produces empty or unparsable structures
- residue coverage falls below the configured threshold

## Local Development Flow

1. Create a bundle under `data/public_dev/`.
2. Implement `submission/train.py`.
3. Run:

```bash
bash benchmark.sh
```

The runner will:

- execute the submission
- collect runtime
- validate outputs
- score the run
- write `results.json`

## Repository Layout

```text
sundai-protein-folding-bench/
  GETTING_STARTED.md
  README.md
  benchmark.py
  benchmark.sh
  scorer.py
  sdk/
    api.py
  reference/
    baseline_train.py
    baseline_config.json
  submission/
    train.py
    config.json
  data/
    public_dev/
      README.md
```
