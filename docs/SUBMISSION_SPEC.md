# Submission Spec

## Upload Format

Participants upload:

```text
baseline_submission/
  train.py
  config.json
```

## Entry Point

The benchmark system runs:
## Submission Bundle

To submit to the leaderboard, create a `.zip` file with the following structure:

```text
submission.zip
└── submission/
    ├── train.py
    └── config.json
```

**The folder inside the zip must be named `submission/`.**

The CLI runner uses the following parameters internally:

```bash
python3 benchmark.py \
  --input_dir /input \
  --output_dir /output \
  --submission /workspace/submission/train.py \
  --config /workspace/submission/config.json \
  --timeout_sec 600
```
At execution time, the service overrides these config fields:

- `backend = "torch"`
- `nsample_per_protein = 1`

MLX is disabled in the hosted starter backend.

## Input Contract

Mounted read-only at `/input`:

- `/input/train/manifest.json`
- `/input/val/manifest.json`
- `/input/test/manifest.json`
- `/input/checkpoints/simplefold_100M.ckpt`

Each test sample must provide a FASTA path relative to `test/manifest.json`:

```json
{
  "target_id": "example_target",
  "sequence_fasta_path": "fastas/example_target.fasta",
  "baseline_structure_path": "baselines/example_target.cif",
  "min_coverage": 0.95
}
```

`baseline_structure_path` is required only for public-dev scoring.

Each FASTA input corresponds to one required prediction output.

## Output Contract

Required:

```text
/output/predictions/<target_id>_sampled_0.cif
```

That means one CIF per input FASTA, with `sampled_0` as the only accepted
sample index.

Optional:

- `/output/checkpoint.pt`
- `/output/metrics.json`
- `/output/logs/*.log`

## Invalid Conditions

- timeout exceeded
- crash or nonzero failure
- missing prediction files
- empty prediction files
- no parseable CA trace in predicted structure
- missing public-dev baseline structure

## Ranking

1. `mean_tm_score` descending
2. `runtime_sec` ascending
