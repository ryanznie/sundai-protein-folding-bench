# Submission Spec

## Upload Format

Participants upload:

```text
submission/
  train.py
  config.json
```

## Entry Point

The benchmark system runs:

```bash
python3 benchmark.py \
  --input_dir /input \
  --output_dir /output \
  --submission /workspace/submission/train.py \
  --config /workspace/submission/config.json \
  --timeout_sec 600
```

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
  "reference_structure_path": "references/example_target.cif",
  "min_coverage": 0.95
}
```

`reference_structure_path` is required only for public-dev scoring.

## Output Contract

Required:

```text
/output/predictions/<target_id>_sampled_0.cif
```

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
- missing public-dev reference structure
- coverage below threshold

## Ranking

1. `mean_tm_score` descending
2. `runtime_sec` ascending
