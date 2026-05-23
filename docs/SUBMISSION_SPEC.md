# Submission Spec

## Upload Format

Participants upload a zip file containing:

```text
submission/
  train.py
  config.json
```

No other files are required for v1.

## Entry Point

The benchmark system will run:

```bash
bash benchmark.sh
```

which calls:

```bash
python benchmark.py \
  --input_dir /input \
  --output_dir /output \
  --submission /workspace/submission/train.py \
  --config /workspace/submission/config.json \
  --timeout_sec 600
```

## Input Contract

Mounted read-only at `/input`:

- `/input/train`
- `/input/val`
- `/input/test`
- `/input/checkpoints/simplefold_100M.ckpt`

Samples are expected to already include preprocessed tensors and precomputed
ESM features.

## Output Contract

Required:

```text
/output/predictions/<target_id>_sampled_0.cif
```

Optional:

- `/output/checkpoint.pt`
- `/output/metrics.json`
- `/output/logs/*.log`

## Invalid Submission Conditions

- timeout exceeded
- crash or nonzero failure
- missing prediction files
- empty prediction files
- output parse failure
- coverage below threshold

## Ranking

Recommended v1 final ranking:

1. `mean_tm_score` descending
2. `total_runtime_sec` ascending

Recommended public metrics:

- `mean_ca_rmsd`
- `mean_gdt_ts_like`
- `coverage`
- `total_runtime_sec`
