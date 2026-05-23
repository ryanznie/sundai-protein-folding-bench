# Getting Started

## What You Edit

Participants still only need:

- `baseline_submission/train.py`
- `baseline_submission/config.json`

Benchmark owners additionally use:

- `tools/build_simplefold_bundle.py`
- `service/`
- `worker/`
- `docker/`

## Build A Public-Dev Bundle

Prepare raw files like:

```text
raw_targets/
  train/
    target_a.fasta
    target_a.cif
  val/
    target_b.fasta
    target_b.cif
  test/
    target_c.fasta
    target_c.cif
```

Then run:

```bash
python3 tools/build_simplefold_bundle.py \
  --raw_dir /path/to/raw_targets \
  --output_dir data/public_dev \
  --checkpoint_path /path/to/simplefold_100M.ckpt \
  --simplefold_repo /path/to/ml-simplefold
```

For hidden test packaging, add `--exclude_public_test_references`.

## Local Benchmark Run

Set optional overrides:

```bash
export INPUT_DIR=$(pwd)/data/public_dev
export OUTPUT_DIR=$(pwd)/output
export TIMEOUT_SEC=600
```

Run:

```bash
bash benchmark.sh
```

## Required Test Manifest Fields

Each test sample should include:

- `target_id`
- `sequence_fasta_path`
- `reference_structure_path` for public-dev only
- optional `min_coverage`

The bundled `baseline_submission/train.py` reads `sequence_fasta_path` and runs the
SimpleFold CLI directly.
