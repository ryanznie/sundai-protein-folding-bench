# Getting Started

## What You Are Allowed To Edit

Edit only:

- `submission/train.py`
- `submission/config.json`

Do not modify:

- `benchmark.py`
- `benchmark.sh`
- `scorer.py`
- `sdk/`
- benchmark data layout

## Runtime Contract

The benchmark system runs:

```bash
bash benchmark.sh
```

`benchmark.sh` invokes `benchmark.py`, which:

- loads the fixed bundle
- calls your submission entrypoint
- enforces a timeout
- validates your outputs
- scores your predictions

## What Your Code Must Do

Your code should:

1. load the fixed starting checkpoint
2. load training and test samples from the mounted bundle
3. optionally fine-tune for the allotted time
4. produce one prediction per hidden target
5. write predictions to `/output/predictions/`

Required prediction filename pattern:

```text
<target_id>_sampled_0.cif
```

## Default Assumptions

This benchmark assumes:

- preprocessing is already done
- ESM features are already computed
- network access is disabled
- runtime is GPU-backed
- total wall-clock time is fixed by the benchmark harness

## Local Run

Set optional overrides:

```bash
export INPUT_DIR=$(pwd)/data/public_dev
export OUTPUT_DIR=$(pwd)/output
export TIMEOUT_SEC=600
```

Then run:

```bash
bash benchmark.sh
```

## Expected Outputs

After a valid run, you should see:

- `output/predictions/*.cif`
- `output/results.json`
- `output/summary.json`

## Recommended First Strategy

Start with:

- no fine-tuning
- load base checkpoint
- run plain inference

Then iterate on:

- lightweight fine-tuning
- parameter freezing
- LoRA / adapters
- step budgeting
- inference scheduling
