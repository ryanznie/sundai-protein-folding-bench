# Sundai Protein Folding Bench

`sundai-protein-folding-bench` is a Kaggle-style benchmark scaffold for
budget-constrained SimpleFold adaptation and inference.

The repo now includes four concrete pieces:

- a real SimpleFold-backed submission path that calls the public `simplefold` CLI
- a bundle builder that packages FASTA files, references, and train/val tokenization
- a structure-based scorer for public-dev bundles
- a minimal FastAPI + Docker worker stack for production orchestration

## Runtime Contract

The benchmark runner mounts a bundle at `/input` and expects predictions at:

```text
/output/predictions/<target_id>_sampled_0.cif
```

The default submission reads `sequence_fasta_path` from `test/manifest.json`,
runs SimpleFold, and copies the emitted CIF into that filename contract.

## Bundle Layout

```text
/input/
  manifest.json
  train/
    manifest.json
    fastas/
    references/
    processed/
    samples/
  val/
    manifest.json
    fastas/
    references/
    processed/
    samples/
  test/
    manifest.json
    fastas/
    references/            # public-dev only
  checkpoints/
    simplefold_100M.ckpt
```

`train` and `val` can include tokenized artifacts produced by the public
SimpleFold preprocessing scripts. `test` must include per-target FASTA files and
may include reference structures only for public-dev scoring.

## Local Flow

1. Prepare a raw dataset with `train/`, `val/`, and `test/` FASTA files.
2. Build a bundle:

```bash
python3 tools/build_simplefold_bundle.py \
  --raw_dir /path/to/raw_targets \
  --output_dir data/public_dev \
  --checkpoint_path /path/to/simplefold_100M.ckpt \
  --simplefold_repo /path/to/ml-simplefold
```

3. Install the public `simplefold` CLI in the benchmark runtime.
4. Run:

```bash
bash benchmark.sh
```

## Public-Dev Scoring

The scorer validates that each prediction exists, is non-empty, contains a CA
trace, and reaches the requested residue coverage. It then computes:

- `tm_score`
- `lddt`
- `rmsd`
- `ca_rmsd`
- `gdt_ts_like`

The default ranking remains:

1. `mean_tm_score` descending
2. `runtime_sec` ascending

## Production Pieces

- `service/`: FastAPI submission and leaderboard API with a SQL schema
- `service/web/`: static frontend for leaderboard and submission registration
- `worker/`: Docker-based worker callback flow
- `docker/`: API and worker Dockerfiles plus a runtime spec

See [docs/ARCHITECTURE.md](/Users/ryanznie/Desktop/Important/Work/Sundai/sundai-protein-folding-bench/docs/ARCHITECTURE.md),
[docs/DEPLOYMENT.md](/Users/ryanznie/Desktop/Important/Work/Sundai/sundai-protein-folding-bench/docs/DEPLOYMENT.md), and
[docs/SUBMISSION_SPEC.md](/Users/ryanznie/Desktop/Important/Work/Sundai/sundai-protein-folding-bench/docs/SUBMISSION_SPEC.md).

## Local Service Env

This workspace does not currently have `fastapi` installed in the default
`python3` environment. For local validation here, the compatible interpreter is:

```bash
../../Adaptive-ML/constitutional-ai/.venv/bin/python
```

Example:

```bash
../../Adaptive-ML/constitutional-ai/.venv/bin/python -m uvicorn service.app:app --reload
```
