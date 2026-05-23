# Architecture

## Overview

`sundai-protein-folding-bench` is split into two layers:

1. Public benchmark repo
2. Private production scoring service

The public repo defines:

- submission contract
- benchmark runner
- local validation
- starter submission template
- public development bundle format

The private service owns:

- submission upload API
- queueing
- GPU job execution
- hidden-set scoring
- leaderboard state

## Public Repo Contract

Participants submit only:

- `submission/train.py`
- `submission/config.json`

The fixed benchmark system owns:

- data bundle
- runtime environment
- timeout rules
- scoring logic

## Production Services

### API

Responsibilities:

- accept `submission.zip`
- create submission jobs
- expose status and leaderboard endpoints

Suggested stack:

- FastAPI
- Postgres

### Worker

Responsibilities:

- unpack submission
- inject into fixed benchmark image
- run `bash benchmark.sh`
- collect outputs and logs
- write scores back to Postgres / object storage

Suggested stack:

- Docker
- NVIDIA container runtime
- Redis or Postgres-backed queue

### Leaderboard UI

Responsibilities:

- show rankings
- show submission history
- link logs and summaries

Suggested stack:

- Next.js or another simple web frontend

## Runtime Sandbox

Recommended runtime guarantees:

- one fixed GPU SKU
- fixed CUDA / PyTorch image
- no internet access
- read-only `/input`
- writable `/output` and `/tmp`
- hard timeout

## Data Flow

1. Benchmark owner preprocesses FASTA once.
2. Benchmark owner precomputes ESM features once.
3. Bundle is packaged into fixed train / val / test splits.
4. Submission runs train/fine-tune + inference only.
5. Scorer validates outputs and computes metrics.

This keeps the challenge focused on adaptation and inference strategy instead of
repeated preprocessing cost.
