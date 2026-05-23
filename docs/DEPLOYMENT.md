# Deployment Plan

## Images

The repo now includes:

1. `docker/api/Dockerfile`
2. `docker/worker/Dockerfile`

The worker runtime contract is described in
[docker/worker/runtime-spec.json](/Users/ryanznie/Desktop/Important/Work/Sundai/sundai-protein-folding-bench/docker/worker/runtime-spec.json).

## Local API Bring-Up

```bash
docker compose up api
```

The API container expects these host mounts, already configured in
[docker-compose.yml](/Users/ryanznie/Desktop/Important/Work/Sundai/sundai-protein-folding-bench/docker-compose.yml):

- `/Users/ryanznie/Desktop/Important/Work/Sundai/bundles/public_lb_v1` -> `/opt/sundai/public_lb_v1`
- `/Users/ryanznie/Desktop/Important/Work/Sundai/ml-simplefold` -> `/opt/sundai/ml-simplefold`
- a writable Docker volume at `/data` for the SQLite DB, uploads, and per-submission logs

Once Docker is running, build and start with:

```bash
docker compose up -d --build api
docker compose logs -f api
```

Or, in the repo-local environment:

```bash
uv sync
uv run uvicorn service.app:app --reload
```

## Worker Flow

1. `POST /submissions`
2. Store uploaded zip in object storage
3. Start `worker/run_submission.py`
4. Worker runs `docker run --gpus all --network none`
5. Worker posts results to `/internal/submissions/{id}/complete`
6. Leaderboard reads best scored submission per team

## Submission Runtime Rules

The hosted backend enforces:

- `backend = "torch"`
- `nsample_per_protein = 1`
- one CIF output per FASTA input at `/output/predictions/<target_id>_sampled_0.cif`

MLX is disabled in the submission backend.

## Production Constraints

- disable egress in benchmark containers
- mount `/input` read-only
- mount `/output` writable
- enforce GPU count and timeout from runtime spec
- mount a repo snapshot so the runner and submission contract stay pinned
