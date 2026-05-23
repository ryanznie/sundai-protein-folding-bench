# Architecture

## Public Repo

The public repo defines:

- bundle builder and bundle schema
- benchmark runner and timeout enforcement
- structure-based public-dev scoring
- starter starter contract

## Production Services

### API

[service/app.py](/Users/ryanznie/Desktop/Important/Work/Sundai/sundai-protein-folding-bench/service/app.py)
accepts starters, records status, stores scores, exposes a leaderboard, and
serves the static frontend from [service/web/index.html](/Users/ryanznie/Desktop/Important/Work/Sundai/sundai-protein-folding-bench/service/web/index.html).

### Worker

[worker/run_starter.py](/Users/ryanznie/Desktop/Important/Work/Sundai/sundai-protein-folding-bench/worker/run_starter.py)
unpacks `starter.zip`, overlays it into the benchmark repo snapshot, runs the
benchmark container, and reports results back to the API.

### Runtime

The benchmark container should contain:

- pinned Python
- pinned SimpleFold install
- pinned benchmark repo snapshot
- access to the benchmark bundle

The worker wrapper enforces:

- `--gpus all`
- `--network none`
- read-only `/input`
- writable `/output`
