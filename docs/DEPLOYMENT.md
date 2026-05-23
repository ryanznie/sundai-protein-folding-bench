# Deployment Plan

## v1 Goal

Ship a production system with:

- fixed `SimpleFold-100M`
- preprocessed / precomputed bundle
- uploaded `submission.zip`
- one-GPU execution
- hidden evaluation split
- public leaderboard

## Recommended Phases

### Phase 1

Internal runner only.

- manual submission intake
- one benchmark image
- one worker machine
- static leaderboard updates

### Phase 2

Self-serve benchmark platform.

- submission API
- DB-backed job queue
- automated worker execution
- leaderboard UI

### Phase 3

Competition hardening.

- auth
- rate limits
- private leaderboard freeze
- team management

## Infrastructure

### Required

- Postgres
- Redis or queue equivalent
- object storage
- one or more GPU runners

### Suggested Container Images

1. API image
2. Leaderboard UI image
3. GPU benchmark worker image

### Worker Image Should Include

- pinned Python
- pinned PyTorch
- pinned SimpleFold code snapshot
- benchmark repo snapshot
- benchmark bundle loader

## Submission Lifecycle

1. User uploads `submission.zip`
2. API stores blob and creates `submissions` row
3. Queue dispatches a worker job
4. Worker runs benchmark
5. Worker uploads logs / outputs
6. Worker writes score summary
7. Leaderboard refreshes best valid submission

## Security Constraints

- disable egress networking in benchmark container
- enforce file path boundaries
- reject oversized uploads
- pin max output size
- clear runner temp dirs after each job
