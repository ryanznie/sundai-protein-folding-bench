# Production Schema

## Suggested Tables

### teams

- `id`
- `name`
- `created_at`

### users

- `id`
- `email`
- `display_name`
- `team_id`
- `created_at`

### submissions

- `id`
- `team_id`
- `created_by_user_id`
- `status`
- `storage_key`
- `runtime_sec`
- `valid`
- `invalid_reason`
- `created_at`
- `completed_at`

### scores

- `id`
- `submission_id`
- `mean_tm_score`
- `mean_lddt`
- `mean_rmsd`
- `mean_ca_rmsd`
- `mean_gdt_ts_like`
- `min_coverage`
- `total_runtime_sec`

### submission_targets

- `id`
- `submission_id`
- `target_id`
- `valid`
- `tm_score`
- `lddt`
- `rmsd`
- `ca_rmsd`
- `gdt_ts_like`
- `coverage`

## Best Submission Logic

For leaderboard display:

- choose each team's best valid submission
- order by ranking formula

Recommended formula:

1. `mean_tm_score` descending
2. `total_runtime_sec` ascending
