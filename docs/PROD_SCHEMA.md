# Production Schema

The API ships a runnable SQL schema at
[service/schema.sql](/Users/ryanznie/Desktop/Important/Work/Sundai/sundai-protein-folding-bench/service/schema.sql).

## Tables

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
- `runtime_spec`
- `runtime_sec`
- `valid`
- `invalid_reason`
- `created_at`
- `completed_at`

### scores

- `submission_id`
- `mean_tm_score`
- `mean_lddt`
- `mean_rmsd`
- `mean_ca_rmsd`
- `mean_gdt_ts_like`
- `min_coverage`
- `total_runtime_sec`
- `raw_summary_json`

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
- `invalid_reason`
- `matched_residues`
- `reference_residues`

## Leaderboard Logic

The API endpoint `/leaderboard` returns each team's best valid submission using:

1. `mean_tm_score` descending
2. `total_runtime_sec` ascending
