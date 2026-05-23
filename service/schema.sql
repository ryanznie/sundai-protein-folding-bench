create table if not exists teams (
    id text primary key,
    name text not null unique,
    created_at text not null default current_timestamp
);

create table if not exists users (
    id text primary key,
    email text not null unique,
    display_name text not null,
    team_id text not null references teams(id),
    created_at text not null default current_timestamp
);

create table if not exists submissions (
    id text primary key,
    team_id text not null references teams(id),
    created_by_user_id text not null references users(id),
    status text not null,
    storage_key text not null,
    runtime_spec text not null,
    original_filename text,
    config_json text,
    runtime_sec real,
    valid integer,
    invalid_reason text,
    created_at text not null default current_timestamp,
    completed_at text
);

create table if not exists scores (
    submission_id text primary key references submissions(id),
    mean_tm_score real,
    mean_lddt real,
    mean_rmsd real,
    mean_ca_rmsd real,
    mean_gdt_ts_like real,
    min_coverage real,
    total_runtime_sec real,
    raw_summary_json text
);

create table if not exists submission_targets (
    id integer primary key autoincrement,
    submission_id text not null references submissions(id),
    target_id text not null,
    valid integer not null,
    tm_score real,
    lddt real,
    rmsd real,
    ca_rmsd real,
    gdt_ts_like real,
    coverage real,
    invalid_reason text,
    matched_residues integer,
    reference_residues integer
);

create index if not exists idx_submissions_team_created_at on submissions(team_id, created_at desc);
create index if not exists idx_submission_targets_submission_id on submission_targets(submission_id);
