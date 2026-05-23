# Dataset Plan

## v1 Curation Goal

Build a small, stable benchmark around `SimpleFold-100M` with:

- a fast public development tier
- a broader public leaderboard tier
- a hidden private-final tier

The current repo only includes bundled public examples, so the curated starter
dataset is intentionally small.

## Curated Public Splits

### `public_dev`

Targets:

- `8cny_A` - 182 aa
- `8i85_A` - 280 aa

Purpose:

- local debugging
- scoring validation
- fast iteration

### `public_lb`

Targets:

- `7ftv_A` - 351 aa
- `8cny_A` - 182 aa
- `8g8r_A` - 426 aa
- `8i85_A` - 280 aa

Purpose:

- baseline public benchmark
- include one longer stress target

## Recommended Hidden Expansion

For a real hackathon, expand beyond the bundled four examples.

Recommended sizes:

- private-final: `20-30` monomer proteins
- optional public-lb expansion: `20-30` total public targets

Recommended selection rules:

- monomer only
- mostly `120-350 aa`
- a few stretch targets up to `400-450 aa`
- avoid unusual assemblies or chemistry
- avoid extremely tiny proteins

## Why Cache ESM

The current SimpleFold stack depends on `esm2_3B`.

For leaderboard use, benchmark owners should precompute and cache:

- preprocessing artifacts
- tokenized / cropped sample payloads
- `esm_s` features

That keeps the challenge focused on fine-tuning and inference strategy rather
than repeated feature extraction.
