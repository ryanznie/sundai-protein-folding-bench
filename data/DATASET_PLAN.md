# Dataset Plan

## v1 Curation Goal

Build a stable, high-quality benchmark around `SimpleFold-100M` with tiered evaluation tracks:

- **Public LB (v1)**: A fast, stable leaderboard for initial ranking.
- **Hackathon (v1)**: An expanded dataset for the main competition, focused on diverse monomer folds.
- **Private Final**: A hidden tier for final verification.

The current repository includes precomputed ESM features and structural artifacts to keep participants focused on folding strategy rather than featurization.

## Active Benchmark Bundles

### `public_lb_v1` (Leaderboard)

A minimal, stable set of 4 targets used for the live public leaderboard.

**Targets:**

- `7ftv_A` - 351 aa (Stress target)
- `8cny_A` - 182 aa
- `8g8r_A` - 426 aa (Large target)
- `8i85_A` - 280 aa

**Purpose:**

- Immediate feedback on submission validity.
- Stable ranking for public competition.

### `simplefold_hackathon_v1` (Hackathon Tier)

A broader dataset derived from competition data, including full train, validation, and test splits.

**Test Targets (10 total):**

- `1a70`, `1em7`, `1off`, `1ubq`, `1yn4`, `2gbn`, `2pko`, `2wyq`, `3wcq`, `7yo8`

**Purpose:**

- Comprehensive evaluation of model performance.
- Diverse range of lengths and fold topologies.
- Ground truth for hyperparameter tuning.

## Hidden Expansion Strategy

For future iterations or final evaluation:

- **Private-Final Tier**: `20-30` additional monomer proteins.
- **Selection Criteria**:
    - Monomer only.
    - Lengths mostly between `120-350 aa`.
    - Stretch targets up to `450 aa`.
    - Avoid unusual assemblies, non-standard chemistry, or extremely tiny proteins.

## Artifact Caching

The benchmark utilizes a precomputed cache system to ensure fast execution and consistent environments:

- **ESM Features**: `esm2_3B` embeddings are precomputed and stored as `.pt` files.
- **Tokenized Samples**: Cropped and tokenized payloads are cached as `.pkl` files.
- **Processed Structures**: Ground truth geometries are stored as `.npz` files for scoring.

This keeps the challenge focused on **fine-tuning** and **inference optimization**.
