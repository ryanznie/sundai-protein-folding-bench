# Public Dev Bundle

This is the curated v1 `dev-fast` split.

Included targets:

- `8cny_A` - 182 aa
- `8i85_A` - 280 aa

Contents today:

```text
public_dev/
  fasta/
    8cny_A.fasta
    8i85_A.fasta
  references/
    8cny_A.cif
    8i85_A.cif
  manifest.json
  train/
    manifest.json
  val/
    manifest.json
  test/
    manifest.json
```

Why these two:

- short enough for a tight feedback loop
- both are monomer examples already bundled with the local SimpleFold repo
- enough to validate scoring, IO, and runtime behavior without making iteration slow

What is still missing for production:

- preprocessed model tensors
- cached `esm_s` features
- a larger public leaderboard split
- a hidden private-final split

Suggested production bundle format:

```text
public_dev/
  manifest.json
  train/
    manifest.json
    samples/
      <target_id>.pt
  val/
    manifest.json
    samples/
      <target_id>.pt
  test/
    manifest.json
    samples/
      <target_id>.pt
  checkpoints/
    simplefold_100M.ckpt
```
