# Public Dev Bundle

This directory is a placeholder for the public development bundle.

Expected layout:

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

Recommended sample payload fields:

- `target_id`
- `record`
- tokenized / cropped model tensors
- `esm_s`
- optional `reference_path` for public scoring

Recommended test manifest fields:

```json
{
  "samples": [
    {
      "target_id": "example_target",
      "public_prediction_stub": "/abs/path/to/example_target_sampled_0.cif",
      "public_metrics": {
        "tm_score": 0.42,
        "lddt": 0.56,
        "rmsd": 8.3,
        "ca_rmsd": 6.9,
        "gdt_ts_like": 0.18
      }
    }
  ]
}
```

In production, benchmark owners should replace `public_prediction_stub` with
real hidden-data scoring and provide the actual preprocessed feature bundle.
