from __future__ import annotations

from pathlib import Path

from sdk.api import ensure_dir, prediction_path


def main(context, config) -> None:
    ensure_dir(context.output_dir / "predictions")
    test_manifest = context.bundle.test_dir / "manifest.json"
    if not test_manifest.exists():
        raise RuntimeError("missing test manifest")

    import json

    with test_manifest.open("r") as handle:
        manifest = json.load(handle)

    for sample in manifest.get("samples", []):
        target_id = sample["target_id"]
        source = sample.get("public_prediction_stub")
        if source is None:
            raise RuntimeError(
                "reference baseline requires public_prediction_stub entries in the test manifest"
            )
        src_path = Path(source)
        dst_path = prediction_path(context.output_dir, target_id)
        dst_path.write_bytes(src_path.read_bytes())
