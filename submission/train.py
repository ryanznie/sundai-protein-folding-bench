from __future__ import annotations

import json
from pathlib import Path

from sdk.api import ensure_dir, prediction_path


def main(context, config) -> None:
    """
    Required entrypoint for benchmark execution.

    Parameters
    ----------
    context:
        Runtime context built by benchmark.py. Important fields:
        - context.input_dir
        - context.output_dir
        - context.timeout_sec
        - context.bundle.train_dir / val_dir / test_dir / checkpoints_dir
        - context.seconds_remaining()
    config:
        Parsed JSON from submission/config.json.

    This starter implementation is intentionally minimal. It shows the expected
    interface and writes placeholder predictions by copying any public stubs
    listed in the local test manifest.
    """
    predictions_dir = ensure_dir(context.output_dir / "predictions")
    test_manifest_path = context.bundle.test_dir / "manifest.json"
    if not test_manifest_path.exists():
        raise RuntimeError("missing test manifest")

    with test_manifest_path.open("r") as handle:
        manifest = json.load(handle)

    for sample in manifest.get("samples", []):
        target_id = sample["target_id"]
        public_stub = sample.get("public_prediction_stub")
        if public_stub is None:
            raise RuntimeError(
                "starter submission requires public_prediction_stub in the dev bundle; "
                "replace this logic with real training and inference for actual competition runs"
            )
        src_path = Path(public_stub)
        dst_path = prediction_path(context.output_dir, target_id)
        ensure_dir(dst_path.parent)
        dst_path.write_bytes(src_path.read_bytes())
