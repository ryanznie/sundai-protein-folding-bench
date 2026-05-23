from __future__ import annotations

from sdk.api import ensure_dir
from sdk.simplefold import (
    can_use_cached_bundle_features,
    run_simplefold_batch_inference,
    run_simplefold_cached_bundle_inference,
)


def main(context, config) -> None:
    ensure_dir(context.output_dir / "predictions")
    manifest = context.bundle.test_manifest()
    samples = manifest.get("samples", [])
    target_ids = [sample["target_id"] for sample in samples]
    fasta_root = context.bundle.test_dir / "fastas"
    if can_use_cached_bundle_features(samples):
        print("[reference] bundle exposes cached ESM + processed structures; using cached inference path", flush=True)
        run_simplefold_cached_bundle_inference(
            split_dir=context.bundle.test_dir,
            samples=samples,
            output_dir=context.output_dir,
            config=config,
        )
        print("[reference] cached bundle inference finished and predictions copied", flush=True)
        return
    if not fasta_root.exists():
        raise RuntimeError("test bundle is missing test/fastas for batch inference")
    run_simplefold_batch_inference(
        fasta_root=fasta_root,
        target_ids=target_ids,
        output_dir=context.output_dir,
        config=config,
    )
