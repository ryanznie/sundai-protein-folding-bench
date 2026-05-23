from __future__ import annotations

from sdk.api import ensure_dir
from sdk.simplefold import (
    can_use_cached_bundle_features,
    run_simplefold_batch_inference,
    run_simplefold_cached_bundle_inference,
)


def main(context, config) -> None:
    """
    Submission entrypoint for local and production runs.

    The default implementation runs the public SimpleFold CLI against the test
    FASTA files listed in the mounted bundle manifest and normalizes the first
    emitted CIF into the benchmark output contract.
    """
    ensure_dir(context.output_dir / "predictions")
    manifest = context.bundle.test_manifest()
    samples = manifest.get("samples", [])
    target_ids = [sample["target_id"] for sample in samples]
    fasta_root = context.bundle.test_dir / "fastas"
    if can_use_cached_bundle_features(samples):
        print("[submission] bundle exposes cached ESM + processed structures; using cached inference path", flush=True)
        run_simplefold_cached_bundle_inference(
            split_dir=context.bundle.test_dir,
            samples=samples,
            output_dir=context.output_dir,
            config=config,
        )
        print("[submission] cached bundle inference finished and predictions copied", flush=True)
        return
    if not fasta_root.exists():
        raise RuntimeError("test bundle is missing test/fastas for batch inference")
    print(
        f"[submission] starting batch inference targets={len(target_ids)} ids={','.join(target_ids)} "
        f"model={config.get('simplefold_model', 'simplefold_100M')} "
        f"steps={config.get('num_steps', 500)} backend={config.get('backend', 'torch')}",
        flush=True,
    )
    run_simplefold_batch_inference(
        fasta_root=fasta_root,
        target_ids=target_ids,
        output_dir=context.output_dir,
        config=config,
    )
    print("[submission] batch inference finished and predictions copied", flush=True)
