from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from sdk.api import ensure_dir, write_json


def discover_samples(raw_split_dir: Path) -> list[tuple[str, Path, Path | None]]:
    fasta_files = sorted(raw_split_dir.glob("*.fasta"))
    samples = []
    for fasta_path in fasta_files:
        target_id = fasta_path.stem
        reference = None
        for suffix in (".cif", ".mmcif", ".pdb"):
            candidate = raw_split_dir / f"{target_id}{suffix}"
            if candidate.exists():
                reference = candidate
                break
        samples.append((target_id, fasta_path, reference))
    return samples


def run_simplefold_preprocessing(
    *,
    simplefold_repo: Path,
    references_dir: Path,
    processed_dir: Path,
    tokenized_dir: Path,
) -> None:
    process_mmcif = simplefold_repo / "src" / "simplefold" / "process_mmcif.py"
    process_structure = simplefold_repo / "src" / "simplefold" / "process_structure.py"
    if not process_mmcif.exists() or not process_structure.exists():
        raise RuntimeError(
            "simplefold repo is missing src/simplefold/process_mmcif.py or process_structure.py"
        )

    subprocess.run(
        [
            sys.executable,
            str(process_mmcif),
            "--data_dir",
            str(references_dir),
            "--out_dir",
            str(processed_dir),
            "--use-assembly",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(process_structure),
            "--target_dir",
            str(processed_dir),
            "--token_dir",
            str(tokenized_dir),
        ],
        check=True,
    )


def _find_first_relative(base_dir: Path, patterns: list[str]) -> str | None:
    for pattern in patterns:
        matches = sorted(base_dir.rglob(pattern))
        if matches:
            return str(matches[0].relative_to(base_dir))
    return None


def build_split(
    *,
    split_name: str,
    raw_root: Path,
    output_root: Path,
    simplefold_repo: Path | None,
    include_public_references: bool,
) -> dict:
    raw_split_dir = raw_root / split_name
    split_dir = ensure_dir(output_root / split_name)
    fastas_dir = ensure_dir(split_dir / "fastas")
    references_dir = ensure_dir(split_dir / "references")
    processed_dir = split_dir / "processed"
    tokenized_dir = split_dir / "samples"

    samples = []
    discovered = discover_samples(raw_split_dir)
    has_references = any(reference_path is not None for _, _, reference_path in discovered)

    for target_id, fasta_path, reference_path in discovered:
        copied_fasta_path = fastas_dir / fasta_path.name
        shutil.copyfile(fasta_path, copied_fasta_path)
        sample = {
            "target_id": target_id,
            "sequence_fasta_path": str(Path("fastas") / fasta_path.name),
        }
        if reference_path is not None and include_public_references:
            copied_reference_path = references_dir / reference_path.name
            shutil.copyfile(reference_path, copied_reference_path)
            sample["reference_structure_path"] = str(Path("references") / reference_path.name)
        samples.append(sample)

    if simplefold_repo is not None and has_references and split_name in {"train", "val"}:
        ensure_dir(processed_dir)
        ensure_dir(tokenized_dir)
        run_simplefold_preprocessing(
            simplefold_repo=simplefold_repo,
            references_dir=references_dir,
            processed_dir=processed_dir,
            tokenized_dir=tokenized_dir,
        )
        for sample in samples:
            target_id = sample["target_id"]
            target_variants = {target_id, target_id.lower(), target_id.upper()}
            token_path = _find_first_relative(
                split_dir,
                [f"samples/tokens/{variant}.pkl" for variant in target_variants],
            )
            record_path = _find_first_relative(
                split_dir,
                [f"samples/records/{variant}.json" for variant in target_variants],
            )
            processed_structure_path = _find_first_relative(
                split_dir,
                [f"processed/structures/{variant}.npz" for variant in target_variants],
            )
            processed_record_path = _find_first_relative(
                split_dir,
                [f"processed/records/{variant}.json" for variant in target_variants],
            )
            if token_path:
                sample["tokenized_sample_path"] = token_path
            if record_path:
                sample["tokenized_record_path"] = record_path
            if processed_structure_path:
                sample["processed_structure_path"] = processed_structure_path
            if processed_record_path:
                sample["processed_record_path"] = processed_record_path

    manifest = {"split": split_name, "samples": samples}
    write_json(split_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SimpleFold-ready benchmark bundle.")
    parser.add_argument("--raw_dir", required=True, help="Directory containing train/val/test FASTA and structure files.")
    parser.add_argument("--output_dir", required=True, help="Output bundle directory.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to simplefold_100M checkpoint.")
    parser.add_argument("--simplefold_repo", help="Optional checkout of apple/ml-simplefold for preprocessing train/val references.")
    parser.add_argument(
        "--exclude_public_test_references",
        action="store_true",
        help="Omit test reference structures from the emitted bundle for hidden evaluation.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    checkpoint_path = Path(args.checkpoint_path).resolve()
    shutil.copyfile(checkpoint_path, checkpoints_dir / checkpoint_path.name)

    simplefold_repo = Path(args.simplefold_repo).resolve() if args.simplefold_repo else None
    include_public_test_references = not args.exclude_public_test_references

    split_manifests = {}
    for split_name in ("train", "val", "test"):
        split_manifests[split_name] = build_split(
            split_name=split_name,
            raw_root=raw_dir,
            output_root=output_dir,
            simplefold_repo=simplefold_repo,
            include_public_references=include_public_test_references or split_name != "test",
        )

    write_json(
        output_dir / "manifest.json",
        {
            "version": 2,
            "description": "SimpleFold-ready benchmark bundle.",
            "checkpoint": f"checkpoints/{checkpoint_path.name}",
            "splits": {
                name: f"{name}/manifest.json"
                for name in split_manifests
            },
        },
    )


if __name__ == "__main__":
    main()
