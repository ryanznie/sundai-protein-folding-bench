from __future__ import annotations

import argparse
import sys
import json
import shutil
import subprocess
from pathlib import Path

import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def maybe_preprocess_split(simplefold_repo: Path, split_dir: Path) -> None:
    references_dir = split_dir / "references"
    processed_dir = split_dir / "processed"
    tokenized_dir = split_dir / "samples"
    if not references_dir.exists():
        return
    if (processed_dir / "manifest.json").exists() and (tokenized_dir / "manifest.json").exists():
        return

    process_mmcif = simplefold_repo / "src" / "simplefold" / "process_mmcif.py"
    process_structure = simplefold_repo / "src" / "simplefold" / "process_structure.py"

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


def find_existing(base: Path, candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = base / candidate
        if path.exists():
            return path
    return None


def prepare_simplefold_imports(simplefold_repo: Path) -> None:
    sys.path.append(str(simplefold_repo / "src" / "simplefold"))
    sys.path.append(str(simplefold_repo / "src"))


def load_esm_components(device: torch.device):
    from utils.esm_utils import _af2_to_esm, esm_registry

    esm_model, esm_dict = esm_registry["esm2_3B"]()
    esm_model = esm_model.to(device)
    esm_model.eval()
    af2_to_esm = _af2_to_esm(esm_dict).to(device)
    return esm_model, esm_dict, af2_to_esm


def compute_and_store_esm(
    *,
    simplefold_repo: Path,
    split_dir: Path,
    split_manifest_path: Path,
    device: torch.device,
    esm_model,
    esm_dict,
    af2_to_esm,
) -> None:
    from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
    from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
    from processor.protein_processor import ProteinDataProcessor
    from utils.datamodule_utils import process_one_inference_structure

    manifest = load_json(split_manifest_path)
    tokenizer = BoltzTokenizer()
    featurizer = BoltzFeaturizer()
    processor = ProteinDataProcessor(
        device=device,
        scale=16.0,
        ref_scale=5.0,
        multiplicity=1,
        inference_multiplicity=1,
        backend="torch",
    )
    esm_dir = ensure_dir(split_dir / "esm")

    for sample in manifest.get("samples", []):
        target_id = sample["target_id"]
        target_variants = [target_id, target_id.lower(), target_id.upper()]
        processed_structure = find_existing(
            split_dir,
            [f"processed/structures/{variant}.npz" for variant in target_variants],
        )
        processed_record = find_existing(
            split_dir,
            [f"processed/records/{variant}.json" for variant in target_variants],
        )
        if processed_structure is None or processed_record is None:
            continue

        batch, _, _ = process_one_inference_structure(
            processed_structure,
            processed_record,
            tokenizer,
            featurizer,
            processor,
            esm_model,
            esm_dict,
            af2_to_esm,
        )
        esm_path = esm_dir / f"{target_id}.pt"
        torch.save(
            {
                "target_id": target_id,
                "esm_s": batch["esm_s"].cpu(),
                "cropped_num_tokens": batch["cropped_num_tokens"].cpu(),
                "max_num_tokens": batch["max_num_tokens"].cpu(),
                "aa_seq": batch["aa_seq"],
            },
            esm_path,
        )
        sample["esm_feature_path"] = str(Path("esm") / esm_path.name)

        if "processed_structure_path" not in sample:
            sample["processed_structure_path"] = str(processed_structure.relative_to(split_dir))
        if "processed_record_path" not in sample:
            sample["processed_record_path"] = str(processed_record.relative_to(split_dir))

    write_json(split_manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute and cache ESM-3B features into a benchmark bundle.")
    parser.add_argument("--bundle_dir", required=True)
    parser.add_argument("--simplefold_repo", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    simplefold_repo = Path(args.simplefold_repo).resolve()
    device = torch.device(args.device)

    prepare_simplefold_imports(simplefold_repo)
    esm_model, esm_dict, af2_to_esm = load_esm_components(device)

    for split_name in ("train", "val", "test"):
        split_dir = bundle_dir / split_name
        split_manifest_path = split_dir / "manifest.json"
        if not split_manifest_path.exists():
            continue
        maybe_preprocess_split(simplefold_repo, split_dir)
        compute_and_store_esm(
            simplefold_repo=simplefold_repo,
            split_dir=split_dir,
            split_manifest_path=split_manifest_path,
            device=device,
            esm_model=esm_model,
            esm_dict=esm_dict,
            af2_to_esm=af2_to_esm,
        )


if __name__ == "__main__":
    main()
