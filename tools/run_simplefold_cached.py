from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import lightning.pytorch as pl
import torch


def _prime_simplefold_sys_path() -> None:
    for entry in list(sys.path):
        candidate = Path(entry)
        if candidate.name == "src" and (candidate / "simplefold").exists():
            package_root = candidate / "simplefold"
            package_root_str = str(package_root)
            if package_root_str not in sys.path:
                sys.path.append(package_root_str)


_prime_simplefold_sys_path()

from simplefold.inference import (
    generate_structure,
    initialize_folding_model,
    initialize_others,
    initialize_plddt_module,
)
from utils.boltz_utils import process_structure, save_structure
from utils.datamodule_utils import Record, collate, extract_sequence_from_tokens
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from boltz_data_pipeline.types import Input, Structure


def resolve_bundle_path(split_dir: Path, relative_path: str) -> Path:
    path = split_dir / relative_path
    if path.exists():
        return path
    lowered = split_dir / "/".join(part.lower() for part in Path(relative_path).parts)
    if lowered.exists():
        return lowered
    raise FileNotFoundError(f"bundle path not found: {relative_path}")


def build_batch(
    *,
    structure_path: Path,
    record_path: Path,
    esm_path: Path,
    tokenizer: BoltzTokenizer,
    featurizer: BoltzFeaturizer,
    processor,
):
    structure = Structure.load(structure_path)
    input_data = Input(structure, {})
    record = json.loads(record_path.read_text())

    tokenized = tokenizer.tokenize(input_data)
    sequence = extract_sequence_from_tokens(tokenized)
    features = featurizer.process(tokenized)

    esm_payload = torch.load(esm_path, map_location="cpu", weights_only=False)
    features["aa_seq"] = sequence
    features["record"] = record
    features["num_repeats"] = torch.tensor(1)
    features["max_num_tokens"] = esm_payload["max_num_tokens"].reshape(())
    features["cropped_num_tokens"] = esm_payload["cropped_num_tokens"].reshape(())

    batch = collate([features])
    batch["esm_s"] = esm_payload["esm_s"]
    batch = processor.preprocess_inference(batch)
    return batch, structure, Record(**record)


def run_cached_inference(args: argparse.Namespace) -> None:
    manifest = json.loads(Path(args.manifest_path).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / f"predictions_{args.simplefold_model}"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(args.seed, workers=True)

    model, device = initialize_folding_model(args)
    plddt_latent_module, plddt_out_module = initialize_plddt_module(args, device)
    tokenizer, featurizer, processor, flow, sampler = initialize_others(args, device)

    split_dir = Path(args.split_dir)
    for sample in manifest.get("samples", []):
        target_id = sample["target_id"]
        structure_path = resolve_bundle_path(split_dir, sample["processed_structure_path"])
        record_path = resolve_bundle_path(split_dir, sample["processed_record_path"])
        esm_path = resolve_bundle_path(split_dir, sample["esm_feature_path"])
        print(
            f"[cached-simplefold] target={target_id} structure={structure_path.name} "
            f"record={record_path.name} esm={esm_path.name}",
            flush=True,
        )
        batch, structure, record = build_batch(
            structure_path=structure_path,
            record_path=record_path,
            esm_path=esm_path,
            tokenizer=tokenizer,
            featurizer=featurizer,
            processor=processor,
        )
        sampled_coord, pad_mask, plddts = generate_structure(
            args,
            batch,
            sampler,
            flow,
            processor,
            model,
            plddt_latent_module,
            plddt_out_module,
            device,
        )
        for i in range(args.nsample_per_protein):
            sampled_coord_i = sampled_coord[i]
            pad_mask_i = pad_mask[i]
            structure_save = process_structure(
                deepcopy(structure),
                sampled_coord_i,
                pad_mask_i,
                record,
                backend=args.backend,
            )
            outname = f"{record.id}_sampled_{i}"
            save_structure(
                structure_save,
                prediction_dir,
                outname,
                output_format=args.output_format,
                plddts=plddts[i] if plddts is not None else None,
            )
            print(f"[cached-simplefold] wrote prediction target={target_id} sample={i}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SimpleFold inference from cached bundle features.")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--simplefold_model", type=str, default="simplefold_100M")
    parser.add_argument("--ckpt_dir", type=str, default="artifacts")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--nsample_per_protein", type=int, default=1)
    parser.add_argument("--plddt", action="store_true")
    parser.add_argument("--output_format", type=str, default="mmcif", choices=["pdb", "mmcif"])
    parser.add_argument("--backend", type=str, default="torch", choices=["torch", "mlx"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_cached_inference(args)


if __name__ == "__main__":
    main()
