from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from sdk.api import ensure_dir, prediction_path

REPO_ROOT = Path(__file__).resolve().parents[1]


def simplefold_command(config: dict[str, Any]) -> list[str]:
    raw = config.get("simplefold_command")
    if isinstance(raw, list) and raw:
        return [str(part) for part in raw]
    env = os.environ.get("SIMPLEFOLD_BIN")
    if env:
        return [env]
    return ["simplefold"]


def cached_runner_command(config: dict[str, Any]) -> list[str]:
    python_command = simplefold_command(config)
    if len(python_command) >= 3 and python_command[1] == "-c":
        return [python_command[0], str(REPO_ROOT / "tools" / "run_simplefold_cached.py")]
    return [*python_command, str(REPO_ROOT / "tools" / "run_simplefold_cached.py")]


def can_use_cached_bundle_features(samples: list[dict[str, Any]]) -> bool:
    required_fields = (
        "esm_feature_path",
        "processed_structure_path",
        "processed_record_path",
    )
    return bool(samples) and all(all(sample.get(field) for field in required_fields) for sample in samples)


def run_simplefold_inference(
    *,
    fasta_path: Path,
    target_id: str,
    destination_path: Path,
    config: dict[str, Any],
) -> None:
    command = simplefold_command(config)
    model_name = str(config.get("simplefold_model", "simplefold_100M"))
    num_steps = int(config.get("num_steps", 500))
    tau = str(config.get("tau", 0.01))
    backend = str(config.get("backend", "torch"))
    nsample = int(config.get("nsample_per_protein", 1))
    ckpt_dir = config.get("ckpt_dir")
    simplefold_workdir = config.get("simplefold_workdir")
    simplefold_env = config.get("simplefold_env") or {}
    plddt = bool(config.get("plddt", False))
    cache_seed_dir = config.get("simplefold_cache_seed_dir")

    with tempfile.TemporaryDirectory(prefix=f"{target_id}_simplefold_") as tmp_dir:
        output_dir = Path(tmp_dir)
        cache_dir = ensure_dir(output_dir / "cache")
        if cache_seed_dir:
            seed_root = Path(str(cache_seed_dir))
            for filename in ("ccd.pkl", "boltz1_conf.ckpt"):
                source = seed_root / filename
                if source.exists():
                    shutil.copyfile(source, cache_dir / filename)
        args = [
            *command,
            "--simplefold_model",
            model_name,
            "--num_steps",
            str(num_steps),
            "--tau",
            tau,
            "--nsample_per_protein",
            str(nsample),
            "--ckpt_dir",
            str(ckpt_dir or "artifacts"),
            "--fasta_path",
            str(fasta_path),
            "--output_dir",
            str(output_dir),
            "--backend",
            backend,
        ]
        if plddt:
            args.append("--plddt")
        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in simplefold_env.items()})
        subprocess.run(
            args,
            check=True,
            cwd=str(simplefold_workdir) if simplefold_workdir else None,
            env=env,
        )

        candidates = sorted(output_dir.rglob("*.cif"))
        if not candidates:
            raise RuntimeError(f"simplefold produced no CIF output for {target_id}")

        ensure_dir(destination_path.parent)
        shutil.copyfile(candidates[0], destination_path)


def run_simplefold_batch_inference(
    *,
    fasta_root: Path,
    target_ids: list[str],
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    command = simplefold_command(config)
    model_name = str(config.get("simplefold_model", "simplefold_100M"))
    num_steps = int(config.get("num_steps", 500))
    tau = str(config.get("tau", 0.01))
    backend = str(config.get("backend", "torch"))
    nsample = int(config.get("nsample_per_protein", 1))
    ckpt_dir = config.get("ckpt_dir")
    simplefold_workdir = config.get("simplefold_workdir")
    simplefold_env = config.get("simplefold_env") or {}
    plddt = bool(config.get("plddt", False))
    cache_seed_dir = config.get("simplefold_cache_seed_dir")

    with tempfile.TemporaryDirectory(prefix="simplefold_batch_") as tmp_dir:
        simplefold_output_dir = Path(tmp_dir)
        cache_dir = ensure_dir(simplefold_output_dir / "cache")
        if cache_seed_dir:
            seed_root = Path(str(cache_seed_dir))
            for filename in ("ccd.pkl", "boltz1_conf.ckpt"):
                source = seed_root / filename
                if source.exists():
                    shutil.copyfile(source, cache_dir / filename)

        args = [
            *command,
            "--simplefold_model",
            model_name,
            "--num_steps",
            str(num_steps),
            "--tau",
            tau,
            "--nsample_per_protein",
            str(nsample),
            "--ckpt_dir",
            str(ckpt_dir or "artifacts"),
            "--fasta_path",
            str(fasta_root),
            "--output_dir",
            str(simplefold_output_dir),
            "--backend",
            backend,
        ]
        if plddt:
            args.append("--plddt")

        env = os.environ.copy()
        env.update({str(key): str(value) for key, value in simplefold_env.items()})
        print(
            f"[simplefold] launching batch inference target_count={len(target_ids)} "
            f"model={model_name} steps={num_steps} backend={backend}",
            flush=True,
        )
        print(
            f"[simplefold] command={' '.join(args)} cwd={str(simplefold_workdir) if simplefold_workdir else os.getcwd()}",
            flush=True,
        )
        subprocess.run(
            args,
            check=True,
            cwd=str(simplefold_workdir) if simplefold_workdir else None,
            env=env,
        )
        print("[simplefold] inference command finished, normalizing output files", flush=True)

        prediction_root = simplefold_output_dir / f"predictions_{model_name}"
        for target_id in target_ids:
            source = prediction_root / f"{target_id}_sampled_0.cif"
            if not source.exists():
                raise RuntimeError(f"simplefold produced no CIF output for {target_id}")
            destination = prediction_path(output_dir, target_id)
            ensure_dir(destination.parent)
            shutil.copyfile(source, destination)
            print(f"[simplefold] copied prediction target_id={target_id} destination={destination}", flush=True)


def run_simplefold_cached_bundle_inference(
    *,
    split_dir: Path,
    samples: list[dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    args = [
        *cached_runner_command(config),
        "--manifest_path",
        str(split_dir / "manifest.json"),
        "--split_dir",
        str(split_dir),
        "--simplefold_model",
        str(config.get("simplefold_model", "simplefold_100M")),
        "--num_steps",
        str(int(config.get("num_steps", 500))),
        "--tau",
        str(config.get("tau", 0.01)),
        "--nsample_per_protein",
        str(int(config.get("nsample_per_protein", 1))),
        "--ckpt_dir",
        str(config.get("ckpt_dir") or "artifacts"),
        "--output_dir",
        str(output_dir),
        "--backend",
        str(config.get("backend", "torch")),
    ]
    if bool(config.get("plddt", False)):
        args.append("--plddt")

    simplefold_workdir = config.get("simplefold_workdir")
    simplefold_env = config.get("simplefold_env") or {}
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in simplefold_env.items()})
    print(
        f"[simplefold] using cached bundle features sample_count={len(samples)} "
        f"model={config.get('simplefold_model', 'simplefold_100M')} steps={config.get('num_steps', 500)}",
        flush=True,
    )
    print(
        f"[simplefold] cached runner command={' '.join(args)} cwd={str(simplefold_workdir) if simplefold_workdir else os.getcwd()}",
        flush=True,
    )
    subprocess.run(
        args,
        check=True,
        cwd=str(simplefold_workdir) if simplefold_workdir else None,
        env=env,
    )

    prediction_root = output_dir / f"predictions_{config.get('simplefold_model', 'simplefold_100M')}"
    for sample in samples:
        target_id = sample["target_id"]
        source = prediction_root / f"{target_id.lower()}_sampled_0.cif"
        if not source.exists():
            source = prediction_root / f"{target_id}_sampled_0.cif"
        if not source.exists():
            raise RuntimeError(f"cached simplefold produced no CIF output for {target_id}")
        destination = prediction_path(output_dir, target_id)
        ensure_dir(destination.parent)
        shutil.copyfile(source, destination)
        print(f"[simplefold] copied cached prediction target_id={target_id} destination={destination}", flush=True)


def write_predictions_from_manifest(context: Any, config: dict[str, Any]) -> None:
    manifest = context.bundle.test_manifest()
    for sample in manifest.get("samples", []):
        target_id = sample["target_id"]
        fasta_path = context.bundle.resolve_path(
            context.bundle.test_dir,
            sample.get("sequence_fasta_path") or sample.get("fasta_path"),
        )
        if fasta_path is None:
            raise RuntimeError(
                f"test manifest sample {target_id} is missing sequence_fasta_path"
            )
        run_simplefold_inference(
            fasta_path=fasta_path,
            target_id=target_id,
            destination_path=prediction_path(context.output_dir, target_id),
            config=config,
        )
