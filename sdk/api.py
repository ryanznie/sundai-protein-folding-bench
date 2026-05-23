from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BundlePaths:
    root: Path
    train_dir: Path
    val_dir: Path
    test_dir: Path
    checkpoints_dir: Path
    manifest_path: Path


@dataclass
class RunContext:
    input_dir: Path
    output_dir: Path
    timeout_sec: int
    started_at: float
    bundle: BundlePaths

    def seconds_remaining(self) -> float:
        return max(0.0, self.timeout_sec - (time.perf_counter() - self.started_at))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def build_bundle_paths(input_dir: Path) -> BundlePaths:
    return BundlePaths(
        root=input_dir,
        train_dir=input_dir / "train",
        val_dir=input_dir / "val",
        test_dir=input_dir / "test",
        checkpoints_dir=input_dir / "checkpoints",
        manifest_path=input_dir / "manifest.json",
    )


def list_split_targets(split_dir: Path) -> list[str]:
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    manifest = load_json(manifest_path)
    return [sample["target_id"] for sample in manifest.get("samples", [])]


def prediction_path(output_dir: Path, target_id: str) -> Path:
    return output_dir / "predictions" / f"{target_id}_sampled_0.cif"


def default_results_path(output_dir: Path) -> Path:
    return output_dir / "results.json"
