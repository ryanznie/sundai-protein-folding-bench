from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TargetScore:
    target_id: str
    valid: bool
    coverage: float
    tm_score: float | None = None
    lddt: float | None = None
    rmsd: float | None = None
    ca_rmsd: float | None = None
    gdt_ts_like: float | None = None
    invalid_reason: str | None = None


def load_manifest(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def validate_prediction_exists(prediction_path: Path) -> tuple[bool, str | None]:
    if not prediction_path.exists():
        return False, "missing_prediction"
    if prediction_path.stat().st_size == 0:
        return False, "empty_prediction"
    return True, None


def score_split(test_manifest_path: Path, predictions_dir: Path) -> dict:
    manifest = load_manifest(test_manifest_path)
    scores: list[TargetScore] = []

    for sample in manifest.get("samples", []):
        target_id = sample["target_id"]
        prediction_path = predictions_dir / f"{target_id}_sampled_0.cif"
        valid, invalid_reason = validate_prediction_exists(prediction_path)

        score = TargetScore(
            target_id=target_id,
            valid=valid,
            coverage=1.0 if valid else 0.0,
            invalid_reason=invalid_reason,
        )

        # Placeholder metric path:
        # If the bundle includes precomputed public metrics or reference paths,
        # the benchmark owner can replace this block with real scoring.
        if valid and "public_metrics" in sample:
            metrics = sample["public_metrics"]
            score.tm_score = metrics.get("tm_score")
            score.lddt = metrics.get("lddt")
            score.rmsd = metrics.get("rmsd")
            score.ca_rmsd = metrics.get("ca_rmsd")
            score.gdt_ts_like = metrics.get("gdt_ts_like")

        scores.append(score)

    valid_scores = [s for s in scores if s.valid]
    invalid_scores = [s for s in scores if not s.valid]

    def mean_or_none(key: str):
        values = [
            getattr(score, key)
            for score in valid_scores
            if getattr(score, key) is not None
        ]
        if not values:
            return None
        return float(sum(values) / len(values))

    return {
        "valid": len(invalid_scores) == 0,
        "targets": [asdict(score) for score in scores],
        "summary": {
            "num_targets": len(scores),
            "num_valid_targets": len(valid_scores),
            "num_invalid_targets": len(invalid_scores),
            "min_coverage": min((s.coverage for s in scores), default=0.0),
            "mean_tm_score": mean_or_none("tm_score"),
            "mean_lddt": mean_or_none("lddt"),
            "mean_rmsd": mean_or_none("rmsd"),
            "mean_ca_rmsd": mean_or_none("ca_rmsd"),
            "mean_gdt_ts_like": mean_or_none("gdt_ts_like"),
        },
    }
