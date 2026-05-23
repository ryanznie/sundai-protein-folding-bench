from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sdk.structures import compute_alignment_metrics, load_ca_trace


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
    matched_residues: int | None = None
    reference_residues: int | None = None
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


def score_target(sample: dict, predictions_dir: Path, split_dir: Path) -> TargetScore:
    target_id = sample["target_id"]
    prediction_path = predictions_dir / f"{target_id}_sampled_0.cif"
    valid, invalid_reason = validate_prediction_exists(prediction_path)
    if not valid:
        return TargetScore(
            target_id=target_id,
            valid=False,
            coverage=0.0,
            invalid_reason=invalid_reason,
        )

    reference_field = sample.get("reference_structure_path")
    if not reference_field:
        return TargetScore(
            target_id=target_id,
            valid=False,
            coverage=0.0,
            invalid_reason="missing_reference_structure",
        )

    reference_path = Path(reference_field)
    if not reference_path.is_absolute():
        reference_path = split_dir / reference_field

    try:
        reference_trace = load_ca_trace(reference_path)
        prediction_trace = load_ca_trace(prediction_path)
        metrics = compute_alignment_metrics(reference_trace, prediction_trace)
    except FileNotFoundError:
        return TargetScore(
            target_id=target_id,
            valid=False,
            coverage=0.0,
            invalid_reason="missing_reference_structure",
        )
    except ValueError as exc:
        return TargetScore(
            target_id=target_id,
            valid=False,
            coverage=0.0,
            invalid_reason=f"parse_error:{exc}",
        )

    min_coverage = float(sample.get("min_coverage", 0.95))
    score = TargetScore(
        target_id=target_id,
        valid=True,
        coverage=metrics.coverage,
        tm_score=metrics.tm_score,
        lddt=metrics.lddt,
        rmsd=metrics.rmsd,
        ca_rmsd=metrics.ca_rmsd,
        gdt_ts_like=metrics.gdt_ts_like,
        matched_residues=metrics.matched_residues,
        reference_residues=metrics.reference_residues,
        invalid_reason=None,
    )
    return score


def score_split(test_manifest_path: Path, predictions_dir: Path) -> dict:
    manifest = load_manifest(test_manifest_path)
    split_dir = test_manifest_path.parent
    samples = manifest.get("samples", [])
    total_targets = len(samples)
    scores = []
    for i, sample in enumerate(samples, 1):
        target_id = sample.get("target_id", "unknown")
        print(f"[benchmark] scoring target [{i}/{total_targets}] {target_id}", flush=True)
        scores.append(score_target(sample, predictions_dir, split_dir))

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
            "num_below_coverage_threshold": sum(
                1
                for sample, score in zip(manifest.get("samples", []), scores)
                if score.coverage < float(sample.get("min_coverage", 0.95))
            ),
            "min_coverage": min((s.coverage for s in scores), default=0.0),
            "mean_tm_score": mean_or_none("tm_score"),
            "mean_lddt": mean_or_none("lddt"),
            "mean_rmsd": mean_or_none("rmsd"),
            "mean_ca_rmsd": mean_or_none("ca_rmsd"),
            "mean_gdt_ts_like": mean_or_none("gdt_ts_like"),
        },
    }
