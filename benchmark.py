from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import time
from pathlib import Path
from types import ModuleType

from sdk.api import (
    RunContext,
    build_bundle_paths,
    ensure_dir,
    load_json,
    prediction_path,
    write_json,
)
from scorer import score_split


class TimeoutExpired(RuntimeError):
    pass


def _handle_timeout(signum, frame):
    raise TimeoutExpired("submission exceeded timeout")


def load_submission_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("submission_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import submission module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_output_contract(context: RunContext) -> dict:
    test_manifest = load_json(context.bundle.test_dir / "manifest.json")
    missing = []
    for sample in test_manifest.get("samples", []):
        target_id = sample["target_id"]
        if not prediction_path(context.output_dir, target_id).exists():
            missing.append(target_id)
    return {"valid": len(missing) == 0, "missing_targets": missing}


def run_submission(context: RunContext, submission_path: Path, config_path: Path) -> dict:
    module = load_submission_module(submission_path)
    config = load_json(config_path) if config_path.exists() else {}

    if not hasattr(module, "main"):
        raise RuntimeError("submission/train.py must define main(context, config)")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.alarm(context.timeout_sec)
    started = time.perf_counter()
    try:
        module.main(context, config)
    finally:
        signal.alarm(0)
    ended = time.perf_counter()
    return {"runtime_sec": ended - started}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sundai protein folding benchmark.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--timeout_sec", type=int, default=600)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(Path(args.output_dir))

    context = RunContext(
        input_dir=input_dir,
        output_dir=output_dir,
        timeout_sec=args.timeout_sec,
        started_at=time.perf_counter(),
        bundle=build_bundle_paths(input_dir),
    )

    outcome = {
        "valid": False,
        "runtime_sec": None,
        "error": None,
        "validation": None,
        "scoring": None,
    }

    try:
        timing = run_submission(context, Path(args.submission), Path(args.config))
        outcome["runtime_sec"] = timing["runtime_sec"]
        outcome["validation"] = validate_output_contract(context)
        if not outcome["validation"]["valid"]:
            outcome["error"] = "missing required prediction outputs"
        else:
            outcome["scoring"] = score_split(
                context.bundle.test_dir / "manifest.json",
                context.output_dir / "predictions",
            )
            outcome["valid"] = bool(outcome["scoring"]["valid"])
    except TimeoutExpired as exc:
        outcome["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        outcome["error"] = f"{type(exc).__name__}: {exc}"

    write_json(output_dir / "results.json", outcome)

    if outcome["scoring"] is not None:
        summary = dict(outcome["scoring"]["summary"])
        summary["runtime_sec"] = outcome["runtime_sec"]
        summary["valid"] = outcome["valid"]
        write_json(output_dir / "summary.json", summary)

    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
