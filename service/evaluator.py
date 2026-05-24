from __future__ import annotations

import os
import json
import logging
import signal
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from shutil import copyfile
from shutil import copytree
from threading import Event
from typing import Callable

from scorer import score_split
from sdk.simplefold import cached_runner_command
from service.logging_utils import append_submission_progress, env_path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("sundai.evaluator")
SIMPLEFOLD_ROOT = env_path(
    "SUNDAI_SIMPLEFOLD_ROOT",
    "/Users/ryanznie/Desktop/Important/Work/Sundai/ml-simplefold",
)
DEFAULT_SIMPLEFOLD_PYTHON = SIMPLEFOLD_ROOT / ".venv" / "bin" / "python"
SIMPLEFOLD_PYTHON = Path(
    os.environ.get(
        "SUNDAI_SIMPLEFOLD_PYTHON",
        str(DEFAULT_SIMPLEFOLD_PYTHON if DEFAULT_SIMPLEFOLD_PYTHON.exists() else sys.executable),
    )
).expanduser()
BUNDLE_ROOT = env_path(
    "SUNDAI_BUNDLE_ROOT",
    "/Users/ryanznie/Desktop/Important/Work/Sundai/bundles/simplefold_hackathon_v1",
)
CACHE_SEED_ROOT = env_path(
    "SUNDAI_SIMPLEFOLD_CACHE_SEED_ROOT",
    str(SIMPLEFOLD_ROOT / "artifacts" / "smoke_test" / "cache"),
)
CHECKPOINT_PATH = BUNDLE_ROOT / "checkpoints" / "simplefold_100M.ckpt"


class SubmissionCancelledError(RuntimeError):
    pass


def should_record_submission_progress(line: str) -> bool:
    return line.startswith("[progress]")


def evaluator_assets_ready() -> tuple[bool, list[str]]:
    required = [
        BUNDLE_ROOT / "manifest.json",
        BUNDLE_ROOT / "test" / "manifest.json",
        CHECKPOINT_PATH,
        CACHE_SEED_ROOT / "ccd.pkl",
        CACHE_SEED_ROOT / "boltz1_conf.ckpt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    LOGGER.info(
        "evaluator assets check ready=%s bundle_root=%s simplefold_root=%s cache_seed_root=%s missing=%s",
        len(missing) == 0,
        BUNDLE_ROOT,
        SIMPLEFOLD_ROOT,
        CACHE_SEED_ROOT,
        missing,
    )
    return len(missing) == 0, missing


def _extract_submission(upload_path: Path, workspace_dir: Path) -> tuple[Path, Path]:
    extract_dir = workspace_dir / "submission_src"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(upload_path) as archive:
        archive.extractall(extract_dir)

    submission_path = extract_dir / "submission" / "train.py"
    config_path = extract_dir / "submission" / "config.json"
    if not submission_path.exists():
        raise FileNotFoundError("uploaded zip must contain submission/train.py")
    if not config_path.exists():
        raise FileNotFoundError("uploaded zip must contain submission/config.json")
    LOGGER.info("submission extracted upload_path=%s workspace_dir=%s", upload_path, workspace_dir)
    return submission_path, config_path


def sanitize_submission_bundle(bundle_dir: Path) -> None:
    manifest_path = bundle_dir / "test" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["samples"] = []
    manifest_path.write_text(json.dumps(manifest, indent=2))

    for dirname in ("references", "processed", "samples", "esm", "fastas"):
        target_dir = bundle_dir / "test" / dirname
        if target_dir.exists():
            for path in sorted(target_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            target_dir.rmdir()


def _build_bundle(bundle_dir: Path) -> None:
    copytree(BUNDLE_ROOT, bundle_dir, dirs_exist_ok=True)
    sanitize_submission_bundle(bundle_dir)
    LOGGER.info("live bundle copied bundle_dir=%s", bundle_dir)
    LOGGER.info("submission bundle sanitized test references removed bundle_dir=%s", bundle_dir)


def _build_runtime_config(uploaded_config_path: Path, workspace_dir: Path, bundle_dir: Path) -> Path:
    config = json.loads(uploaded_config_path.read_text())
    config.setdefault("num_steps", 50)
    config.setdefault("tau", 0.01)
    config["nsample_per_protein"] = 1
    config["backend"] = "torch"
    config["plddt"] = bool(config.get("plddt", False))
    config["simplefold_command"] = [
        str(SIMPLEFOLD_PYTHON),
        "-c",
        "from simplefold.cli import main; main()",
    ]
    config["simplefold_workdir"] = str(SIMPLEFOLD_ROOT)
    config["simplefold_env"] = {
        "PYTHONPATH": str(SIMPLEFOLD_ROOT / "src"),
        "MPLCONFIGDIR": str(workspace_dir / "mplconfig"),
    }
    config["ckpt_dir"] = str(bundle_dir / "checkpoints")
    config["simplefold_cache_seed_dir"] = str(CACHE_SEED_ROOT)
    config_path = workspace_dir / "runtime_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    LOGGER.info("runtime config built config_path=%s config=%s", config_path, config)
    return config_path


def _load_runtime_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text())


def _resolve_hidden_ckpt_dir(
    *,
    workspace_dir: Path,
    output_dir: Path,
    model_name: str,
    fallback_ckpt_dir: Path,
) -> Path:
    output_ckpt_dir = output_dir / "checkpoints"
    named_ckpt = output_ckpt_dir / f"{model_name}.ckpt"
    if named_ckpt.exists():
        return output_ckpt_dir

    adapted_ckpt = output_ckpt_dir / "adapted.ckpt"
    if adapted_ckpt.exists():
        hidden_ckpt_dir = workspace_dir / "hidden_checkpoints"
        hidden_ckpt_dir.mkdir(parents=True, exist_ok=True)
        copyfile(adapted_ckpt, hidden_ckpt_dir / f"{model_name}.ckpt")
        return hidden_ckpt_dir

    return fallback_ckpt_dir


def run_hidden_test_inference(
    *,
    workspace_dir: Path,
    output_dir: Path,
    runtime_config: dict,
    submission_log_path: Path | None,
    cancel_event: Event | None,
) -> dict:
    started = time.perf_counter()
    hidden_output_dir = workspace_dir / "hidden_test_output"
    hidden_output_dir.mkdir(parents=True, exist_ok=True)

    model_name = str(runtime_config.get("simplefold_model", "simplefold_100M"))
    hidden_ckpt_dir = _resolve_hidden_ckpt_dir(
        workspace_dir=workspace_dir,
        output_dir=output_dir,
        model_name=model_name,
        fallback_ckpt_dir=BUNDLE_ROOT / "checkpoints",
    )

    hidden_config = dict(runtime_config)
    hidden_config["ckpt_dir"] = str(hidden_ckpt_dir)

    args = [
        *cached_runner_command(hidden_config),
        "--manifest_path",
        str(BUNDLE_ROOT / "test" / "manifest.json"),
        "--split_dir",
        str(BUNDLE_ROOT / "test"),
        "--simplefold_model",
        model_name,
        "--num_steps",
        str(int(hidden_config.get("num_steps", 500))),
        "--tau",
        str(hidden_config.get("tau", 0.01)),
        "--nsample_per_protein",
        "1",
        "--ckpt_dir",
        str(hidden_ckpt_dir),
        "--output_dir",
        str(hidden_output_dir),
        "--backend",
        "torch",
    ]
    if bool(hidden_config.get("plddt", False)):
        args.append("--plddt")

    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in (hidden_config.get("simplefold_env") or {}).items()})

    LOGGER.info("starting hidden test inference command=%s", args)
    process = subprocess.Popen(
        args,
        cwd=str(hidden_config.get("simplefold_workdir") or SIMPLEFOLD_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=env,
    )
    hidden_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        if cancel_event and cancel_event.is_set():
            LOGGER.info("hidden inference cancellation requested pid=%s", process.pid)
            terminate_process_tree(process)
        stripped = line.rstrip()
        hidden_lines.append(line)
        if submission_log_path and should_record_submission_progress(stripped):
            append_submission_progress(
                submission_log_path,
                stripped.removeprefix("[progress] ").strip(),
            )
        LOGGER.info("hidden inference stream %s", stripped)
    returncode = process.wait()
    combined_output = "".join(hidden_lines)
    if returncode != 0:
        raise RuntimeError(
            "hidden test inference failed\n"
            f"stdout:\n{combined_output}\n"
        )

    prediction_root = hidden_output_dir / f"predictions_{model_name}"
    normalized_prediction_dir = hidden_output_dir / "predictions"
    normalized_prediction_dir.mkdir(parents=True, exist_ok=True)
    for sample in json.loads((BUNDLE_ROOT / "test" / "manifest.json").read_text()).get("samples", []):
        target_id = sample["target_id"]
        source = prediction_root / f"{target_id.lower()}_sampled_0.cif"
        if not source.exists():
            source = prediction_root / f"{target_id}_sampled_0.cif"
        if not source.exists():
            raise RuntimeError(f"hidden test inference produced no CIF output for {target_id}")
        copyfile(source, normalized_prediction_dir / f"{target_id}_sampled_0.cif")

    scoring = score_split(BUNDLE_ROOT / "test" / "manifest.json", normalized_prediction_dir)
    return {
        "scoring": scoring,
        "stdout": combined_output,
        "runtime_sec": time.perf_counter() - started,
    }


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        process.terminate()


def run_uploaded_submission(
    upload_path: Path,
    *,
    cancel_event: Event | None = None,
    process_started: Callable[[subprocess.Popen[str]], None] | None = None,
    submission_log_path: Path | None = None,
) -> dict:
    ready, missing = evaluator_assets_ready()
    if not ready:
        raise FileNotFoundError(f"missing evaluator assets: {missing}")
    if cancel_event and cancel_event.is_set():
        raise SubmissionCancelledError("submission cancelled before benchmark start")

    with tempfile.TemporaryDirectory(prefix="sundai_live_eval_") as tmp_dir:
        workspace_dir = Path(tmp_dir)
        bundle_dir = workspace_dir / "input_bundle"
        output_dir = workspace_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        submission_path, uploaded_config_path = _extract_submission(upload_path, workspace_dir)
        _build_bundle(bundle_dir)
        config_path = _build_runtime_config(uploaded_config_path, workspace_dir, bundle_dir)
        runtime_config = _load_runtime_config(config_path)
        if submission_log_path:
            append_submission_progress(submission_log_path, "starting run for train/val")

        command = [
            sys.executable,
            str(REPO_ROOT / "benchmark.py"),
            "--input_dir",
            str(bundle_dir),
            "--output_dir",
            str(output_dir),
            "--submission",
            str(submission_path),
            "--config",
            str(config_path),
            "--skip_scoring",
            "--timeout_sec",
            "1800",
        ]
        LOGGER.info(
            "starting live benchmark upload_path=%s workspace_dir=%s command=%s",
            upload_path,
            workspace_dir,
            command,
        )
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        if process_started is not None:
            process_started(process)
        output_lines: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            if cancel_event and cancel_event.is_set():
                LOGGER.info("benchmark cancellation requested pid=%s", process.pid)
                terminate_process_tree(process)
            stripped = line.rstrip()
            output_lines.append(line)
            if submission_log_path and should_record_submission_progress(stripped):
                append_submission_progress(
                    submission_log_path,
                    stripped.removeprefix("[progress] ").strip(),
                )
            LOGGER.info("benchmark stream %s", stripped)
        returncode = process.wait()
        combined_output = "".join(output_lines)
        if cancel_event and cancel_event.is_set():
            raise SubmissionCancelledError("submission cancelled during benchmark execution")

        results_path = output_dir / "results.json"
        if not results_path.exists():
            raise RuntimeError(
                "benchmark did not produce results.json\n"
                f"stdout:\n{combined_output}\n\nstderr:\n"
            )
        results = json.loads(results_path.read_text())
        validation = results.get("validation") or {}
        if results.get("error") is None and validation.get("valid"):
            hidden_result = run_hidden_test_inference(
                workspace_dir=workspace_dir,
                output_dir=output_dir,
                runtime_config=runtime_config,
                submission_log_path=submission_log_path,
                cancel_event=cancel_event,
            )
            results["scoring"] = hidden_result["scoring"]
            results["valid"] = bool(hidden_result["scoring"]["valid"])
            combined_output = combined_output + hidden_result["stdout"]
            results["runtime_sec"] = float(results.get("runtime_sec") or 0.0) + float(hidden_result["runtime_sec"])
        results["stdout"] = combined_output
        results["stderr"] = ""
        results["returncode"] = returncode
        LOGGER.info(
            "live benchmark finished returncode=%s valid=%s runtime_sec=%s error=%s",
            returncode,
            results.get("valid"),
            results.get("runtime_sec"),
            results.get("error"),
        )
        return results
