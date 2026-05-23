from __future__ import annotations

import os
import json
import logging
import signal
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from shutil import copytree
from threading import Event
from typing import Callable

from service.logging_utils import env_path


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
    "/Users/ryanznie/Desktop/Important/Work/Sundai/bundles/public_lb_v1",
)
CACHE_SEED_ROOT = env_path(
    "SUNDAI_SIMPLEFOLD_CACHE_SEED_ROOT",
    str(SIMPLEFOLD_ROOT / "artifacts" / "smoke_test" / "cache"),
)
CHECKPOINT_PATH = BUNDLE_ROOT / "checkpoints" / "simplefold_100M.ckpt"


class SubmissionCancelledError(RuntimeError):
    pass


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
        raise FileNotFoundError("uploaded zip must contain baseline_submission/train.py")
    if not config_path.exists():
        raise FileNotFoundError("uploaded zip must contain baseline_submission/config.json")
    LOGGER.info("submission extracted upload_path=%s workspace_dir=%s", upload_path, workspace_dir)
    return submission_path, config_path


def _build_bundle(bundle_dir: Path) -> None:
    copytree(BUNDLE_ROOT, bundle_dir, dirs_exist_ok=True)
    LOGGER.info("live bundle copied bundle_dir=%s", bundle_dir)


def _build_runtime_config(uploaded_config_path: Path, workspace_dir: Path, bundle_dir: Path) -> Path:
    config = json.loads(uploaded_config_path.read_text())
    config.setdefault("track", "simplefold_100M_finetune_under_budget")
    config.setdefault("simplefold_model", "simplefold_100M")
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
    submission_logger: logging.Logger | None = None,
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
            if submission_logger:
                submission_logger.info(stripped)
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
