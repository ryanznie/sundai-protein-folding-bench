from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def post_json(url: str, payload: dict) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        if response.status >= 300:
            raise RuntimeError(f"failed callback to {url}: {response.status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one benchmark submission inside Docker.")
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--submission-zip", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--benchmark-image", required=True)
    parser.add_argument("--api-base-url", required=True)
    parser.add_argument("--repo-dir", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    submission_zip = Path(args.submission_zip).resolve()
    input_dir = Path(args.input_dir).resolve()
    repo_dir = Path(args.repo_dir).resolve()

    with tempfile.TemporaryDirectory(prefix=f"submission_{args.submission_id}_") as tmp_dir:
        workspace_dir = Path(tmp_dir) / "workspace"
        output_dir = Path(tmp_dir) / "output"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(submission_zip) as archive:
            archive.extractall(workspace_dir)

        benchmark_repo = workspace_dir / "benchmark_repo"
        shutil.copytree(repo_dir, benchmark_repo, dirs_exist_ok=True)
        extracted_submission_dir = workspace_dir / "submission"
        if extracted_submission_dir.exists():
            shutil.copytree(
                extracted_submission_dir,
                benchmark_repo / "submission",
                dirs_exist_ok=True,
            )

        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--network",
                "none",
                "-v",
                f"{input_dir}:/input:ro",
                "-v",
                f"{output_dir}:/output",
                "-v",
                f"{benchmark_repo}:/workspace",
                args.benchmark_image,
                "bash",
                "/workspace/benchmark.sh",
            ],
            check=True,
        )

        results = json.loads((output_dir / "results.json").read_text())
        payload = {
            "status": "completed" if results.get("valid") else "failed",
            "valid": bool(results.get("valid")),
            "runtime_sec": results.get("runtime_sec"),
            "error": results.get("error"),
            "summary": results.get("scoring", {}).get("summary") if results.get("scoring") else None,
            "targets": results.get("scoring", {}).get("targets") if results.get("scoring") else None,
        }
        post_json(
            f"{args.api_base_url.rstrip('/')}/internal/submissions/{args.submission_id}/complete",
            payload,
        )


if __name__ == "__main__":
    main()
