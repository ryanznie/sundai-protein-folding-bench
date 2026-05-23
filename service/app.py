from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from shutil import copyfileobj
from time import sleep
from zipfile import BadZipFile, ZipFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from service.evaluator import (
    SubmissionCancelledError,
    evaluator_assets_ready,
    run_uploaded_submission,
    terminate_process_tree,
)
from service.logging_utils import (
    build_submission_logger,
    configure_logging,
    env_path,
    submission_log_path,
)


configure_logging(os.environ.get("SUNDAI_LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("sundai.service")

DB_PATH = env_path("SUNDAI_DB_PATH", str(Path(__file__).with_name("leaderboard.db")))
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
WEB_ROOT = Path(__file__).with_name("web")
UPLOAD_ROOT = env_path("SUNDAI_UPLOAD_ROOT", str(Path(__file__).with_name("uploads")))
LOG_ROOT = env_path("SUNDAI_LOG_ROOT", str(Path(__file__).with_name("logs")))
JOB_STATE_LOCK = threading.Lock()
JOB_CANCEL_EVENTS: dict[str, threading.Event] = {}
JOB_PROCESSES: dict[str, subprocess.Popen[str]] = {}


class SubmissionCreate(BaseModel):
    team_name: str = Field(min_length=1)
    created_by: str = Field(min_length=1)
    storage_key: str = Field(min_length=1)
    runtime_spec: str = Field(default="docker/worker/runtime-spec.json")


class SubmissionResult(BaseModel):
    status: str
    valid: bool
    runtime_sec: float | None = None
    error: str | None = None
    summary: dict | None = None
    targets: list[dict] | None = None


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    LOGGER.info("initializing service db_path=%s upload_root=%s log_root=%s", DB_PATH, UPLOAD_ROOT, LOG_ROOT)
    with connect() as connection:
        connection.executescript(SCHEMA_PATH.read_text())
        existing_columns = {
            row["name"]
            for row in connection.execute("pragma table_info(submissions)").fetchall()
        }
        if "original_filename" not in existing_columns:
            connection.execute("alter table submissions add column original_filename text")
        if "config_json" not in existing_columns:
            connection.execute("alter table submissions add column config_json text")
        connection.commit()


def load_submission_config_from_zip(storage_path: Path) -> dict | None:
    try:
        with ZipFile(storage_path) as archive:
            with archive.open("starter/config.json") as handle:
                return json.loads(handle.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("failed to read submission config from zip path=%s error=%s", storage_path, exc)
        return None


def upsert_team(connection: sqlite3.Connection, team_name: str, created_by: str) -> tuple[str, str]:
    team_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, team_name))
    user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, created_by))
    connection.execute(
        """
        insert into teams(id, name)
        values(?, ?)
        on conflict(id) do update set name=excluded.name
        """,
        (team_id, team_name),
    )
    connection.execute(
        """
        insert into users(id, email, display_name, team_id)
        values(?, ?, ?, ?)
        on conflict(id) do update set email=excluded.email, display_name=excluded.display_name, team_id=excluded.team_id
        """,
        (user_id, created_by, created_by, team_id),
    )
    return team_id, user_id


def set_submission_status(submission_id: str, *, status: str, valid: int | None = None, runtime_sec: float | None = None, invalid_reason: str | None = None) -> None:
    clauses = ["status = ?"]
    values: list[object] = [status]
    if valid is not None:
        clauses.append("valid = ?")
        values.append(valid)
    if runtime_sec is not None:
        clauses.append("runtime_sec = ?")
        values.append(runtime_sec)
    if invalid_reason is not None:
        clauses.append("invalid_reason = ?")
        values.append(invalid_reason)
    if status in {"completed", "failed", "cancelled"}:
        clauses.append("completed_at = current_timestamp")
    values.append(submission_id)
    with connect() as connection:
        connection.execute(
            f"update submissions set {', '.join(clauses)} where id = ?",
            values,
        )
        connection.commit()
    LOGGER.info(
        "submission status update submission_id=%s status=%s valid=%s runtime_sec=%s invalid_reason=%s",
        submission_id,
        status,
        valid,
        runtime_sec,
        invalid_reason,
    )


def persist_completion(submission_id: str, payload: SubmissionResult) -> None:
    with connect() as connection:
        submission = connection.execute(
            "select id from submissions where id = ?",
            (submission_id,),
        ).fetchone()
        if submission is None:
            raise RuntimeError(f"submission not found: {submission_id}")

        connection.execute(
            """
            update submissions
            set status = ?, valid = ?, runtime_sec = ?, invalid_reason = ?, completed_at = current_timestamp
            where id = ?
            """,
            (
                payload.status,
                1 if payload.valid else 0,
                payload.runtime_sec,
                payload.error,
                submission_id,
            ),
        )

        if payload.summary is not None:
            connection.execute(
                """
                insert into scores(
                    submission_id, mean_tm_score, mean_lddt, mean_rmsd, mean_ca_rmsd,
                    mean_gdt_ts_like, min_coverage, total_runtime_sec, raw_summary_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(submission_id) do update set
                    mean_tm_score=excluded.mean_tm_score,
                    mean_lddt=excluded.mean_lddt,
                    mean_rmsd=excluded.mean_rmsd,
                    mean_ca_rmsd=excluded.mean_ca_rmsd,
                    mean_gdt_ts_like=excluded.mean_gdt_ts_like,
                    min_coverage=excluded.min_coverage,
                    total_runtime_sec=excluded.total_runtime_sec,
                    raw_summary_json=excluded.raw_summary_json
                """,
                (
                    submission_id,
                    payload.summary.get("mean_tm_score"),
                    payload.summary.get("mean_lddt"),
                    payload.summary.get("mean_rmsd"),
                    payload.summary.get("mean_ca_rmsd"),
                    payload.summary.get("mean_gdt_ts_like"),
                    payload.summary.get("min_coverage"),
                    payload.runtime_sec,
                    json.dumps(payload.summary),
                ),
            )

        if payload.targets is not None:
            connection.execute(
                "delete from submission_targets where submission_id = ?",
                (submission_id,),
            )
            for target in payload.targets:
                connection.execute(
                    """
                    insert into submission_targets(
                        submission_id, target_id, valid, tm_score, lddt, rmsd,
                        ca_rmsd, gdt_ts_like, coverage, invalid_reason,
                        matched_residues, reference_residues
                    ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        submission_id,
                        target["target_id"],
                        1 if target["valid"] else 0,
                        target.get("tm_score"),
                        target.get("lddt"),
                        target.get("rmsd"),
                        target.get("ca_rmsd"),
                        target.get("gdt_ts_like"),
                        target.get("coverage"),
                        target.get("invalid_reason"),
                        target.get("matched_residues"),
                        target.get("reference_residues"),
                    ),
                )
        connection.commit()
    LOGGER.info(
        "submission persisted submission_id=%s status=%s valid=%s runtime_sec=%s has_summary=%s targets=%s",
        submission_id,
        payload.status,
        payload.valid,
        payload.runtime_sec,
        payload.summary is not None,
        len(payload.targets or []),
    )


def run_submission_job(submission_id: str, upload_path: Path) -> None:
    submission_logger, log_path = build_submission_logger(LOG_ROOT, submission_id)
    with JOB_STATE_LOCK:
        cancel_event = JOB_CANCEL_EVENTS.setdefault(submission_id, threading.Event())

    def register_process(process: subprocess.Popen[str]) -> None:
        with JOB_STATE_LOCK:
            JOB_PROCESSES[submission_id] = process
        submission_logger.info("benchmark process started pid=%s", process.pid)

    try:
        submission_logger.info("job started upload_path=%s", upload_path)
        sleep(0.8)
        if cancel_event.is_set():
            set_submission_status(submission_id, status="cancelled", valid=0, invalid_reason="cancelled by user")
            submission_logger.info("job cancelled before execution")
            return
        set_submission_status(submission_id, status="running")
        submission_logger.info("submission marked running")
        result = run_uploaded_submission(
            upload_path,
            cancel_event=cancel_event,
            process_started=register_process,
            submission_logger=submission_logger,
        )
        submission_logger.info(
            "benchmark finished returncode=%s runtime_sec=%s error=%s",
            result.get("returncode"),
            result.get("runtime_sec"),
            result.get("error"),
        )
        scoring = result.get("scoring")
        is_valid = bool(scoring["valid"]) if scoring else False
        persist_completion(
            submission_id,
            SubmissionResult(
                status="completed" if result.get("valid") else "failed",
                valid=is_valid,
                runtime_sec=float(result["runtime_sec"]) if result.get("runtime_sec") is not None else None,
                error=result.get("error"),
                summary=scoring["summary"] if scoring else None,
                targets=scoring["targets"] if scoring else None,
            ),
        )
        submission_logger.info(
            "job completed status=%s valid=%s score_targets=%s",
            "completed" if result.get("valid") else "failed",
            is_valid,
            len(scoring["targets"]) if scoring else 0,
        )
    except SubmissionCancelledError as exc:
        submission_logger.info("job cancelled error=%s", exc)
        set_submission_status(
            submission_id,
            status="cancelled",
            valid=0,
            invalid_reason=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        submission_logger.exception("job failed error=%s", exc)
        set_submission_status(
            submission_id,
            status="failed",
            valid=0,
            invalid_reason=f"{type(exc).__name__}: {exc}",
        )
    finally:
        with JOB_STATE_LOCK:
            JOB_PROCESSES.pop(submission_id, None)
            JOB_CANCEL_EVENTS.pop(submission_id, None)
        LOGGER.info("submission job finished submission_id=%s log_path=%s", submission_id, log_path)


def enqueue_submission(submission_id: str, upload_path: Path) -> None:
    with JOB_STATE_LOCK:
        JOB_CANCEL_EVENTS[submission_id] = threading.Event()
    thread = threading.Thread(
        target=run_submission_job,
        args=(submission_id, upload_path),
        daemon=True,
    )
    thread.start()
    LOGGER.info("submission enqueued submission_id=%s upload_path=%s", submission_id, upload_path)


def cancel_submission_job(submission_id: str) -> tuple[bool, str]:
    with connect() as connection:
        submission = connection.execute(
            "select status from submissions where id = ?",
            (submission_id,),
        ).fetchone()
    if submission is None:
        raise HTTPException(status_code=404, detail="submission not found")

    status = submission["status"]
    if status in {"completed", "failed", "cancelled"}:
        return False, status

    with JOB_STATE_LOCK:
        cancel_event = JOB_CANCEL_EVENTS.setdefault(submission_id, threading.Event())
        cancel_event.set()
        process = JOB_PROCESSES.get(submission_id)

    if process is not None:
        terminate_process_tree(process)
        LOGGER.info("cancellation signal sent submission_id=%s pid=%s", submission_id, process.pid)
        return True, "running"

    if status == "queued":
        set_submission_status(submission_id, status="cancelled", valid=0, invalid_reason="cancelled by user")
        LOGGER.info("queued submission cancelled submission_id=%s", submission_id)
        return True, "queued"

    LOGGER.info("cancellation requested waiting for process registration submission_id=%s status=%s", submission_id, status)
    return True, status


app = FastAPI(title="Sundai Protein Folding Bench API")
app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/", response_class=HTMLResponse)
def serve_index() -> str:
    return (WEB_ROOT / "index.html").read_text()


@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon() -> RedirectResponse:
    return RedirectResponse(url="/static/favicon.svg")


@app.post("/submissions")
def create_submission(payload: SubmissionCreate) -> dict:
    submission_id = str(uuid.uuid4())
    with connect() as connection:
        team_id, user_id = upsert_team(connection, payload.team_name, payload.created_by)
        connection.execute(
            """
            insert into submissions(
                id, team_id, created_by_user_id, status, storage_key, runtime_spec
            ) values (?, ?, ?, 'queued', ?, ?)
            """,
            (submission_id, team_id, user_id, payload.storage_key, payload.runtime_spec),
        )
        connection.commit()
    LOGGER.info("submission row created submission_id=%s team_name=%s created_by=%s storage_key=%s", submission_id, payload.team_name, payload.created_by, payload.storage_key)
    return {"submission_id": submission_id, "status": "queued"}


@app.post("/submissions/upload")
async def upload_submission(
    team_name: str = Form(...),
    created_by: str = Form("local@localhost"),
    file: UploadFile = File(...),
) -> dict:
    filename = file.filename or ""
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="submission file must end with .zip")

    submission_id = str(uuid.uuid4())
    safe_name = Path(filename).name
    dated_dir = UPLOAD_ROOT / datetime.now(UTC).strftime("%Y%m%d")
    dated_dir.mkdir(parents=True, exist_ok=True)
    storage_path = dated_dir / f"{submission_id}_{safe_name}"

    with storage_path.open("wb") as handle:
        copyfileobj(file.file, handle)
    await file.close()
    try:
        with ZipFile(storage_path) as archive:
            if "starter/train.py" not in archive.namelist():
                raise HTTPException(
                    status_code=400,
                    detail="zip must contain starter/train.py",
                )
    except BadZipFile as exc:
        storage_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"uploaded file is not a valid zip archive: {exc}") from exc
    LOGGER.info(
        "upload received submission_id=%s team_name=%s created_by=%s filename=%s stored_at=%s",
        submission_id,
        team_name,
        created_by,
        safe_name,
        storage_path,
    )
    config_payload = load_submission_config_from_zip(storage_path)
    runtime_spec = "docker/worker/runtime-spec.json"

    with connect() as connection:
        team_id, user_id = upsert_team(connection, team_name, created_by)
        connection.execute(
            """
            insert into submissions(
                id, team_id, created_by_user_id, status, storage_key, runtime_spec, original_filename, config_json
            ) values (?, ?, ?, 'queued', ?, ?, ?, ?)
            """,
            (
                submission_id,
                team_id,
                user_id,
                str(storage_path),
                runtime_spec,
                safe_name,
                json.dumps(config_payload, indent=2) if config_payload is not None else None,
            ),
        )
        connection.commit()
    enqueue_submission(submission_id, storage_path)
    return {
        "submission_id": submission_id,
        "status": "queued",
        "storage_key": str(storage_path),
        "filename": safe_name,
        "config": config_payload,
    }


@app.get("/submissions/{submission_id}")
def get_submission(submission_id: str) -> dict:
    with connect() as connection:
        submission = connection.execute(
            "select * from submissions where id = ?",
            (submission_id,),
        ).fetchone()
        if submission is None:
            raise HTTPException(status_code=404, detail="submission not found")
        score = connection.execute(
            "select * from scores where submission_id = ?",
            (submission_id,),
        ).fetchone()
        targets = connection.execute(
            "select * from submission_targets where submission_id = ? order by target_id",
            (submission_id,),
        ).fetchall()
    return {
        "submission": dict(submission),
        "score": dict(score) if score else None,
        "targets": [dict(row) for row in targets],
        "log_path": str(submission_log_path(LOG_ROOT, submission_id)),
        "submission_config": json.loads(submission["config_json"]) if submission["config_json"] else None,
    }


@app.post("/submissions/{submission_id}/cancel")
def cancel_submission(submission_id: str) -> dict:
    cancelled, previous_status = cancel_submission_job(submission_id)
    return {
        "submission_id": submission_id,
        "cancelled": cancelled,
        "previous_status": previous_status,
    }


@app.get("/submissions/{submission_id}/log")
def get_submission_log(submission_id: str) -> dict:
    path = submission_log_path(LOG_ROOT, submission_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="submission log not found")
    return {
        "submission_id": submission_id,
        "path": str(path),
        "contents": path.read_text(),
    }


@app.post("/internal/submissions/{submission_id}/complete")
def complete_submission(submission_id: str, payload: SubmissionResult) -> dict:
    with connect() as connection:
        submission = connection.execute(
            "select id from submissions where id = ?",
            (submission_id,),
        ).fetchone()
    if submission is None:
        raise HTTPException(status_code=404, detail="submission not found")
    persist_completion(submission_id, payload)
    return {"submission_id": submission_id, "status": payload.status}


@app.get("/runtime/config")
def runtime_config() -> dict:
    ready, missing = evaluator_assets_ready()
    return {
        "mode": "live-inference",
        "ready": ready,
        "missing": missing,
        "targets": ["7ftv_A", "8cny_A", "8g8r_A", "8i85_A"],
        "source_root": "/Users/ryanznie/Desktop/Important/Work/Sundai/bundles/public_lb_v1",
        "notes": "Using full public_lb_v1 backend bundle with live SimpleFold inference.",
    }


@app.get("/leaderboard")
def leaderboard() -> dict:
    with connect() as connection:
        rows = connection.execute(
            """
            with ranked as (
                select
                    teams.name as team_name,
                    submissions.id as submission_id,
                    submissions.status,
                    submissions.valid,
                    submissions.created_at,
                    submissions.completed_at,
                    scores.mean_tm_score,
                    scores.mean_lddt,
                    scores.mean_ca_rmsd,
                    scores.mean_gdt_ts_like,
                    scores.min_coverage,
                    scores.total_runtime_sec,
                    row_number() over (
                        partition by teams.id
                        order by
                            coalesce(submissions.valid, 0) desc,
                            scores.mean_tm_score desc,
                            scores.total_runtime_sec asc
                    ) as team_rank
                from submissions
                join teams on teams.id = submissions.team_id
                join scores on scores.submission_id = submissions.id
            )
            select *
            from ranked
            where team_rank = 1
            order by coalesce(valid, 0) desc, mean_tm_score desc, total_runtime_sec asc
            """
        ).fetchall()
    return {"rows": [dict(row) for row in rows]}
