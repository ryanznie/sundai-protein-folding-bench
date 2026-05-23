from __future__ import annotations

import logging
import os
from pathlib import Path


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def submission_log_path(log_root: Path, submission_id: str) -> Path:
    return log_root / f"{submission_id}.log"


def build_submission_logger(log_root: Path, submission_id: str) -> tuple[logging.Logger, Path]:
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = submission_log_path(log_root, submission_id)
    logger = logging.getLogger(f"sundai.submission.{submission_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = True

    abs_log_path = str(log_path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", "") == abs_log_path:
            return logger, log_path

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
    return logger, log_path


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser().resolve()
