#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
INPUT_DIR="${INPUT_DIR:-$ROOT_DIR/data/public_dev}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/output}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"

mkdir -p "$OUTPUT_DIR"

SUBMISSION_DIR="${SUBMISSION_DIR:-}"
if [ -z "$SUBMISSION_DIR" ]; then
  if [ -d "$ROOT_DIR/submission" ]; then
    SUBMISSION_DIR="$ROOT_DIR/submission"
  else
    SUBMISSION_DIR="$ROOT_DIR/baseline_submission"
  fi
fi

python3 "$ROOT_DIR/benchmark.py" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --submission "$SUBMISSION_DIR/train.py" \
  --config "$SUBMISSION_DIR/config.json" \
  --timeout_sec "$TIMEOUT_SEC"
