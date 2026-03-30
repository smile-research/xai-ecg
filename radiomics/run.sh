#!/bin/bash
set -euo pipefail

PYTHON="uv run python"
SCRIPT="$(dirname "$0")/main.py"
OUTPUT_DIR="$(dirname "$0")/output"
CHUNK=200
TOTAL=1800
MAX_JOBS=${MAX_JOBS:-5}

mkdir -p "$OUTPUT_DIR"

# Build list of chunk ranges and run in parallel
seq 0 $CHUNK $((TOTAL - 1)) | xargs -P "$MAX_JOBS" -I{} bash -c '
    from={}; to=$((from + '"$CHUNK"'))
    echo "Running $from - $to ..."
    '"$PYTHON"' "'"$SCRIPT"'" -f "$from" -t "$to" -p "'"$OUTPUT_DIR"'"
    echo "Done $from - $to"
'

echo "All done."
