#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WRINKLES_DIR="$SCRIPT_DIR/../output/wrinkles"
if [ ! -d "$WRINKLES_DIR" ]; then
    echo "ERROR: wrinkles output not found at $WRINKLES_DIR — run wrinkles scenario first."
    exit 1
fi

cd "$SCRIPT_DIR/.." && uv run python "$SCRIPT_DIR/dewarp_ecg_batch.py" \
    "$WRINKLES_DIR" \
    "$SCRIPT_DIR/../output/dewarped"
