#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../ecg-image-kit/codes/ecg-image-generator" && uv run python gen_ecg_images_from_data_batch.py \
     -i "${ECG_INPUT_DIR:-$SCRIPT_DIR/../fold10_images_for_xai}" \
     -o "$SCRIPT_DIR/../output/text" \
     -se 42 \
     --hw_text
