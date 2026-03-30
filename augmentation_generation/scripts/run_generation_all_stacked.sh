#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR/../ecg-image-kit/codes/ecg-image-generator" && uv run python gen_ecg_images_from_data_batch.py \
     -i "${ECG_INPUT_DIR:-$SCRIPT_DIR/../fold10_images_for_xai}" \
     -o "$SCRIPT_DIR/../output/all_stacked" \
     -se 42 \
     --print_header \
     --store_config 2 \
     --random_resolution \
     --random_padding \
     --pad_inches 0 \
     --calibration_pulse 1 \
     --random_grid_present 1 \
     --random_print_header 1 \
     --random_bw 1 \
     --random_grid_color \
     --rotate 7 \
     --wrinkles \
     --hw_text
