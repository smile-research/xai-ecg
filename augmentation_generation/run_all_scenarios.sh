#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$SCRIPT_DIR/fold10_images_for_xai"

copy_missing_records() {
    local output_dir="$1"
    local temp_dir="$2"
    local copied=0
    for hea_file in "$INPUT_DIR"/*.hea; do
        [ -f "$hea_file" ] || continue
        stem=$(basename "${hea_file%.hea}")
        if ! compgen -G "$output_dir/${stem}*.png" > /dev/null 2>&1; then
            cp "$INPUT_DIR/${stem}.hea" "$temp_dir/"
            cp "$INPUT_DIR/${stem}.dat" "$temp_dir/"
            ((copied++))
        fi
    done
    echo "$copied"
}

for TYPE in all clear no_header text wrinkles all_stacked dewarped; do
    echo "=== Preparing: $TYPE ==="
    OUTPUT_DIR="$SCRIPT_DIR/output/$TYPE"
    mkdir -p "$OUTPUT_DIR"

    # dewarped is derived from wrinkles output (PNGs), not from raw .hea/.dat files
    if [ "$TYPE" = "dewarped" ]; then
        WRINKLES_DIR="$SCRIPT_DIR/output/wrinkles"
        missing=0
        for png in "$WRINKLES_DIR"/*.png; do
            [ -f "$png" ] || continue
            stem=$(basename "$png")
            [ ! -f "$OUTPUT_DIR/$stem" ] && ((missing++))
        done
        if [ "$missing" -eq 0 ]; then
            echo "=== Skipping: dewarped (all outputs already exist) ==="
            continue
        fi
        echo "=== Running: dewarped ($missing new records) ==="
        bash "$SCRIPT_DIR/scripts/run_generation_dewarped.sh"
        echo "=== Done: dewarped ==="
        continue
    fi

    TEMP_DIR="$SCRIPT_DIR/tmp_$TYPE"
    rm -rf "$TEMP_DIR"
    mkdir -p "$TEMP_DIR"

    copied=$(copy_missing_records "$OUTPUT_DIR" "$TEMP_DIR")

    if [ "$copied" -eq 0 ]; then
        echo "=== Skipping: $TYPE (all outputs already exist) ==="
        rm -rf "$TEMP_DIR"
        continue
    fi

    echo "=== Running: $TYPE ($copied new records) ==="
    export ECG_INPUT_DIR="$TEMP_DIR"
    bash "$SCRIPT_DIR/scripts/run_generation_${TYPE}.sh"
    unset ECG_INPUT_DIR
    rm -rf "$TEMP_DIR"

    echo "=== Done: $TYPE ==="
done
