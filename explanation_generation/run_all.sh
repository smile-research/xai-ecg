#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUGMENTATION_DIR="$SCRIPT_DIR/../augmentation_generation/output"
DF="$SCRIPT_DIR/configs/xai_labels.csv"
OUTPUT_DIR="$SCRIPT_DIR/outputs"

# Optional range args: --from N --to M (default: 0-20)
FROM_INDEX="0"
TO_INDEX="100"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM_INDEX="$2"; shift 2 ;;
        --to)   TO_INDEX="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

RANGE_ARGS=""
[ -n "$FROM_INDEX" ] && RANGE_ARGS="$RANGE_ARGS --from_index $FROM_INDEX"
[ -n "$TO_INDEX"   ] && RANGE_ARGS="$RANGE_ARGS --to_index $TO_INDEX"

MODELS=(
    "your-entity/ecg-xai-models/model-AAAA:v0"
    "your-entity/ecg-xai-models/model-BBBB:v0"
    "your-entity/ecg-xai-models/model-CCCC:v0"
)

# Keys are the directory names under augmentation_generation/output/
# Values are the augmentation names used in output filenames
declare -A AUGMENTATIONS
AUGMENTATIONS["all"]="full"
AUGMENTATIONS["clear"]="clear_images"
AUGMENTATIONS["no_header"]="no_header"
AUGMENTATIONS["text"]="text"
AUGMENTATIONS["wrinkles"]="wrinkles"
AUGMENTATIONS["all_stacked"]="all_stacked"
AUGMENTATIONS["dewarped"]="dewarped"

mkdir -p "$SCRIPT_DIR/configs/generated"

for MODEL in "${MODELS[@]}"; do
    MODEL_SLUG=$(echo "$MODEL" | tr '/' '_')
    for AUG_DIR in "${!AUGMENTATIONS[@]}"; do
        AUG_NAME="${AUGMENTATIONS[$AUG_DIR]}"
        DATA_DIR="$AUGMENTATION_DIR/$AUG_DIR/"
        CONFIG_FILE="$SCRIPT_DIR/configs/generated/${MODEL_SLUG}_${AUG_NAME}.json"

        echo "=== Model: $MODEL | Augmentation: $AUG_NAME ==="

        cat > "$CONFIG_FILE" <<EOF
{
    "data_module_config": {
        "data_dir": "$DATA_DIR",
        "batch_size": 16,
        "num_workers": 16
    },
    "dataset": {
        "df": "$DF",
        "folds_pred": {
            "test": [10]
        }
    },
    "wandb_config": {
        "entity": "your-entity",
        "project": "ecg-xai-inference"
    },
    "model": "$MODEL"
}
EOF

        uv run python "$SCRIPT_DIR/generate_explanations.py" \
            --config_file "$CONFIG_FILE" \
            --output_dir "$OUTPUT_DIR" \
            $RANGE_ARGS
        echo "=== Done: $MODEL | $AUG_NAME ==="
    done
done
