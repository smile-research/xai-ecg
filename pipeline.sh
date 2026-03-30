#!/bin/bash
# =============================================================================
# pipeline.sh — End-to-end research pipeline
#
# Stages:
#   1. Data preparation     — Download PTB-XL waveforms, prepare fold 10
#   2. ECG image generation — Render waveforms to images under multiple
#                             augmentation scenarios
#   3. Custom augmentation  — (Optional) Apply single_augmentation/ configs
#                             to arbitrary images
#   4. XAI explanations     — Generate saliency / GradientShap / Occlusion /
#                             LIME / IntegratedGradients for every model ×
#                             augmentation combination
#   5. Radiomics extraction — Extract PyRadiomics features from explanation
#                             maps
#   6. Statistical analysis — Aggregate features, compute IoU between
#                             explanation methods
#
# Usage:
#   bash pipeline.sh                        # run all stages
#   bash pipeline.sh --from-stage 4         # resume from stage 4
#   bash pipeline.sh --only-stage 3         # run only stage 3
#   bash pipeline.sh --stage3-config <json> # custom augmentation config
#   bash pipeline.sh --stage3-input  <dir>  # custom augmentation input dir
#   bash pipeline.sh --dry-run              # print what would run
#
# Prerequisites:
#   - uv (package manager)   — https://docs.astral.sh/uv/
#   - .env in explanation_generation/ with WANDB_API_KEY
#   - Python 3.10 (augmentation_generation, single_augmentation,
#     explanation_generation) and 3.12+ (radiomics)
# =============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Directories ──────────────────────────────────────────────────────────────
AUG_GEN_DIR="$ROOT_DIR/augmentation_generation"
SINGLE_AUG_DIR="$ROOT_DIR/single_augmentation"
EXPLAIN_DIR="$ROOT_DIR/explanation_generation"
RADIOMICS_DIR="$ROOT_DIR/radiomics"

# ── Defaults ─────────────────────────────────────────────────────────────────
FROM_STAGE=1
ONLY_STAGE=""
DRY_RUN=false
STAGE3_CONFIG="$SINGLE_AUG_DIR/configs/example_config.json"
STAGE3_INPUT="$AUG_GEN_DIR/output/clear"
STAGE3_OUTPUT="$SINGLE_AUG_DIR/output"

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from-stage)    FROM_STAGE="$2";    shift 2 ;;
        --only-stage)    ONLY_STAGE="$2";    shift 2 ;;
        --dry-run)       DRY_RUN=true;       shift   ;;
        --stage3-config) STAGE3_CONFIG="$2"; shift 2 ;;
        --stage3-input)  STAGE3_INPUT="$2";  shift 2 ;;
        --stage3-output) STAGE3_OUTPUT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^# =====/{ /^# =====/d; s/^# \?//p }' "$0"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

should_run() {
    local stage=$1
    if [[ -n "$ONLY_STAGE" ]]; then
        [[ "$stage" -eq "$ONLY_STAGE" ]]
    else
        [[ "$stage" -ge "$FROM_STAGE" ]]
    fi
}

run() {
    echo "  \$ $*"
    if [[ "$DRY_RUN" == false ]]; then
        "$@"
    fi
}

banner() {
    local stage=$1; shift
    echo ""
    echo "==========================================================="
    echo "  Stage $stage: $*"
    echo "==========================================================="
    echo ""
}

# =============================================================================
# Stage 1 — Data preparation
# =============================================================================
if should_run 1; then
    banner 1 "Data preparation"
    echo "Prepare PTB-XL waveform data and extract fold-10 records."
    echo ""

    cd "$AUG_GEN_DIR"
    run uv run python prepare_ptbxl_data.py
fi

# =============================================================================
# Stage 2 — ECG image generation (all augmentation scenarios)
# =============================================================================
if should_run 2; then
    banner 2 "ECG image generation"
    echo "Render fold-10 waveforms into images under each scenario:"
    echo "  all, clear, no_header, text, wrinkles, all_stacked, dewarped"
    echo ""
    echo "Output: $AUG_GEN_DIR/output/<scenario>/"
    echo ""

    cd "$AUG_GEN_DIR"
    run bash run_all_scenarios.sh
fi

# =============================================================================
# Stage 3 — Custom augmentation (single_augmentation/)
# =============================================================================
if should_run 3; then
    banner 3 "Custom augmentation"
    echo "Apply a user-defined augmentation pipeline to images."
    echo ""
    echo "  Config: $STAGE3_CONFIG"
    echo "  Input:  $STAGE3_INPUT"
    echo "  Output: $STAGE3_OUTPUT"
    echo ""

    cd "$SINGLE_AUG_DIR"
    run uv run python apply_augmentations.py \
        --input_dir "$STAGE3_INPUT" \
        --output_dir "$STAGE3_OUTPUT" \
        --config "$STAGE3_CONFIG"
fi

# =============================================================================
# Stage 4 — XAI explanation generation
# =============================================================================
if should_run 4; then
    banner 4 "XAI explanation generation"
    echo "For each model × augmentation combination, generate explanations"
    echo "using Saliency, GradientShap, Occlusion, LIME, IntegratedGradients."
    echo ""
    echo "Models:"
    echo "  - Model 1  (model-CCCC:v0)"
    echo "  - Model 2  (model-AAAA:v0)"
    echo "  - Model 5  (model-BBBB:v0)"
    echo ""
    echo "Output: $EXPLAIN_DIR/outputs/"
    echo ""

    # Verify W&B credentials
    if [[ ! -f "$EXPLAIN_DIR/.env" ]]; then
        echo "ERROR: $EXPLAIN_DIR/.env not found (needs WANDB_API_KEY)."
        exit 1
    fi

    cd "$EXPLAIN_DIR"
    run bash run_all.sh
fi

# =============================================================================
# Stage 5 — Radiomics feature extraction
# =============================================================================
if should_run 5; then
    banner 5 "Radiomics feature extraction"
    echo "Extract PyRadiomics features from explanation maps (LIME channel)."
    echo "Processes in batches of 200 samples."
    echo ""
    echo "Output: $RADIOMICS_DIR/output/"
    echo ""

    cd "$RADIOMICS_DIR"
    run bash run.sh
fi

# =============================================================================
# Stage 6 — Statistical analysis & IoU
# =============================================================================
if should_run 6; then
    banner 6 "Statistical analysis"
    echo "Aggregate radiomics CSVs, compute IoU between explanation methods"
    echo "across augmentation scenarios."
    echo ""
    echo "Output: $RADIOMICS_DIR/results/"
    echo ""

    cd "$RADIOMICS_DIR"
    run uv run python calculate_stats.py \
        --npz_dir   statystyki/explanations2 \
        --csv_dir   output \
        --output_dir results
fi

# =============================================================================
echo ""
echo "==========================================================="
echo "  Pipeline complete."
echo "==========================================================="
